"""Loopback OpenAI-style HTTP adapter for the AC922 JSON-line worker.

The adapter does not log request bodies or prompts. Worker stdout/stderr are
kept in a small rotating file under AC922_RUN_DIR; the worker protocol itself
uses stdin and a dedicated inherited event descriptor.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


LOG = logging.getLogger("ac922_api")
MAX_PROMPT_BYTES = 4 << 20
MAX_HTTP_BODY_BYTES = MAX_PROMPT_BYTES
MAX_ADVERTISED_CONTEXT = 262140
EXPECTED_MODEL = "flash-next-iq3-xxs"
MAX_QUEUE = 16
MAX_WAVE = 4
COALESCE_SECONDS = 0.030
REQUEST_DEADLINE_SECONDS = 1800.0
STREAM_QUEUE_DEPTH = 256
SSE_HEARTBEAT_SECONDS = 15.0
WORKER_LOG_BYTES = 4 << 20
WORKER_LOG_BACKUPS = 2
DEFAULT_MAX_TOKENS = 512
MAX_WAVE_COMMAND_BYTES = 7 << 20


class ApiFault(Exception):
    def __init__(self, status: int, code: str, message: str, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.error_type = error_type


@dataclass(frozen=True)
class ReadyInfo:
    max_context: int
    pool_tokens: int
    max_parallel: int
    model: str


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    public_id: str
    kind: str
    model: str
    prompt: str | None
    messages: list[dict[str, str]] | None
    max_tokens: int
    stream: bool
    raw: bool
    created: int

    def worker_request(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "id": self.request_id,
            "max_tokens": self.max_tokens,
            "raw": self.raw,
        }
        if self.messages is not None:
            request["messages"] = self.messages
        else:
            request["prompt"] = self.prompt or ""
        return request


@dataclass
class Outcome:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    error: ApiFault | None = None


@dataclass
class RequestJob:
    spec: RequestSpec
    enqueued_at: float
    deadline: float
    result: asyncio.Future[Outcome]
    chunks: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=STREAM_QUEUE_DEPTH))
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    decoder: Any = field(default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace"))
    text_parts: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    reserved_tokens: int | None = None
    output_tokens_seen: int = 0
    state: str = "queued"
    wave_id: str | None = None
    cancel_sent: bool = False
    stream_overflow: bool = False
    pool_overflow: bool = False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_request(payload: Any, kind: str, model: str, max_context: int) -> RequestSpec:
    """Validate the supported, intentionally small OpenAI request subset."""
    if not isinstance(payload, dict):
        raise ApiFault(400, "invalid_json", "Request body must be a JSON object.")
    allowed = {"model", "temperature", "top_p", "max_tokens", "n", "stream"}
    if kind == "chat":
        allowed.add("messages")
    elif kind == "completion":
        allowed.add("prompt")
    else:
        raise ValueError(f"unknown API kind: {kind}")
    unknown = set(payload) - allowed
    if unknown:
        name = sorted(str(key) for key in unknown)[0]
        raise ApiFault(400, "unsupported_parameter", f"Unsupported request parameter: {name}.")

    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or requested_model != model:
        raise ApiFault(400, "model_not_found", "The requested model is not available.")

    temperature = payload.get("temperature", 0)
    if not _is_number(temperature) or float(temperature) != 0.0:
        raise ApiFault(400, "unsupported_parameter", "Only temperature=0 is supported.")
    top_p = payload.get("top_p", 1)
    if not _is_number(top_p) or float(top_p) != 1.0:
        raise ApiFault(400, "unsupported_parameter", "Only top_p=1 is supported.")
    n = payload.get("n", 1)
    if not _is_int(n) or n != 1:
        raise ApiFault(400, "unsupported_parameter", "Only n=1 is supported.")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ApiFault(400, "invalid_request", "stream must be a boolean.")
    max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    if not _is_int(max_tokens) or max_tokens < 1 or max_tokens > 8192:
        raise ApiFault(400, "invalid_request", "max_tokens must be an integer from 1 through 8192.")
    if max_tokens > max_context:
        raise ApiFault(400, "context_limit", "max_tokens exceeds the worker's advertised context size.")

    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    if kind == "completion":
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ApiFault(400, "invalid_request", "prompt must be a string.")
        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > MAX_PROMPT_BYTES:
            raise ApiFault(413, "prompt_too_large", "Prompt exceeds the 4 MiB limit.", "request_too_large")
    else:
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ApiFault(400, "invalid_request", "messages must be a non-empty array.")
        if len(raw_messages) > 256:
            raise ApiFault(400, "invalid_request", "messages may contain at most 256 entries.")
        messages = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ApiFault(400, "invalid_message", f"messages[{index}] must contain only role and string content.")
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ApiFault(400, "invalid_message", f"messages[{index}] has an unsupported role or non-string content.")
            messages.append({"role": role, "content": content})
        encoded_messages = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded_messages) > MAX_PROMPT_BYTES:
            raise ApiFault(413, "prompt_too_large", "Combined message content exceeds the 4 MiB limit.", "request_too_large")

    request_id = str(uuid.uuid4())
    spec = RequestSpec(
        request_id=request_id,
        public_id=("chatcmpl-" if kind == "chat" else "cmpl-") + request_id,
        kind=kind,
        model=model,
        prompt=prompt,
        messages=messages,
        max_tokens=max_tokens,
        stream=stream,
        raw=(kind == "completion"),
        created=int(time.time()),
    )
    command_size = len(json.dumps(spec.worker_request(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if command_size + 256 > MAX_WAVE_COMMAND_BYTES:
        raise ApiFault(413, "prompt_too_large", "Encoded request is too large for the worker protocol.", "request_too_large")
    return spec


class BoundedWorkerLog:
    """Small append-only worker stdout/stderr log with two bounded rotations."""

    def __init__(self, path: Path, max_bytes: int = WORKER_LOG_BYTES, backups: int = WORKER_LOG_BACKUPS):
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(self.fd, 0o600)
        self.size = os.fstat(self.fd).st_size

    def _rotate(self) -> None:
        os.close(self.fd)
        for index in range(self.backups, 1, -1):
            older = self.path.with_name(f"{self.path.name}.{index - 1}")
            newer = self.path.with_name(f"{self.path.name}.{index}")
            if older.exists():
                os.replace(older, newer)
        first = self.path.with_name(f"{self.path.name}.1")
        if self.path.exists():
            os.replace(self.path, first)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(self.fd, 0o600)
        self.size = 0

    def write(self, data: bytes) -> None:
        if not data:
            return
        if len(data) > self.max_bytes:
            data = data[-self.max_bytes :]
        if self.size + len(data) > self.max_bytes:
            self._rotate()
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]
        self.size += len(data)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class SubprocessWorker:
    """Owns the worker process, JSON stdin, event FD, and bounded output log."""

    def __init__(self, command_json: str | None = None, run_dir: str | None = None):
        self.command_json = command_json or os.environ.get("AC922_WORKER_COMMAND_JSON", "")
        self.run_dir = Path(run_dir or os.environ.get("AC922_RUN_DIR", "/run/ac922-api"))
        self.process: asyncio.subprocess.Process | None = None
        self._event_transport: asyncio.Transport | None = None
        self._send_lock = asyncio.Lock()
        self._log_sink: BoundedWorkerLog | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._on_event: Callable[[dict[str, Any]], Any] | None = None
        self._on_exit: Callable[[int | None, str], Any] | None = None
        self._exit_reported = False
        self._stopping = False

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _load_command(self) -> tuple[list[str], dict[str, str]]:
        if not self.command_json:
            raise RuntimeError("AC922_WORKER_COMMAND_JSON is not configured")
        with open(self.command_json, "r", encoding="utf-8") as source:
            config = json.load(source)
        if not isinstance(config, dict) or set(config) != {"argv", "env"}:
            raise RuntimeError("worker command file must contain exactly argv and env")
        argv, extra_env = config["argv"], config["env"]
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise RuntimeError("worker argv must be a non-empty array of strings")
        if not isinstance(extra_env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in extra_env.items()):
            raise RuntimeError("worker env must map strings to strings")
        return argv, extra_env

    async def start(self, on_event: Callable[[dict[str, Any]], Any], on_exit: Callable[[int | None, str], Any]) -> None:
        self._on_event = on_event
        self._on_exit = on_exit
        argv, extra_env = self._load_command()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_sink = BoundedWorkerLog(self.run_dir / "worker.log")
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        child_env = os.environ.copy()
        child_env.update(extra_env)
        child_env["AC922_API_EVENT_FD"] = str(write_fd)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
                pass_fds=(write_fd,),
            )
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            self._log_sink.close()
            self._log_sink = None
            raise
        os.close(write_fd)
        event_file = os.fdopen(read_fd, "rb", buffering=0)
        event_reader = asyncio.StreamReader(limit=1 << 20)
        event_protocol = asyncio.StreamReaderProtocol(event_reader)
        loop = asyncio.get_running_loop()
        self._event_transport, _ = await loop.connect_read_pipe(lambda: event_protocol, event_file)
        self._tasks = [
            asyncio.create_task(self._read_events(event_reader), name="ac922-worker-events"),
            asyncio.create_task(self._capture_output(), name="ac922-worker-log"),
            asyncio.create_task(self._watch_process(), name="ac922-worker-exit"),
        ]

    async def send(self, command: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.returncode is not None or process.stdin is None:
            raise RuntimeError("worker process is not running")
        data = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self._send_lock:
            process.stdin.write(data)
            await process.stdin.drain()

    async def _read_events(self, reader: asyncio.StreamReader) -> None:
        reason = "worker event pipe closed"
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    reason = "worker emitted malformed event JSON"
                    break
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    reason = "worker emitted an invalid event object"
                    break
                callback = self._on_event
                if callback is not None:
                    await callback(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = f"worker event reader failed ({type(exc).__name__})"
        if not self._stopping and self.alive:
            await self._report_exit(None, reason)
            await self._terminate()

    async def _capture_output(self) -> None:
        process, sink = self.process, self._log_sink
        if process is None or process.stdout is None or sink is None:
            return
        try:
            while True:
                chunk = await process.stdout.read(64 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("worker output capture failed (%s)", type(exc).__name__)

    async def _watch_process(self) -> None:
        process = self.process
        if process is None:
            return
        code = await process.wait()
        if not self._stopping:
            await self._report_exit(code, "worker process exited")

    async def _report_exit(self, code: int | None, reason: str) -> None:
        if self._exit_reported:
            return
        self._exit_reported = True
        callback = self._on_exit
        if callback is not None:
            await callback(code, reason)

    async def _terminate(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def stop(self) -> None:
        self._stopping = True
        if self.alive:
            try:
                await self.send({"type": "shutdown"})
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)  # type: ignore[union-attr]
            except asyncio.TimeoutError:
                await self._terminate()
        if self._event_transport is not None:
            self._event_transport.close()
            self._event_transport = None
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._log_sink is not None:
            self._log_sink.close()
            self._log_sink = None


class InferenceService:
    """Bounded queue and single-wave dispatcher around one worker process."""

    def __init__(
        self,
        backend: Any | None = None,
        *,
        max_queue: int = MAX_QUEUE,
        max_wave: int = MAX_WAVE,
        coalesce_seconds: float = COALESCE_SECONDS,
        request_deadline_seconds: float = REQUEST_DEADLINE_SECONDS,
    ):
        self.backend = backend or SubprocessWorker()
        self.max_queue = max_queue
        self.max_wave = max_wave
        self.coalesce_seconds = coalesce_seconds
        self.request_deadline_seconds = request_deadline_seconds
        self._condition = asyncio.Condition()
        self._queued: deque[RequestJob] = deque()
        self._jobs: dict[str, RequestJob] = {}
        self._active: dict[str, RequestJob] = {}
        self._waves: dict[str, asyncio.Future[bool]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._alive = False
        self._ready: ReadyInfo | None = None
        self._stopping = False
        self._startup_error: str | None = None
        self._failure_exit_task: asyncio.Task[None] | None = None

    @property
    def ready_info(self) -> ReadyInfo | None:
        return self._ready if self._alive and not self._stopping else None

    @property
    def is_alive(self) -> bool:
        return self._alive and not self._stopping

    @property
    def queue_depth(self) -> int:
        return len(self._queued)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def unready_reason(self) -> str:
        if self._stopping:
            return "stopping"
        if not self._alive:
            return self._startup_error or "worker_not_running"
        return "worker_not_ready"

    async def start(self) -> None:
        self._stopping = False
        try:
            await self.backend.start(self._on_event, self._on_exit)
            self._alive = bool(self.backend.alive)
            if not self._alive:
                self._startup_error = "worker_not_running"
        except Exception as exc:
            self._alive = False
            self._ready = None
            self._startup_error = f"worker_start_failed_{type(exc).__name__}"
            LOG.error("worker start failed (%s)", type(exc).__name__)
            self._schedule_worker_failure_exit()
        self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="ac922-wave-dispatcher")

    async def stop(self) -> None:
        async with self._condition:
            self._stopping = True
            self._ready = None
            for job in list(self._jobs.values()):
                self._finish_locked(job, Outcome(error=ApiFault(503, "service_stopping", "Inference service is stopping.", "server_error")))
            self._queued.clear()
            for wave in self._waves.values():
                if not wave.done():
                    wave.set_result(False)
            self._condition.notify_all()
        try:
            await self.backend.stop()
        finally:
            self._alive = False
            failure_exit_task = self._failure_exit_task
            if failure_exit_task is not None and not failure_exit_task.done():
                failure_exit_task.cancel()
                await asyncio.gather(failure_exit_task, return_exceptions=True)
            self._failure_exit_task = None
            dispatcher = self._dispatcher
            if dispatcher is not None and not dispatcher.done():
                dispatcher.cancel()
                await asyncio.gather(dispatcher, return_exceptions=True)
            self._dispatcher = None

    def _finish_locked(self, job: RequestJob, outcome: Outcome) -> None:
        if job.result.done():
            return
        job.state = "done"
        self._jobs.pop(job.spec.request_id, None)
        self._active.pop(job.spec.request_id, None)
        job.result.set_result(outcome)
        job.wake.set()

    async def submit(self, spec: RequestSpec) -> RequestJob:
        async with self._condition:
            if self.ready_info is None:
                raise ApiFault(503, "worker_unready", "Inference worker is not ready.", "server_error")
            if len(self._queued) >= self.max_queue:
                raise ApiFault(429, "queue_full", "Inference queue is full; retry later.", "busy_error")
            loop = asyncio.get_running_loop()
            now = loop.time()
            job = RequestJob(
                spec=spec,
                enqueued_at=now,
                deadline=now + self.request_deadline_seconds,
                result=loop.create_future(),
            )
            self._queued.append(job)
            self._jobs[spec.request_id] = job
            self._condition.notify_all()
            LOG.info("request queued id=%s kind=%s stream=%s queue=%d", spec.request_id, spec.kind, spec.stream, len(self._queued))
            return job

    async def cancel(self, job: RequestJob, reason: str) -> None:
        send_cancel = False
        async with self._condition:
            if job.result.done():
                return
            if job.state in {"queued", "dispatching", "retry"}:
                try:
                    self._queued.remove(job)
                except ValueError:
                    pass
                self._finish_locked(job, Outcome(error=ApiFault(499, "request_cancelled", reason, "request_error")))
                self._condition.notify_all()
                LOG.info("queued request cancelled id=%s reason=%s queue=%d", job.spec.request_id, reason, len(self._queued))
            elif job.state == "active" and not job.cancel_sent:
                job.cancel_sent = True
                send_cancel = True
        if send_cancel:
            try:
                await self.backend.send({"type": "cancel", "id": job.spec.request_id})
                LOG.info("active request cancellation sent id=%s reason=%s", job.spec.request_id, reason)
            except Exception as exc:
                await self._on_exit(None, f"cancel_send_failed_{type(exc).__name__}")

    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._condition:
                while not self._queued and not self._stopping:
                    await self._condition.wait()
                if self._stopping:
                    return
                first_deadline = self._queued[0].enqueued_at + self.coalesce_seconds
            await asyncio.sleep(max(0.0, first_deadline - loop.time()))

            async with self._condition:
                now = loop.time()
                batch: list[RequestJob] = []
                wave_id = str(uuid.uuid4())
                while self._queued and len(batch) < self.max_wave:
                    job = self._queued[0]
                    if job.result.done():
                        self._queued.popleft()
                        continue
                    if now >= job.deadline:
                        self._queued.popleft()
                        self._finish_locked(job, Outcome(error=ApiFault(504, "request_timeout", "Request expired while queued.", "timeout_error")))
                        continue
                    candidate = batch + [job]
                    candidate_command = {
                        "type": "wave",
                        "id": wave_id,
                        "requests": [item.spec.worker_request() for item in candidate],
                    }
                    command_size = len(json.dumps(candidate_command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
                    if command_size > MAX_WAVE_COMMAND_BYTES:
                        if batch:
                            break
                        self._queued.popleft()
                        self._finish_locked(job, Outcome(error=ApiFault(413, "prompt_too_large", "Request does not fit the worker command size limit.", "request_too_large")))
                        continue
                    self._queued.popleft()
                    job.state = "dispatching"
                    batch.append(job)
                if not batch:
                    continue
            await self._run_wave(batch)

    async def _run_wave(self, batch: list[RequestJob]) -> None:
        """Run one fixed wave, recursively halving only on pool-overflow rejection."""
        loop = asyncio.get_running_loop()
        async with self._condition:
            active_batch: list[RequestJob] = []
            now = loop.time()
            for job in batch:
                if job.result.done():
                    continue
                if now >= job.deadline:
                    self._finish_locked(job, Outcome(error=ApiFault(504, "request_timeout", "Request expired before worker admission.", "timeout_error")))
                    continue
                job.state = "active"
                job.pool_overflow = False
                active_batch.append(job)
            if not active_batch:
                return

            wave_id = str(uuid.uuid4())
            wave_done = loop.create_future()
            self._waves[wave_id] = wave_done
            for job in active_batch:
                job.wave_id = wave_id
                self._active[job.spec.request_id] = job
            self._condition.notify_all()

        command = {"type": "wave", "id": wave_id, "requests": [job.spec.worker_request() for job in active_batch]}
        pool_overflow = False
        try:
            await self.backend.send(command)
            pool_overflow = await asyncio.wait_for(asyncio.shield(wave_done), timeout=self.request_deadline_seconds + 5.0)
        except asyncio.TimeoutError:
            await self._on_exit(None, "wave_deadline_exceeded")
            try:
                await self.backend.stop()
            except Exception:
                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._on_exit(None, f"worker_send_failed_{type(exc).__name__}")
        finally:
            async with self._condition:
                self._waves.pop(wave_id, None)

        if not pool_overflow:
            return
        retry = [job for job in active_batch if not job.result.done()]
        if len(retry) <= 1:
            async with self._condition:
                for job in retry:
                    self._finish_locked(job, Outcome(error=ApiFault(
                        400,
                        "context_pool_exceeded",
                        "This request exceeds the worker's available shared context pool.",
                        "invalid_request_error",
                    )))
                self._condition.notify_all()
            return

        midpoint = (len(retry) + 1) // 2
        LOG.info("wave split after shared-pool rejection count=%d left=%d right=%d", len(retry), midpoint, len(retry) - midpoint)
        await self._run_wave(retry[:midpoint])
        await self._run_wave(retry[midpoint:])

    async def _on_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "ready":
            info = self._parse_ready(event)
            if info is None:
                await self._on_exit(None, "invalid_worker_ready_event")
                return
            async with self._condition:
                self._ready = info
                self._alive = bool(self.backend.alive)
                self._startup_error = None
                self._condition.notify_all()
            LOG.info("worker ready model=%s max_context=%d pool_tokens=%d max_parallel=%d", info.model, info.max_context, info.pool_tokens, info.max_parallel)
            return
        if event_type == "wave_done":
            wave_id = event.get("id")
            if not isinstance(wave_id, str):
                await self._on_exit(None, "invalid_worker_wave_done_event")
                return
            async with self._condition:
                wave_jobs = [job for job in self._active.values() if job.wave_id == wave_id and not job.result.done()]
                wave = self._waves.get(wave_id)
                if wave is not None and not wave.done():
                    safe_pool_rejection = bool(wave_jobs) and all(
                        job.pool_overflow and job.input_tokens is None and job.output_tokens_seen == 0
                        for job in wave_jobs
                    )
                    if safe_pool_rejection:
                        for job in wave_jobs:
                            if job.cancel_sent:
                                self._finish_locked(job, Outcome(error=ApiFault(499, "request_cancelled", "Request was cancelled before worker admission.", "request_error")))
                            else:
                                job.state = "retry"
                                self._active.pop(job.spec.request_id, None)
                        wave.set_result(True)
                    else:
                        for job in wave_jobs:
                            self._finish_locked(job, Outcome(error=ApiFault(
                                502, "worker_protocol_error", "Worker ended a wave before completing this request.", "server_error"
                            )))
                        wave.set_result(False)
                self._condition.notify_all()
            return

        if event_type not in {"accepted", "token", "done", "error"}:
            await self._on_exit(None, "unknown_worker_event_type")
            return

        request_id = event.get("id")
        if not isinstance(request_id, str):
            await self._on_exit(None, "worker_event_missing_request_id")
            return
        cancel_overflow: RequestJob | None = None
        async with self._condition:
            job = self._active.get(request_id)
            if job is None or job.result.done():
                return
            if event_type == "accepted":
                input_tokens = event.get("input_tokens")
                reserved_tokens = event.get("reserved_tokens")
                if not _is_int(input_tokens) or input_tokens < 0 or not _is_int(reserved_tokens) or reserved_tokens < 0:
                    self._finish_locked(job, Outcome(error=ApiFault(502, "worker_protocol_error", "Worker returned invalid admission counts.", "server_error")))
                elif job.input_tokens is not None:
                    self._finish_locked(job, Outcome(error=ApiFault(502, "worker_protocol_error", "Worker accepted a request more than once.", "server_error")))
                else:
                    job.input_tokens = input_tokens
                    job.reserved_tokens = reserved_tokens
            elif event_type == "token":
                piece_bytes = event.get("piece_bytes")
                token_id = event.get("token_id")
                if job.input_tokens is None or not _is_int(token_id) or not isinstance(piece_bytes, list) or any(not _is_int(b) or b < 0 or b > 255 for b in piece_bytes):
                    self._finish_locked(job, Outcome(error=ApiFault(502, "worker_protocol_error", "Worker returned an invalid token event.", "server_error")))
                else:
                    text = job.decoder.decode(bytes(piece_bytes), final=False)
                    job.output_tokens_seen += 1
                    if job.spec.stream:
                        if text and not job.stream_overflow:
                            try:
                                job.chunks.put_nowait(text)
                            except asyncio.QueueFull:
                                job.stream_overflow = True
                                cancel_overflow = job
                            job.wake.set()
                    else:
                        job.text_parts.append(text)
            elif event_type == "error":
                status = event.get("status", 502)
                if not _is_int(status) or status not in {400, 413, 429, 503}:
                    status = 502
                message = event.get("message", "Worker request failed.")
                if not isinstance(message, str):
                    message = "Worker request failed."
                message = " ".join(message.split())[:500] or "Worker request failed."
                if status == 400 and message == "shared context pool exceeded" and job.input_tokens is None and job.output_tokens_seen == 0:
                    job.pool_overflow = True
                else:
                    error_type = "busy_error" if status == 429 else "server_error" if status >= 500 else "invalid_request_error"
                    self._finish_locked(job, Outcome(error=ApiFault(status, "worker_error", message, error_type)))
            elif event_type == "done":
                finish_reason = event.get("finish_reason")
                input_tokens = event.get("input_tokens", job.input_tokens or 0)
                output_tokens = event.get("output_tokens")
                if finish_reason not in {"stop", "length", "cancelled"} or not _is_int(input_tokens) or input_tokens < 0 or not _is_int(output_tokens) or output_tokens < 0:
                    self._finish_locked(job, Outcome(error=ApiFault(502, "worker_protocol_error", "Worker returned invalid completion metadata.", "server_error")))
                else:
                    tail = job.decoder.decode(b"", final=True)
                    if job.spec.stream:
                        if tail and not job.stream_overflow:
                            try:
                                job.chunks.put_nowait(tail)
                            except asyncio.QueueFull:
                                job.stream_overflow = True
                    else:
                        job.text_parts.append(tail)
                    if job.stream_overflow:
                        outcome = Outcome(error=ApiFault(429, "stream_consumer_too_slow", "Client is not consuming the token stream quickly enough.", "busy_error"))
                    else:
                        outcome = Outcome(
                            text="".join(job.text_parts),
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            finish_reason=finish_reason,
                        )
                    self._finish_locked(job, outcome)
            job.wake.set()
            self._condition.notify_all()
        if cancel_overflow is not None:
            await self.cancel(cancel_overflow, "stream_consumer_too_slow")

    @staticmethod
    def _parse_ready(event: dict[str, Any]) -> ReadyInfo | None:
        max_context = event.get("max_context")
        pool_tokens = event.get("pool_tokens")
        max_parallel = event.get("max_parallel")
        model = event.get("model")
        if (
            not _is_int(max_context) or max_context < 1
            or not _is_int(pool_tokens) or pool_tokens < 1
            or not _is_int(max_parallel) or max_parallel != MAX_WAVE
            or not isinstance(model, str) or model != EXPECTED_MODEL
        ):
            return None
        return ReadyInfo(min(max_context, MAX_ADVERTISED_CONTEXT), pool_tokens, max_parallel, model)

    async def _on_exit(self, code: int | None, reason: str) -> None:
        async with self._condition:
            if not self._alive and self._ready is None and not self._jobs:
                self._startup_error = reason
            else:
                self._alive = False
                self._ready = None
                self._startup_error = reason
                failure = Outcome(error=ApiFault(503, "worker_unavailable", "Inference worker exited; pending requests were failed.", "server_error"))
                for job in list(self._jobs.values()):
                    self._finish_locked(job, failure)
                self._queued.clear()
                for wave in self._waves.values():
                    if not wave.done():
                        wave.set_result(False)
                self._condition.notify_all()
        LOG.error("worker unavailable exit_code=%s reason=%s", code, reason)
        self._schedule_worker_failure_exit()

    def _schedule_worker_failure_exit(self) -> None:
        if self._stopping or not isinstance(self.backend, SubprocessWorker):
            return
        if os.environ.get("AC922_EXIT_ON_WORKER_FAILURE") != "1":
            return
        if self._failure_exit_task is None or self._failure_exit_task.done():
            self._failure_exit_task = asyncio.create_task(self._delayed_failure_exit(), name="ac922-worker-failure-exit")

    async def _delayed_failure_exit(self) -> None:
        await asyncio.sleep(1.0)
        if not self._stopping:
            LOG.critical("worker failed; exiting with status 70 for supervisor restart")
            os._exit(70)

    async def wait_result(self, job: RequestJob, request: Any) -> Outcome:
        while not job.result.done():
            if await request.is_disconnected():
                await self.cancel(job, "client_disconnected")
                raise ApiFault(499, "client_disconnected", "Client disconnected.", "request_error")
            remaining = job.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                await self.cancel(job, "request_deadline_exceeded")
                raise ApiFault(504, "request_timeout", "Request exceeded the 1800-second deadline.", "timeout_error")
            try:
                await asyncio.wait_for(asyncio.shield(job.result), timeout=min(0.25, remaining))
            except asyncio.TimeoutError:
                continue
        outcome = job.result.result()
        if outcome.error is not None:
            raise outcome.error
        return outcome


async def _read_json_limited(request: Any) -> Any:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_HTTP_BODY_BYTES:
                raise ApiFault(413, "request_too_large", "JSON request body exceeds the size limit.", "request_too_large")
        except ValueError:
            raise ApiFault(400, "invalid_content_length", "Invalid Content-Length header.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_HTTP_BODY_BYTES:
            raise ApiFault(413, "request_too_large", "JSON request body exceeds the size limit.", "request_too_large")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ApiFault(400, "invalid_json", "Request body is not valid JSON.")


def _usage(outcome: Outcome) -> dict[str, int]:
    return {
        "prompt_tokens": outcome.input_tokens,
        "completion_tokens": outcome.output_tokens,
        "total_tokens": outcome.input_tokens + outcome.output_tokens,
    }


def _chunk_payload(spec: RequestSpec, model: str, content: str | None = None, finish_reason: str | None = None, role: bool = False) -> dict[str, Any]:
    if spec.kind == "chat":
        delta: dict[str, str] = {}
        if role:
            delta["role"] = "assistant"
        if content:
            delta["content"] = content
        return {
            "id": spec.public_id,
            "object": "chat.completion.chunk",
            "created": spec.created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    return {
        "id": spec.public_id,
        "object": "text_completion",
        "created": spec.created,
        "model": model,
        "choices": [{"index": 0, "text": content or "", "finish_reason": finish_reason, "index": 0}],
    }


async def stream_response_body(request: Any, backend: InferenceService, job: RequestJob):
    """Yield OpenAI SSE chunks plus comment heartbeats during long quiet prefills."""
    spec = job.spec
    info = backend.ready_info
    model = info.model if info is not None else spec.model
    try:
        yield "data: " + json.dumps(_chunk_payload(spec, model, role=(spec.kind == "chat")), ensure_ascii=False) + "\n\n"
        next_heartbeat = asyncio.get_running_loop().time() + SSE_HEARTBEAT_SECONDS
        while True:
            if not job.chunks.empty():
                piece = job.chunks.get_nowait()
                if piece:
                    next_heartbeat = asyncio.get_running_loop().time() + SSE_HEARTBEAT_SECONDS
                    yield "data: " + json.dumps(_chunk_payload(spec, model, content=piece), ensure_ascii=False) + "\n\n"
                continue
            if job.result.done():
                outcome = job.result.result()
                if outcome.error is not None:
                    error = {"error": {"message": outcome.error.message, "type": outcome.error.error_type, "code": outcome.error.code}}
                    yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"
                else:
                    yield "data: " + json.dumps(_chunk_payload(spec, model, finish_reason=outcome.finish_reason), ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
                return
            if await request.is_disconnected():
                await backend.cancel(job, "client_disconnected")
                return
            loop = asyncio.get_running_loop()
            remaining = job.deadline - loop.time()
            if remaining <= 0:
                await backend.cancel(job, "request_deadline_exceeded")
                error = {"error": {"message": "Request exceeded the 1800-second deadline.", "type": "timeout_error", "code": "request_timeout"}}
                yield "data: " + json.dumps(error) + "\n\n"
                yield "data: [DONE]\n\n"
                return
            until_heartbeat = next_heartbeat - loop.time()
            if until_heartbeat <= 0:
                yield ": keep-alive\n\n"
                next_heartbeat = loop.time() + SSE_HEARTBEAT_SECONDS
                continue
            job.wake.clear()
            if not job.chunks.empty() or job.result.done():
                continue
            try:
                timeout = min(0.25, remaining, until_heartbeat)
                await asyncio.wait_for(job.wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        await backend.cancel(job, "client_disconnected")
        raise
    finally:
        if not job.result.done():
            await backend.cancel(job, "client_disconnected")


def create_app(service: InferenceService | None = None) -> Any:
    try:
        from contextlib import asynccontextmanager
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ModuleNotFoundError as exc:
        raise RuntimeError("FastAPI is required to create the AC922 HTTP app") from exc

    # FastAPI resolves postponed endpoint annotations from this module's
    # globals, not the local scope in which its Request class was imported.
    globals()["Request"] = Request

    inference = service or InferenceService()

    @asynccontextmanager
    async def lifespan(app: Any):
        await inference.start()
        yield
        await inference.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.inference = inference

    @app.exception_handler(ApiFault)
    async def api_fault_handler(request: Request, exc: ApiFault):
        del request
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"message": exc.message, "type": exc.error_type, "code": exc.code}},
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "worker_alive": inference.is_alive}

    @app.get("/ready")
    async def ready():
        info = inference.ready_info
        if info is None:
            return JSONResponse(status_code=503, content={"status": "unready", "reason": inference.unready_reason})
        return {"status": "ready", **info.__dict__}

    @app.get("/v1/models")
    async def models():
        info = inference.ready_info
        if info is None:
            raise ApiFault(503, "worker_unready", "Inference worker is not ready.", "server_error")
        return {"object": "list", "data": [{"id": info.model, "object": "model", "owned_by": "local"}]}

    async def handle_request(request: Request, kind: str):
        info = inference.ready_info
        if info is None:
            raise ApiFault(503, "worker_unready", "Inference worker is not ready.", "server_error")
        payload = await _read_json_limited(request)
        spec = parse_request(payload, kind, info.model, info.max_context)
        job = await inference.submit(spec)
        if spec.stream:
            return StreamingResponse(
                stream_response_body(request, inference, job),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        outcome = await inference.wait_result(job, request)
        if spec.kind == "chat":
            return {
                "id": spec.public_id,
                "object": "chat.completion",
                "created": spec.created,
                "model": info.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": outcome.text}, "finish_reason": outcome.finish_reason}],
                "usage": _usage(outcome),
            }
        return {
            "id": spec.public_id,
            "object": "text_completion",
            "created": spec.created,
            "model": info.model,
            "choices": [{"index": 0, "text": outcome.text, "finish_reason": outcome.finish_reason}],
            "usage": _usage(outcome),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle_request(request, "chat")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle_request(request, "completion")

    return app


try:
    app = create_app()
except RuntimeError as exc:
    if "FastAPI is required" not in str(exc):
        raise
    app = None


if __name__ == "__main__":
    if app is None:
        raise SystemExit("FastAPI is required; install/use the AC922 service environment")
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("AC922_API_PORT", "8001")),
        access_log=False,
    )
