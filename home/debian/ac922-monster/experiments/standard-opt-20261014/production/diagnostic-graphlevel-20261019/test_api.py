"""Focused protocol, queue, and HTTP tests; never starts the model worker."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import api

try:
    import httpx
except ModuleNotFoundError:
    httpx = None


MODEL = "flash-next-iq3-xxs"
READY = {
    "type": "ready",
    "max_context": 262144,
    "pool_tokens": 524288,
    "max_parallel": 4,
    "model": MODEL,
}


class FakeWorker:
    """In-memory worker implementing the documented event contract."""

    def __init__(self, *, blocked: bool = False, pool_limit: int | None = None):
        self.alive = False
        self.blocked = blocked
        self.pool_limit = pool_limit
        self.on_event = None
        self.on_exit = None
        self.commands: list[dict] = []
        self.active_wave: dict | None = None
        self.active_ids: set[str] = set()
        self.started = asyncio.Event()

    async def start(self, on_event, on_exit):
        self.on_event = on_event
        self.on_exit = on_exit
        self.alive = True
        await on_event(dict(READY))

    async def stop(self):
        self.alive = False

    async def send(self, command):
        self.commands.append(command)
        if command["type"] == "shutdown":
            self.alive = False
            return
        if command["type"] == "cancel":
            request_id = command["id"]
            if request_id in self.active_ids:
                await self._finish_request(request_id, "cancelled")
            return
        if command["type"] != "wave":
            raise AssertionError(f"unexpected worker command {command['type']}")
        self.active_wave = command
        self.active_ids = {request["id"] for request in command["requests"]}
        self.started.set()
        if self.pool_limit is not None and len(command["requests"]) > self.pool_limit:
            for request in command["requests"]:
                await self.on_event({"type": "error", "id": request["id"], "status": 400, "message": "shared context pool exceeded"})
            await self.on_event({"type": "wave_done", "id": command["id"]})
            self.active_ids.clear()
            self.active_wave = None
            return
        for request in command["requests"]:
            await self.on_event({"type": "accepted", "id": request["id"], "input_tokens": 4, "reserved_tokens": 100})
        if not self.blocked:
            for request in command["requests"]:
                await self._finish_request(request["id"], "stop")

    async def _finish_request(self, request_id: str, finish_reason: str):
        if request_id not in self.active_ids:
            return
        if finish_reason == "stop":
            # Split a multi-byte UTF-8 character across token events. The API
            # must hold incomplete bytes until the next event.
            for token_id, piece in enumerate((b"hello \xe4\xb8", b"\x96\xe7\x95\x8c"), start=1):
                await self.on_event({"type": "token", "id": request_id, "token_id": token_id, "piece_bytes": list(piece)})
            output_tokens = 2
        else:
            output_tokens = 0
        await self.on_event({
            "type": "done",
            "id": request_id,
            "input_tokens": 4,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
        })
        self.active_ids.remove(request_id)
        if not self.active_ids and self.active_wave is not None:
            wave_id = self.active_wave["id"]
            await self.on_event({"type": "wave_done", "id": wave_id})
            self.active_wave = None


def chat_spec(stream: bool = False) -> api.RequestSpec:
    return api.parse_request(
        {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "stream": stream},
        "chat",
        MODEL,
        262140,
    )


class ValidationTests(unittest.TestCase):
    def test_chat_and_completion_worker_shapes(self):
        chat = api.parse_request({
            "model": MODEL,
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "max_tokens": 77,
        }, "chat", MODEL, 262140)
        self.assertFalse(chat.raw)
        self.assertEqual(chat.worker_request()["messages"][1]["content"], "u")
        completion = api.parse_request({"model": MODEL, "prompt": "raw", "max_tokens": 12}, "completion", MODEL, 262140)
        self.assertTrue(completion.raw)
        self.assertEqual(completion.worker_request()["prompt"], "raw")
        self.assertEqual(completion.worker_request()["max_tokens"], 12)

    def test_unsupported_sampling_and_extra_parameters_rejected(self):
        invalid = (
            {"tools": []},
            {"stop": ["END"]},
            {"temperature": 0.1},
            {"top_p": 0.8},
            {"n": 2},
            {"model": "other-model"},
        )
        for extra in invalid:
            payload = {"model": MODEL, "prompt": "x"}
            payload.update(extra)
            with self.subTest(extra=extra), self.assertRaises(api.ApiFault):
                api.parse_request(payload, "completion", MODEL, 262140)

    def test_messages_reject_non_string_content_and_non_string_role(self):
        for message in (
            {"role": "user", "content": [{"type": "image_url"}]},
            {"role": ["user"], "content": "x"},
            {"role": "developer", "content": "unsupported by worker"},
            {"role": "user", "content": "x", "name": "unsupported"},
        ):
            with self.subTest(message=message), self.assertRaises(api.ApiFault):
                api.parse_request({"model": MODEL, "messages": [message]}, "chat", MODEL, 262140)

    def test_prompt_and_context_limits(self):
        with self.assertRaises(api.ApiFault) as prompt_error:
            api.parse_request({"model": MODEL, "prompt": "x" * (api.MAX_PROMPT_BYTES + 1)}, "completion", MODEL, 262140)
        self.assertEqual(prompt_error.exception.status, 413)
        with self.assertRaises(api.ApiFault) as context_error:
            api.parse_request({"model": MODEL, "prompt": "x", "max_tokens": 101}, "completion", MODEL, 100)
        self.assertEqual(context_error.exception.code, "context_limit")

    def test_worker_ready_context_is_capped_by_reserved_mtp_rows(self):
        info = api.InferenceService._parse_ready({**READY, "max_context": 262144})
        self.assertEqual(info.max_context, 262140)
        self.assertIsNone(api.InferenceService._parse_ready({**READY, "max_parallel": 3}))
        self.assertIsNone(api.InferenceService._parse_ready({**READY, "model": "wrong"}))

    def test_worker_output_limit_is_front_rejected(self):
        with self.assertRaises(api.ApiFault) as error:
            api.parse_request({"model": MODEL, "prompt": "x", "max_tokens": 8193}, "completion", MODEL, 262140)
        self.assertEqual(error.exception.status, 400)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_events_decode_utf8_and_release_active_slot(self):
        worker = FakeWorker()
        service = api.InferenceService(worker, coalesce_seconds=0.001)
        await service.start()
        self.assertIsNotNone(service.ready_info)
        job = await service.submit(chat_spec())
        outcome = await asyncio.wait_for(asyncio.shield(job.result), timeout=2)
        self.assertEqual(outcome.text, "hello 世界")
        self.assertEqual(outcome.input_tokens, 4)
        self.assertEqual(outcome.output_tokens, 2)
        self.assertEqual(service.active_count, 0)
        self.assertEqual(worker.commands[0]["type"], "wave")
        self.assertEqual(len(worker.commands[0]["requests"]), 1)
        await service.stop()

    async def test_shared_pool_rejection_splits_four_requests_without_retrying_completed_requests(self):
        worker = FakeWorker(pool_limit=2)
        service = api.InferenceService(worker, coalesce_seconds=0.02)
        await service.start()
        jobs = await asyncio.gather(*(service.submit(chat_spec()) for _ in range(4)))
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(asyncio.shield(job.result) for job in jobs)),
            timeout=3,
        )
        self.assertEqual([outcome.text for outcome in outcomes], ["hello 世界"] * 4)
        waves = [command for command in worker.commands if command.get("type") == "wave"]
        self.assertEqual([len(command["requests"]) for command in waves], [4, 2, 2])
        all_ids = [request["id"] for command in waves for request in command["requests"]]
        self.assertEqual(len(all_ids), 8)
        self.assertEqual(service.active_count, 0)
        await service.stop()

    async def test_single_request_pool_rejection_returns_400(self):
        worker = FakeWorker(pool_limit=0)
        service = api.InferenceService(worker, coalesce_seconds=0.001)
        await service.start()
        job = await service.submit(chat_spec())
        outcome = await asyncio.wait_for(asyncio.shield(job.result), timeout=2)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.error.status, 400)
        self.assertEqual(service.active_count, 0)
        await service.stop()

    async def test_stream_heartbeat_is_sse_comment_and_preserves_final_done(self):
        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        worker = FakeWorker(blocked=True)
        service = api.InferenceService(worker, coalesce_seconds=0.001)
        await service.start()
        job = await service.submit(chat_spec(stream=True))
        await asyncio.wait_for(worker.started.wait(), timeout=1)

        with patch.object(api, "SSE_HEARTBEAT_SECONDS", 0.01):
            stream = api.stream_response_body(ConnectedRequest(), service, job)
            initial = await anext(stream)
            heartbeat = await asyncio.wait_for(anext(stream), timeout=0.5)
            self.assertIn('"role": "assistant"', initial)
            self.assertEqual(heartbeat, ": keep-alive\n\n")

            await service.cancel(job, "test_complete")
            finish = await anext(stream)
            done = await anext(stream)
            self.assertIn('"finish_reason": "cancelled"', finish)
            self.assertEqual(done, "data: [DONE]\n\n")
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
        await service.stop()

    async def test_queue_cancel_releases_reservation_and_active_cancel_reaches_worker(self):
        worker = FakeWorker(blocked=True)
        service = api.InferenceService(worker, max_queue=2, coalesce_seconds=0.001)
        await service.start()
        active = await service.submit(chat_spec())
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        first_queued = await service.submit(chat_spec())
        second_queued = await service.submit(chat_spec())
        with self.assertRaises(api.ApiFault) as full:
            await service.submit(chat_spec())
        self.assertEqual(full.exception.status, 429)
        self.assertEqual(service.queue_depth, 2)

        await service.cancel(first_queued, "client_disconnected")
        self.assertEqual(service.queue_depth, 1)
        replacement = await service.submit(chat_spec())
        self.assertEqual(service.queue_depth, 2)

        await service.cancel(active, "client_disconnected")
        await asyncio.wait_for(asyncio.shield(active.result), timeout=1)
        self.assertTrue(any(c.get("type") == "cancel" and c.get("id") == active.spec.request_id for c in worker.commands))
        await service.cancel(second_queued, "client_disconnected")
        await service.cancel(replacement, "client_disconnected")
        self.assertEqual(service.queue_depth, 0)
        await service.stop()

    async def test_unready_worker_rejects_submission_and_does_not_restart(self):
        worker = FakeWorker()
        service = api.InferenceService(worker)
        await service.start()
        await service._on_exit(1, "test_exit")
        self.assertIsNone(service.ready_info)
        with self.assertRaises(api.ApiFault) as error:
            await service.submit(chat_spec())
        self.assertEqual(error.exception.status, 503)
        await service.stop()


class SupervisorExitTests(unittest.IsolatedAsyncioTestCase):
    def make_subprocess_service(self) -> api.InferenceService:
        service = api.InferenceService(api.SubprocessWorker(command_json="/unused-test-command.json"))
        service._alive = True
        service._ready = api.ReadyInfo(262140, 524288, 4, MODEL)
        return service

    async def test_opt_in_real_worker_failure_exits_after_flush_delay(self):
        service = self.make_subprocess_service()
        with patch.dict(os.environ, {"AC922_EXIT_ON_WORKER_FAILURE": "1"}):
            with patch("api.os._exit") as exit_process:
                await service._on_exit(1, "mock worker exit")
                self.assertFalse(exit_process.called)
                await asyncio.sleep(1.05)
                exit_process.assert_called_once_with(70)
        await service.stop()

    async def test_default_and_intentional_shutdown_do_not_exit(self):
        service = self.make_subprocess_service()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AC922_EXIT_ON_WORKER_FAILURE", None)
            with patch("api.os._exit") as exit_process:
                await service._on_exit(1, "mock worker exit")
                await asyncio.sleep(0.02)
                await service.stop()
                await asyncio.sleep(1.02)
                exit_process.assert_not_called()


@unittest.skipIf(httpx is None or api.app is None, "FastAPI/httpx are unavailable in this local Python environment")
class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = FakeWorker()
        self.service = api.InferenceService(self.worker, coalesce_seconds=0.001)
        await self.service.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.create_app(self.service)),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.service.stop()

    async def test_health_ready_models_and_chat_completion(self):
        health = await self.client.get("/health")
        ready = await self.client.get("/ready")
        models = await self.client.get("/v1/models")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.json()["max_context"], 262140)
        self.assertEqual(models.json()["data"][0]["id"], MODEL)

        response = await self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        })
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["choices"][0]["message"]["content"], "hello 世界")
        self.assertEqual(data["usage"]["total_tokens"], 6)
        request = self.worker.commands[-1]["requests"][0]
        self.assertFalse(request["raw"])
        self.assertEqual(request["messages"][0]["content"], "hello")

    async def test_completion_and_streaming_utf8(self):
        completion = await self.client.post("/v1/completions", json={"model": MODEL, "prompt": "raw prompt"})
        self.assertEqual(completion.status_code, 200, completion.text)
        self.assertEqual(completion.json()["choices"][0]["text"], "hello 世界")
        self.assertTrue(self.worker.commands[-1]["requests"][0]["raw"])

        streamed = await self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        })
        self.assertEqual(streamed.status_code, 200)
        self.assertIn("世界", streamed.text)
        self.assertIn("data: [DONE]", streamed.text)

    async def test_unknown_knobs_and_wrong_model_return_400_without_worker_dispatch(self):
        initial_waves = sum(command.get("type") == "wave" for command in self.worker.commands)
        requests = (
            {"model": MODEL, "prompt": "x", "tools": []},
            {"model": MODEL, "prompt": "x", "stop": "END"},
            {"model": MODEL, "prompt": "x", "temperature": 0.4},
            {"model": "other", "prompt": "x"},
            {"model": MODEL, "messages": [{"role": "developer", "content": "unsupported"}]},
        )
        for payload in requests:
            response = await self.client.post("/v1/completions", json=payload)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("error", response.json())
        final_waves = sum(command.get("type") == "wave" for command in self.worker.commands)
        self.assertEqual(final_waves, initial_waves)


if __name__ == "__main__":
    unittest.main()
