"""Private acceptance wrapper; the model stays loaded across HTTP and in-band tests.

The mode-0600 Unix socket is local to the owner and is sealed after acceptance.
The promoted service's next start uses plain api.py, without this test wrapper.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import time
import uuid

import api
import uvicorn


class AcceptanceWorker(api.SubprocessWorker):
    def __init__(self):
        super().__init__()
        self.direct_epoch: str | None = None
        self.direct_events: asyncio.Queue[dict] = asyncio.Queue(maxsize=65536)
        self.mode_lock = asyncio.Lock()

    async def start(self, on_event, on_exit):
        async def event(event):
            if self.direct_epoch is not None and event.get("type") != "ready":
                await self.direct_events.put(event)
            else:
                await on_event(event)

        async def exited(code, reason):
            if self.direct_epoch is not None:
                await self.direct_events.put({"type": "acceptance_worker_exit", "code": code, "reason": reason})
            await on_exit(code, reason)

        await super().start(event, exited)


class AcceptanceService(api.InferenceService):
    async def submit(self, spec):
        async with self.backend.mode_lock:
            if self.backend.direct_epoch is not None:
                raise api.ApiFault(503, "acceptance_in_progress", "Private in-band acceptance is running.", "server_error")
            return await super().submit(spec)


async def serve():
    backend = AcceptanceWorker()
    service = AcceptanceService(backend)
    app = api.create_app(service)
    run_dir = Path(os.environ["AC922_RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # Linux sockaddr_un.sun_path is bounded independently of filesystem paths.
    socket_path = Path(os.environ.get("AC922_ACCEPTANCE_SOCKET",
                                     "/home/debian/ac922-monster/production/pool-boundary-control.sock"))
    if socket_path.exists():
        socket_path.unlink()
    admin = None

    async def handle(reader, writer):
        nonlocal admin
        claimed = False
        try:
            command = json.loads(await asyncio.wait_for(reader.readline(), 10))
            if command.get("type") == "seal":
                if admin is not None:
                    admin.close()
                    await admin.wait_closed()
                    socket_path.unlink(missing_ok=True)
                writer.write(b'{"status":"SEALED"}\n')
                await writer.drain()
                return
            requests = command["requests"]
            if not isinstance(requests, list) or not 1 <= len(requests) <= 4:
                raise ValueError("in-band wave requires one to four requests")
            async with backend.mode_lock:
                async with service._condition:
                    if service.ready_info is None or service._active or service._queued or service._wave_runtime is not None:
                        raise RuntimeError("HTTP scheduler must be ready and fully drained")
                    if backend.direct_epoch is not None:
                        raise RuntimeError("another in-band wave is active")
                    epoch = command.get("id", "lastwave-" + str(uuid.uuid4()))
                    if not isinstance(epoch, str) or len(epoch) > 120:
                        raise ValueError("invalid in-band wave id")
                    backend.direct_epoch = epoch
                    claimed = True
            while not backend.direct_events.empty():
                backend.direct_events.get_nowait()
            began = time.monotonic()
            await backend.send({"type": "wave", "id": epoch, "refill": False, "requests": requests})
            output = {item["id"]: bytearray() for item in requests}
            result = {"id": epoch, "accepted": {}, "cache_status": {}, "done": {}, "errors": [],
                      "first_token_s": {}, "last_token_s": {}}
            async with asyncio.timeout(1500):
                while True:
                    event = await backend.direct_events.get()
                    kind, request_id = event["type"], event.get("id")
                    if kind in {"accepted", "done", "cache_status"}:
                        result[kind][request_id] = event
                    elif kind == "token":
                        elapsed = time.monotonic()-began
                        result["first_token_s"].setdefault(request_id, elapsed)
                        result["last_token_s"][request_id] = elapsed
                        output[request_id].extend(event["piece_bytes"])
                    elif kind == "error":
                        result["errors"].append(event)
                    elif kind == "wave_done":
                        if event["id"] != epoch:
                            raise RuntimeError("in-band epoch mismatch")
                        result["timings"] = event.get("timings", {})
                        break
                    elif kind == "acceptance_worker_exit":
                        raise RuntimeError("worker exited during acceptance")
                    else:
                        raise RuntimeError("unexpected in-band event " + kind)
            result["wall_s"] = time.monotonic()-began
            result["texts"] = {key: value.decode("utf-8", errors="replace") for key,value in output.items()}
            writer.write(json.dumps(result, ensure_ascii=False).encode()+b"\n")
            await writer.drain()
        except Exception as error:
            writer.write(json.dumps({"status": "ERROR", "error": str(error)}).encode()+b"\n")
            await writer.drain()
            if claimed:
                await backend.stop()
        finally:
            if claimed:
                async with backend.mode_lock:
                    backend.direct_epoch = None
            writer.close()
            await writer.wait_closed()

    admin = await asyncio.start_unix_server(handle, path=str(socket_path), limit=8 << 20)
    socket_path.chmod(0o600)
    try:
        config = uvicorn.Config(app, host="127.0.0.1", port=int(os.environ.get("AC922_API_PORT", "8002")), access_log=False)
        await uvicorn.Server(config).serve()
    finally:
        admin.close()
        await admin.wait_closed()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(serve())
