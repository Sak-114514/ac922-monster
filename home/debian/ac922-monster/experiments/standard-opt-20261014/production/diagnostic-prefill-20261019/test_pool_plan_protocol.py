"""CPU-only regression for terminal retention-plan pool errors."""

from __future__ import annotations

import asyncio
import unittest

import api


MODEL = "flash-next-iq3-xxs"
READY = {
    "type": "ready",
    "max_context": 262144,
    "pool_tokens": 524288,
    "max_parallel": 4,
    "model": MODEL,
    "refill_supported": True,
}
POOL_MESSAGE = (
    "Requested context layout exceeds the available shared pool; "
    "reduce simultaneous context budgets."
)


def completion_spec(prompt: str) -> api.RequestSpec:
    return api.parse_request(
        {"model": MODEL, "prompt": prompt, "max_tokens": 8},
        "completion",
        MODEL,
        262140,
    )


class RetentionPlanWorker:
    """One initial pool rejection followed by a normal response on same worker."""

    def __init__(self) -> None:
        self.alive = False
        self.on_event = None
        self.on_exit = None
        self.commands: list[dict] = []
        self.start_calls = 0
        self.stop_calls = 0
        self.release_calls = 0
        self.close_calls = 0
        self.active_wave_id: str | None = None
        self.close_requested: asyncio.Event | None = None

    async def start(self, on_event, on_exit) -> None:
        self.start_calls += 1
        self.on_event, self.on_exit = on_event, on_exit
        self.alive = True
        await on_event(dict(READY))

    async def stop(self) -> None:
        self.stop_calls += 1
        self.alive = False

    async def send(self, command: dict) -> None:
        self.commands.append(dict(command))
        command_type = command.get("type")
        if command_type == "close_wave":
            if command.get("id") != self.active_wave_id or self.close_requested is None:
                raise AssertionError("close_wave must match the active refill-capable wave")
            self.close_calls += 1
            self.close_requested.set()
            return
        if command_type != "wave":
            raise AssertionError(f"unexpected backend command: {command_type}")

        self.active_wave_id = command["id"]
        self.close_requested = asyncio.Event()
        requests = command["requests"]
        if len([item for item in self.commands if item.get("type") == "wave"]) == 1:
            for request in requests:
                await self.on_event({
                    "type": "error",
                    "id": request["id"],
                    "status": 400,
                    "code": "retention_plan_pool_exceeded",
                    "message": POOL_MESSAGE,
                })
        else:
            for request in requests:
                await self.on_event({
                    "type": "accepted",
                    "id": request["id"],
                    "input_tokens": 6,
                    "reserved_tokens": request["max_tokens"],
                })
                await self.on_event({
                    "type": "token",
                    "id": request["id"],
                    "token_id": 1,
                    "piece_bytes": list(b" Paris."),
                })
                await self.on_event({
                    "type": "done",
                    "id": request["id"],
                    "input_tokens": 6,
                    "output_tokens": 1,
                    "finish_reason": "stop",
                })

        if command.get("refill") is True:
            # Production emits wave_closing, waits for the API's close_wave,
            # then finishes. No bank_idle was emitted, so no refill is legal.
            await self.on_event({"type": "wave_closing", "id": command["id"]})
            if self.close_requested is None:
                raise AssertionError("close acknowledgement event was not initialized")
            await asyncio.wait_for(self.close_requested.wait(), timeout=1)
        await self.on_event({"type": "wave_done", "id": command["id"]})
        self.active_wave_id = None
        self.close_requested = None


class PoolPlanProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_pool_plan_error_is_terminal_and_next_request_uses_same_worker(self):
        worker = RetentionPlanWorker()
        service = api.InferenceService(worker, coalesce_seconds=0.03, request_deadline_seconds=2)
        await service.start()
        try:
            rejected_jobs = await asyncio.gather(*(
                service.submit(completion_spec(f"oversized cohort item {index}"))
                for index in range(4)
            ))
            rejected = await asyncio.wait_for(
                asyncio.gather(*(asyncio.shield(job.result) for job in rejected_jobs)),
                timeout=2,
            )

            self.assertEqual(len(rejected), 4)
            for outcome in rejected:
                self.assertIsNotNone(outcome.error)
                self.assertEqual(outcome.error.status, 400)
                self.assertEqual(outcome.error.message, POOL_MESSAGE)
            waves = [command for command in worker.commands if command.get("type") == "wave"]
            self.assertEqual(len(waves), 1, "pool-plan error must not trigger recursive split/retry")
            self.assertEqual(len(waves[0]["requests"]), 4)
            self.assertEqual(service.active_count, 0)
            self.assertTrue(service.is_alive)
            self.assertEqual(worker.start_calls, 1)
            self.assertEqual(worker.stop_calls, 0, "terminal capacity rejection must not stop backend")
            self.assertEqual(worker.release_calls, 0, "API must not issue backend context-release calls")
            self.assertEqual(worker.close_calls, 1, "refill-capable waves must complete the close handshake")
            self.assertEqual([command["type"] for command in worker.commands], ["wave", "close_wave"])

            followup = await service.submit(completion_spec("The capital of France is"))
            completed = await asyncio.wait_for(asyncio.shield(followup.result), timeout=2)
            self.assertIsNone(completed.error)
            self.assertEqual(completed.text, " Paris.")
            self.assertEqual(worker.start_calls, 1, "follow-up must use the already-running worker")
            self.assertEqual(worker.stop_calls, 0)
            self.assertEqual(worker.release_calls, 0)
            self.assertEqual(worker.close_calls, 2)
            waves = [command for command in worker.commands if command.get("type") == "wave"]
            self.assertEqual([len(command["requests"]) for command in waves], [4, 1])
            self.assertEqual([command["type"] for command in worker.commands], [
                "wave", "close_wave", "wave", "close_wave",
            ])
            self.assertTrue(service.is_alive)
        finally:
            await service.stop()



if __name__ == "__main__":
    unittest.main()
