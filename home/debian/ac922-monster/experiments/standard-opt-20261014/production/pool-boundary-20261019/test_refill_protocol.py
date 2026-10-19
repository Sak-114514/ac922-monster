"""CPU-only async regression tests for the optional bank-refill protocol."""

from __future__ import annotations

import asyncio
import unittest

import api


MODEL = "flash-next-iq3-xxs"
READY = {
    "type": "ready",
    "max_context": 262140,
    "pool_tokens": 524288,
    "max_parallel": 4,
    "model": MODEL,
    "refill_supported": True,
}


def chat_spec(text: str) -> api.RequestSpec:
    return api.parse_request(
        {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 8},
        "chat", MODEL, 262140,
    )


class RefillWorker:
    def __init__(self, reject_refill: set[str] | None = None):
        self.alive = False
        self.on_event = None
        self.on_exit = None
        self.commands: list[dict] = []
        self.active_ids: set[str] = set()
        self.active_wave: str | None = None
        self.wave_count = 0
        self.reject_refill = reject_refill or set()
        self.initial_started = asyncio.Event()
        self.refill_seen = asyncio.Event()
        self.refill_accepted = asyncio.Event()
        self.close_waiting = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.allow_close.set()
        self.hold_refill = False
        self.allow_refill = asyncio.Event()
        self.allow_refill.set()
        self.defer_refill = False
        self.deferred_refill: dict | None = None

    async def start(self, on_event, on_exit):
        self.on_event, self.on_exit = on_event, on_exit
        self.alive = True
        await on_event(dict(READY))

    async def stop(self):
        self.alive = False

    async def send(self, command):
        self.commands.append(dict(command))
        kind = command["type"]
        if kind == "wave":
            self.wave_count += 1
            self.active_wave = command["id"]
            self.active_ids = {item["id"] for item in command["requests"]}
            self.initial_started.set()
            for request in command["requests"]:
                await self.on_event({
                    "type": "accepted", "id": request["id"],
                    "input_tokens": 3, "reserved_tokens": request["max_tokens"],
                })
            if self.wave_count == 1:
                await self.on_event({
                    "type": "bank_idle", "wave_id": self.active_wave, "bank": 1,
                    "capacity": 262140, "ubatch": 256,
                })
            else:
                for request in command["requests"]:
                    await self._finish(request["id"])
                await self.request_close()
        elif kind == "refill":
            self.refill_seen.set()
            if self.defer_refill:
                self.deferred_refill = dict(command)
                return
            if self.hold_refill:
                await self.allow_refill.wait()
            request = command["requests"][0]
            request_id = request["id"]
            if request_id in self.reject_refill:
                await self.on_event({
                    "type": "refill_rejected", "wave_id": command["id"],
                    "bank": command["bank"], "id": request_id,
                    "reason": "capacity_or_geometry",
                })
            else:
                self.active_ids.add(request_id)
                await self.on_event({
                    "type": "accepted", "id": request_id,
                    "input_tokens": 3, "reserved_tokens": request["max_tokens"],
                })
                self.refill_accepted.set()
        elif kind == "cancel":
            await self._finish(command["id"], "cancelled")
        elif kind == "close_wave":
            self.close_waiting.set()
            if not self.allow_close.is_set():
                await self.allow_close.wait()
            self.active_wave = None
            await self.on_event({"type": "wave_done", "id": command["id"]})
        else:
            raise AssertionError(f"unexpected command: {kind}")

    async def _finish(self, request_id: str, reason: str = "length"):
        if request_id not in self.active_ids:
            return
        await self.on_event({
            "type": "done", "id": request_id, "input_tokens": 3,
            "output_tokens": 0, "finish_reason": reason,
        })
        self.active_ids.remove(request_id)

    async def finish(self, request_id: str):
        await self._finish(request_id)

    async def reject_deferred_refill(self):
        assert self.deferred_refill is not None
        command = self.deferred_refill
        request_id = command["requests"][0]["id"]
        await self.on_event({
            "type": "refill_rejected", "wave_id": command["id"],
            "bank": command["bank"], "id": request_id,
            "reason": "capacity_or_geometry",
        })

    async def request_close(self):
        assert self.active_wave is not None
        await self.on_event({"type": "wave_closing", "id": self.active_wave})


class RefillProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = RefillWorker()
        self.service = api.InferenceService(
            self.worker, coalesce_seconds=0, request_deadline_seconds=2,
        )
        await self.service.start()

    async def asyncTearDown(self):
        await self.service.stop()

    async def _await(self, event: asyncio.Event):
        await asyncio.wait_for(event.wait(), timeout=1)

    async def test_idle_bank_refills_and_wave_drains_after_close_ack(self):
        first = await self.service.submit(chat_spec("first"))
        await self._await(self.worker.initial_started)
        second = await self.service.submit(chat_spec("second"))
        await self._await(self.worker.refill_seen)

        refill = next(command for command in self.worker.commands if command["type"] == "refill")
        self.assertEqual(refill["bank"], 1)
        self.assertEqual(refill["id"], first.wave_id)
        self.assertEqual([item["id"] for item in refill["requests"]], [second.spec.request_id])
        self.assertEqual(self.service.active_count, 2)

        await self.worker.finish(first.spec.request_id)
        await self.worker.finish(second.spec.request_id)
        await self.worker.request_close()
        self.assertEqual((await first.result).finish_reason, "length")
        self.assertEqual((await second.result).finish_reason, "length")
        self.assertEqual([command["type"] for command in self.worker.commands][-1], "close_wave")

    async def test_terminal_epoch_idle_bank_waits_for_next_wave(self):
        first = await self.service.submit(chat_spec("first cohort"))
        await self._await(self.worker.initial_started)
        await self.worker.finish(first.spec.request_id)

        second = await self.service.submit(chat_spec("next cohort"))
        await asyncio.sleep(0)
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 0)
        self.assertEqual(self.service.queue_depth, 1)

        await self.worker.request_close()
        await asyncio.wait_for(first.result, timeout=1)
        await asyncio.wait_for(second.result, timeout=1)
        self.assertEqual(sum(command["type"] == "wave" for command in self.worker.commands), 2)
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 0)

    async def test_rejected_request_is_not_retried_in_epoch_and_runs_next_wave(self):
        first = await self.service.submit(chat_spec("first"))
        await self._await(self.worker.initial_started)
        second = await self.service.submit(chat_spec("second"))
        self.worker.reject_refill.add(second.spec.request_id)
        await self._await(self.worker.refill_seen)
        await asyncio.sleep(0)
        refill_count = sum(command["type"] == "refill" for command in self.worker.commands)
        self.assertEqual(refill_count, 1)
        self.assertEqual(second.state, "queued")
        self.assertEqual(self.service.queue_depth, 1)

        await self.worker.finish(first.spec.request_id)
        await self.worker.request_close()
        await asyncio.wait_for(second.result, timeout=1)
        self.assertEqual(sum(command["type"] == "wave" for command in self.worker.commands), 2)
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 1)
        self.assertTrue(first.result.done())

    async def test_refill_rejection_after_wave_closing_is_drained_and_requeued(self):
        self.worker.defer_refill = True
        first = await self.service.submit(chat_spec("first"))
        await self._await(self.worker.initial_started)
        second = await self.service.submit(chat_spec("second"))
        await self._await(self.worker.refill_seen)
        await self.worker.finish(first.spec.request_id)

        self.worker.allow_close.clear()
        close_task = asyncio.create_task(self.worker.request_close())
        await self._await(self.worker.close_waiting)
        await self.worker.reject_deferred_refill()
        self.assertEqual(second.state, "queued")

        self.worker.allow_close.set()
        await close_task
        await asyncio.wait_for(first.result, timeout=1)
        await asyncio.wait_for(second.result, timeout=1)
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 1)
        self.assertEqual(sum(command["type"] == "wave" for command in self.worker.commands), 2)

    async def test_closing_epoch_never_dispatches_late_refill(self):
        first = await self.service.submit(chat_spec("first"))
        await self._await(self.worker.initial_started)
        await self.worker.finish(first.spec.request_id)
        self.worker.allow_close.clear()
        close_task = asyncio.create_task(self.worker.request_close())
        await self._await(self.worker.close_waiting)
        queued = await self.service.submit(chat_spec("after close started"))
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 0)

        self.worker.allow_close.set()
        await close_task
        await asyncio.wait_for(first.result, timeout=1)
        await asyncio.wait_for(queued.result, timeout=1)
        self.assertEqual(sum(command["type"] == "refill" for command in self.worker.commands), 0)
        self.assertEqual(sum(command["type"] == "wave" for command in self.worker.commands), 2)

    async def test_inflight_refill_reserves_bounded_queue_capacity(self):
        await self.service.stop()
        self.worker = RefillWorker()
        self.worker.hold_refill = True
        self.worker.allow_refill.clear()
        self.service = api.InferenceService(
            self.worker, max_queue=1, coalesce_seconds=0, request_deadline_seconds=2,
        )
        await self.service.start()
        first = await self.service.submit(chat_spec("first"))
        await self._await(self.worker.initial_started)
        second = await self.service.submit(chat_spec("second"))
        await self._await(self.worker.refill_seen)
        self.assertTrue(second.refill_slot_reserved)
        with self.assertRaises(api.ApiFault) as error:
            await self.service.submit(chat_spec("third"))
        self.assertEqual(error.exception.status, 429)

        self.worker.allow_refill.set()
        await self._await(self.worker.refill_accepted)
        await self.worker.finish(first.spec.request_id)
        await self.worker.finish(second.spec.request_id)
        await self.worker.request_close()
        await asyncio.wait_for(first.result, timeout=1)
        await asyncio.wait_for(second.result, timeout=1)

    def test_legacy_ready_event_disables_refill(self):
        legacy = dict(READY)
        legacy.pop("refill_supported")
        info = api.InferenceService._parse_ready(legacy)
        self.assertIsNotNone(info)
        self.assertFalse(info.refill_supported)


class CacheStatusWorker:
    def __init__(self, prefix_cache_enabled: bool | None, status_before_accept: bool = False):
        self.alive = False
        self.on_event = None
        self.on_exit = None
        self.prefix_cache_enabled = prefix_cache_enabled
        self.status_before_accept = status_before_accept
        self.active_wave: str | None = None
        self.request_id: str | None = None
        self.started = asyncio.Event()
        self.commands: list[dict] = []

    async def start(self, on_event, on_exit):
        self.on_event, self.on_exit = on_event, on_exit
        self.alive = True
        ready = dict(READY)
        ready["refill_supported"] = False
        ready.pop("prefix_cache_enabled", None)
        if self.prefix_cache_enabled is not None:
            ready["prefix_cache_enabled"] = self.prefix_cache_enabled
        await on_event(ready)

    async def stop(self):
        self.alive = False

    async def send(self, command):
        self.commands.append(dict(command))
        if command["type"] != "wave":
            raise AssertionError(f"unexpected cache test command: {command['type']}")
        self.active_wave = command["id"]
        request = command["requests"][0]
        self.request_id = request["id"]
        if self.status_before_accept:
            await self.on_event({
                "type": "cache_status", "id": self.request_id, "hit": False,
                "cached_tokens": 0, "checkpoint_tokens": 0,
            })
        await self.on_event({
            "type": "accepted", "id": self.request_id,
            "input_tokens": 5, "reserved_tokens": request["max_tokens"],
        })
        self.started.set()
        if self.status_before_accept:
            await self.on_event({"type": "wave_done", "id": self.active_wave})

    async def finish(self):
        assert self.request_id is not None and self.active_wave is not None
        await self.on_event({
            "type": "token", "id": self.request_id, "token_id": 1,
            "piece_bytes": list(b"x"),
        })
        await self.on_event({
            "type": "done", "id": self.request_id, "input_tokens": 5,
            "output_tokens": 1, "finish_reason": "length",
        })
        await self.on_event({"type": "wave_done", "id": self.active_wave})


class PreAdmissionRejectWorker:
    def __init__(self, pool_overflow: bool):
        self.alive = False
        self.on_event = None
        self.on_exit = None
        self.pool_overflow = pool_overflow
        self.wave_count = 0

    async def start(self, on_event, on_exit):
        self.on_event, self.on_exit = on_event, on_exit
        self.alive = True
        await on_event({**READY, "prefix_cache_enabled": False})

    async def stop(self):
        self.alive = False

    async def send(self, command):
        if command["type"] != "wave":
            raise AssertionError(f"unexpected pre-admission command: {command['type']}")
        self.wave_count += 1
        message = "shared context pool exceeded" if self.pool_overflow else "invalid request shape"
        for request in command["requests"]:
            await self.on_event({
                "type": "error", "id": request["id"], "status": 400, "message": message,
            })
        await self.on_event({"type": "wave_done", "id": command["id"]})


class CacheStatusProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, worker: CacheStatusWorker):
        service = api.InferenceService(worker, coalesce_seconds=0, request_deadline_seconds=2)
        await service.start()
        job = await service.submit(chat_spec("cache status"))
        await asyncio.wait_for(worker.started.wait(), timeout=1)
        return service, job

    async def test_cache_status_after_acceptance_is_reported_in_usage(self):
        worker = CacheStatusWorker(prefix_cache_enabled=True)
        service, job = await self._run(worker)
        self.assertTrue(service.ready_info.prefix_cache_enabled)
        await service._on_event({
            "type": "cache_status", "id": job.spec.request_id, "hit": True,
            "cached_tokens": 2, "checkpoint_tokens": 2,
        })
        await worker.finish()
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertTrue(job.cache_hit)
        self.assertEqual(job.checkpoint_tokens, 2)
        self.assertEqual(api._usage(outcome), {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "total_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 2},
        })
        await service.stop()

    async def test_cache_status_before_acceptance_is_rejected(self):
        worker = CacheStatusWorker(prefix_cache_enabled=True, status_before_accept=True)
        service, job = await self._run(worker)
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.error.code, "worker_protocol_error")
        await service.stop()

    async def test_cached_tokens_cannot_exceed_accepted_prompt(self):
        worker = CacheStatusWorker(prefix_cache_enabled=True)
        service, job = await self._run(worker)
        await service._on_event({
            "type": "cache_status", "id": job.spec.request_id, "hit": True,
            "cached_tokens": 6, "checkpoint_tokens": 3,
        })
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.error.code, "worker_protocol_error")
        await service.stop()

    async def test_legacy_ready_without_prefix_field_keeps_zero_cached_usage(self):
        worker = CacheStatusWorker(prefix_cache_enabled=None)
        service, job = await self._run(worker)
        self.assertFalse(service.ready_info.prefix_cache_enabled)
        await worker.finish()
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertEqual(api._usage(outcome)["prompt_tokens_details"], {"cached_tokens": 0})
        await service.stop()

    async def test_unclosed_initial_pool_overflow_does_not_kill_worker(self):
        worker = PreAdmissionRejectWorker(pool_overflow=True)
        service = api.InferenceService(worker, coalesce_seconds=0, request_deadline_seconds=2)
        await service.start()
        job = await service.submit(chat_spec("too large for initial wave"))
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertEqual(outcome.error.status, 400)
        self.assertTrue(service.is_alive)
        self.assertEqual(worker.wave_count, 1)
        await service.stop()

    async def test_unclosed_terminal_pre_admission_error_is_allowed(self):
        worker = PreAdmissionRejectWorker(pool_overflow=False)
        service = api.InferenceService(worker, coalesce_seconds=0, request_deadline_seconds=2)
        await service.start()
        job = await service.submit(chat_spec("invalid shape"))
        outcome = await asyncio.wait_for(job.result, timeout=1)
        self.assertEqual(outcome.error.status, 400)
        self.assertTrue(service.is_alive)
        await service.stop()


if __name__ == "__main__":
    unittest.main()
