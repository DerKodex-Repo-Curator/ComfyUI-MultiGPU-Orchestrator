import asyncio
import copy
import json
import sys
import threading
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator import (  # noqa: E402
    MultiGpuOrchestrator, WorkerState, create_direct_prompt_routing_middleware, synthesize_jobs_payload,
)


class MemoryQueue:
    """The parent queue contract used by native ComfyUI history handlers."""

    def __init__(self):
        self.mutex = threading.RLock()
        self.history = {}

    def get_history(self, prompt_id=None, max_items=None, offset=-1):
        with self.mutex:
            if prompt_id is not None:
                return copy.deepcopy({prompt_id: self.history[prompt_id]}) if prompt_id in self.history else {}
            items = list(self.history.items())
            if offset < 0:
                offset = max(len(items) - max_items, 0) if max_items is not None else 0
            return copy.deepcopy(dict(items[offset:] if max_items is None else items[offset:offset + max_items]))

    def wipe_history(self):
        with self.mutex:
            self.history.clear()

    def delete_history_item(self, prompt_id):
        with self.mutex:
            self.history.pop(prompt_id, None)


def result(prompt_id, status="success"):
    return {
        "prompt": [0, prompt_id, {"1": {"class_type": "SaveImage"}}, {"create_time": 123}, ["1"]],
        "outputs": {"1": {"images": [{"filename": "image.png", "subfolder": "MaoMao", "type": "output"}]}},
        "status": {"status_str": status, "completed": status == "success", "messages": [["execution_" + status, {"prompt_id": prompt_id}]]},
        "meta": {"1": {"node_id": "1"}},
    }


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_jobs_import_completed_results_and_retain_them_after_worker_stop(self):
        orchestrator = self.make_orchestrator()
        worker = WorkerState(0, 9000, "http://worker", status="healthy")
        orchestrator.workers = [worker]
        orchestrator._worker_history = AsyncMock(return_value={"done": result("done")})
        orchestrator._fetch_worker_json_with_query = AsyncMock(return_value={"jobs": []})
        queue = orchestrator.prompt_server.prompt_queue
        app = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])

        async def native_jobs(request):
            return web.json_response(synthesize_jobs_payload([], [queue.get_history()], limit=None))

        async def native_detail(request):
            jobs = synthesize_jobs_payload([], [queue.get_history(request.match_info["job_id"])])["jobs"]
            return web.json_response(jobs[0] if jobs else {"error": "missing"}, status=200 if jobs else 404)

        app.router.add_get("/api/jobs", native_jobs)
        app.router.add_get("/api/jobs/{job_id}", native_detail)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/api/jobs")
            jobs = (await response.json())["jobs"]
            self.assertEqual([job["id"] for job in jobs], ["done"])
            self.assertEqual(jobs[0]["status"], "completed")
            self.assertEqual(jobs[0]["outputs_count"], 1)
            self.assertEqual(queue.history["done"], result("done"))
            worker.status = "stopped"
            response = await client.get("/api/jobs/done")
            self.assertEqual(await response.json(), jobs[0])
            response = await client.get("/api/jobs")
            self.assertEqual((await response.json())["jobs"], jobs)

    def make_orchestrator(self):
        server = SimpleNamespace(prompt_queue=MemoryQueue(), send=AsyncMock())
        orchestrator = MultiGpuOrchestrator(server)
        orchestrator._started = True
        orchestrator._start_primary_asset_seed = Mock()
        return orchestrator

    def add_parent_routes(self, app, orchestrator):
        queue = orchestrator.prompt_server.prompt_queue

        async def native_get(request):
            query = request.rel_url.query
            return web.json_response(queue.get_history(
                prompt_id=request.match_info.get("prompt_id"),
                max_items=int(query["max_items"]) if "max_items" in query else None,
                offset=int(query.get("offset", -1)),
            ))

        async def native_control(request):
            payload = await request.json()
            if payload.get("clear"):
                queue.wipe_history()
            for pid in payload.get("delete", []):
                queue.delete_history_item(pid)
            return web.Response(status=200)

        for prefix in ("", "/api"):
            app.router.add_get(prefix + "/history", native_get)
            app.router.add_get(prefix + "/history/{prompt_id}", native_get)
            app.router.add_post(prefix + "/history", native_control)
        app.router.add_get("/mgpu/history", orchestrator.proxy_history)
        app.router.add_post("/mgpu/history", orchestrator.proxy_history_control)

    async def test_api_prompt_without_client_imports_history_and_survives_worker_stop(self):
        async with AsyncExitStack() as stack:
            orchestrator = self.make_orchestrator()
            orchestrator._session = await stack.enter_async_context(aiohttp.ClientSession())
            worker_history = {}
            submitted = []
            completed = result("worker-prompt")
            imported = asyncio.Event()
            orchestrator._start_primary_asset_seed = Mock(side_effect=imported.set)
            worker_app = web.Application()

            async def prompt(request):
                submitted.append(await request.json())
                return web.json_response({"prompt_id": "worker-prompt", "number": 0, "node_errors": {}})

            async def queue(request):
                return web.json_response({"queue_running": [], "queue_pending": [] if worker_history else [completed["prompt"]]})

            async def history(request):
                pid = request.match_info.get("prompt_id")
                return web.json_response(worker_history if pid is None else {pid: worker_history[pid]} if pid in worker_history else {})

            worker_app.router.add_post("/prompt", prompt)
            worker_app.router.add_get("/queue", queue)
            worker_app.router.add_get("/history", history)
            worker_app.router.add_get("/history/{prompt_id}", history)
            worker_server = await stack.enter_async_context(TestServer(worker_app))
            worker = WorkerState(0, worker_server.port, str(worker_server.make_url("/")).rstrip("/"), status="healthy")
            orchestrator.workers = [worker]
            parent = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])
            native_prompt = AsyncMock(return_value=web.json_response({"prompt_id": "native"}))
            parent.router.add_post("/prompt", native_prompt)
            parent.router.add_post("/api/prompt", native_prompt)
            self.add_parent_routes(parent, orchestrator)
            client = await stack.enter_async_context(TestClient(TestServer(parent)))
            stack.push_async_callback(orchestrator.close)
            payload = {"prompt": {"1": {"class_type": "SaveImage"}}, "extra_data": {"extra_pnginfo": {"workflow": {"id": "test"}}}}
            response = await client.post("/prompt", json=payload)
            self.assertEqual(await response.json(), {"prompt_id": "worker-prompt", "number": 0, "node_errors": {}})
            self.assertEqual(submitted, [payload])
            native_prompt.assert_not_awaited()
            worker_history["worker-prompt"] = completed
            # No browser bridge and no history GET: background import must work.
            await asyncio.wait_for(imported.wait(), timeout=3)
            self.assertEqual(orchestrator.prompt_server.prompt_queue.history["worker-prompt"], completed)
            await orchestrator.stop_worker(0)
            orchestrator.prompt_server.prompt_queue.history["local"] = result("local")
            for prefix in ("", "/api"):
                response = await client.get(prefix + "/history/worker-prompt")
                self.assertEqual(await response.json(), {"worker-prompt": completed})
                response = await client.get(prefix + "/history?max_items=1&offset=0")
                self.assertEqual(await response.json(), {"worker-prompt": completed})
                response = await client.get(prefix + "/history/missing")
                self.assertEqual(await response.json(), {})
            response = await client.get("/mgpu/history")
            self.assertEqual(set(await response.json()), {"local", "worker-prompt"})
            response = await client.post("/api/history", json={"delete": ["worker-prompt"]})
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.read(), b"")
            response = await client.get("/history")
            self.assertEqual(set(await response.json()), {"local"})
            response = await client.post("/mgpu/history", json={"clear": True})
            self.assertEqual(response.status, 200)
            self.assertEqual(orchestrator.prompt_server.prompt_queue.history, {})

    async def test_native_history_merges_workers_once_and_controls_all_histories(self):
        async with AsyncExitStack() as stack:
            orchestrator = self.make_orchestrator()
            worker0 = WorkerState(0, 9000, "http://worker0", status="healthy")
            worker1 = WorkerState(1, 9001, "http://worker1", status="healthy")
            orchestrator.workers = [worker0, worker1]
            histories = {0: {"a": result("a")}, 1: {"b": result("b", "error")}}
            orchestrator._worker_history = AsyncMock(side_effect=lambda worker, pid=None: histories[worker.gpu_index])
            # Simulate a worker that ignores deletion or is unavailable afterwards.
            orchestrator._fanout_post = AsyncMock(return_value=[])
            parent = web.Application(middlewares=[create_direct_prompt_routing_middleware(orchestrator)])
            self.add_parent_routes(parent, orchestrator)
            client = await stack.enter_async_context(TestClient(TestServer(parent)))
            orchestrator.prompt_server.prompt_queue.history["local"] = result("local")
            response = await client.get("/api/history?max_items=1&offset=1")
            self.assertEqual(await response.json(), {"a": result("a")})
            response = await client.post("/history", json={"delete": ["a"]})
            self.assertEqual(response.status, 200)
            orchestrator._fanout_post.assert_awaited_with([worker0, worker1], "/history", {"delete": ["a"]})
            response = await client.get("/api/history")
            self.assertEqual(set(await response.json()), {"local", "b"})
            response = await client.post("/history", json={"clear": True})
            self.assertEqual(response.status, 200)
            response = await client.get("/history")
            self.assertEqual(await response.json(), {})
            histories[1]["new"] = result("new")
            response = await client.get("/history")
            self.assertEqual(await response.json(), {"new": result("new")})

    async def test_terminal_websocket_event_publishes_history_before_notification(self):
        orchestrator = self.make_orchestrator()
        worker = WorkerState(0, 9000, "http://worker", status="healthy")
        orchestrator.workers = [worker]
        orchestrator.prompt_to_worker["done"] = worker
        orchestrator._worker_history = AsyncMock(return_value={"done": result("done")})

        async def send(message_type, data, client_id):
            self.assertEqual(orchestrator.prompt_server.prompt_queue.get_history("done"), {"done": result("done")})

        orchestrator.prompt_server.send = AsyncMock(side_effect=send)
        await orchestrator._forward_ws_text(worker, "client", json.dumps({"type": "executing", "data": {"prompt_id": "done", "node": None}}))
        orchestrator.prompt_server.send.assert_awaited_once_with("executing", {"prompt_id": "done", "node": None}, "client")

    async def test_pending_history_poll_retires_cancelled_jobs_and_close_cancels_poll(self):
        orchestrator = self.make_orchestrator()
        worker = WorkerState(0, 9000, "http://worker", status="healthy")
        orchestrator.workers = [worker]
        orchestrator._fetch_worker_json = AsyncMock(return_value={"queue_running": [], "queue_pending": []})
        orchestrator._worker_history = AsyncMock(return_value={})
        orchestrator._pending_history["cancelled"] = worker
        await asyncio.wait_for(orchestrator._sync_pending_history(), timeout=1)
        self.assertEqual(orchestrator._pending_history, {})
        blocked = asyncio.Event()

        async def fetch(*args):
            blocked.set()
            await asyncio.Event().wait()

        orchestrator._fetch_worker_json = fetch
        orchestrator._remember_prompt_worker(worker, b'{"prompt_id":"running"}')
        await blocked.wait()
        task = orchestrator._history_sync_task
        await asyncio.wait_for(orchestrator.close(), timeout=1)
        self.assertTrue(task.done())
        self.assertIsNone(orchestrator._history_sync_task)

    async def test_worker_history_timeout_preserves_parent_history_and_worker_health(self):
        orchestrator = self.make_orchestrator()
        worker = WorkerState(0, 9000, "http://worker", status="healthy")
        orchestrator.workers = [worker]
        orchestrator.prompt_server.prompt_queue.history["local"] = result("local")
        orchestrator._worker_history = AsyncMock(side_effect=asyncio.TimeoutError)
        await orchestrator.sync_history()
        self.assertEqual(worker.status, "healthy")
        self.assertEqual(orchestrator.prompt_server.prompt_queue.get_history(), {"local": result("local")})
        orchestrator._fanout_post = AsyncMock(side_effect=asyncio.TimeoutError)
        request = SimpleNamespace(json=AsyncMock(return_value={"clear": True}))
        response = await orchestrator.proxy_history_control(request)
        self.assertEqual(response.status, 200)
        self.assertEqual(orchestrator.prompt_server.prompt_queue.history, {})

    async def test_history_retention_does_not_reimport_evicted_worker_entries(self):
        orchestrator = self.make_orchestrator()
        history = {pid: result(pid) for pid in ("a", "b", "c")}
        with patch.dict(sys.modules, {"execution": SimpleNamespace(MAXIMUM_HISTORY_SIZE=2)}):
            orchestrator._merge_parent_history(history)
            self.assertEqual(list(orchestrator.prompt_server.prompt_queue.history), ["b", "c"])
            orchestrator._merge_parent_history(history)
            self.assertEqual(list(orchestrator.prompt_server.prompt_queue.history), ["b", "c"])
