import asyncio
from typing import Dict, Any
from uuid import uuid4

from huggingface_hub import snapshot_download


class ModelLoader:
    """Manage background model downloads."""

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._status: Dict[str, Dict[str, Any]] = {}

    async def _download(self, model_id: str, task_id: str) -> None:
        try:
            await asyncio.to_thread(snapshot_download, repo_id=model_id)
            self._status[task_id]["status"] = "completed"
        except Exception as e:  # pragma: no cover - network operations
            self._status[task_id]["status"] = "failed"
            self._status[task_id]["error"] = str(e)

    def start(self, model_id: str) -> str:
        """Start downloading a model in the background."""
        task_id = uuid4().hex
        self._status[task_id] = {"status": "in_progress", "model": model_id}
        self._tasks[task_id] = asyncio.create_task(self._download(model_id, task_id))
        return task_id

    def get_status(self, task_id: str) -> Dict[str, Any]:
        return self._status.get(task_id, {"status": "not_found"})
