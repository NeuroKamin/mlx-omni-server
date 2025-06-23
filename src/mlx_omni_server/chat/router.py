import asyncio
import contextvars
import gc
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Generator, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel

router = APIRouter(tags=["chat—completions"])


class ModelManager:
    """Thread-safe model manager with idle model cleanup"""

    def __init__(self, ttl_seconds: int = 30 * 60, cleanup_interval: int = 60):
        self._models: Dict[str, BaseTextModel] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, float] = {}
        self._main_lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup_task(self) -> None:
        """Start background cleanup task if not running"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval)
            await self._remove_unused_models()

    async def _remove_unused_models(self) -> None:
        """Remove models that haven't been used recently"""
        now = time.time()
        for key, last in list(self._last_used.items()):
            if now - last > self._ttl:
                async with self._locks[key]:
                    self._models.pop(key, None)
                self._last_used.pop(key, None)
                self._locks.pop(key, None)
                gc.collect()

    async def get_model(self, model_id: str, adapter_path: str = None) -> BaseTextModel:
        """Get or create a model instance in a thread-safe way"""
        model_key = f"{model_id}:{adapter_path or 'none'}"

        # Create model-specific lock if it doesn't exist
        async with self._main_lock:
            if model_key not in self._locks:
                self._locks[model_key] = asyncio.Lock()

        # Use model-specific lock for actual model operations
        async with self._locks[model_key]:
            if model_key not in self._models:
                self._models[model_key] = load_model(model_id, adapter_path)
            self._last_used[model_key] = time.time()
            return self._models[model_key]


# Global model manager instance
_model_manager = ModelManager()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@router.options("/chat/completions")
@router.options("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest = None):
    """Create a chat completion"""

    # Handle OPTIONS preflight request
    if request is None:
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )

    text_model = await _model_manager.get_model(
        request.model, request.get_extra_params().get("adapter_path")
    )

    if not request.stream:
        completion = text_model.generate(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        try:
            for chunk in text_model.stream_generate(request):
                chunk_data = (
                    f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
                )
                yield chunk_data
                # Force flush by yielding immediately
        except Exception as e:
            # Send error as Server-Sent Event
            error_data = {
                "error": {"message": str(e), "type": "server_error", "code": 500}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Transfer-Encoding": "chunked",
        },
    )
