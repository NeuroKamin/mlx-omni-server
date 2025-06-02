import json
import asyncio
import contextvars
from typing import Generator, Dict
from contextlib import asynccontextmanager

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .mlx.models import load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel

router = APIRouter(tags=["chat—completions"])


class ModelManager:
    """Thread-safe model manager for handling concurrent requests"""
    
    def __init__(self):
        self._models: Dict[str, BaseTextModel] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._main_lock = asyncio.Lock()
    
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
            return self._models[model_key]

# Global model manager instance
_model_manager = ModelManager()

@router.get("/v1/test")
async def test():
    return JSONResponse(content={"message": "Hello, World!"})

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
            }
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
                chunk_data = f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
                yield chunk_data
                # Force flush by yielding immediately
        except Exception as e:
            # Send error as Server-Sent Event
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
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
