from fastapi import APIRouter, HTTPException, Request

from .model_loader import ModelLoader
from .models_service import ModelsService
from .schema import (
    Model,
    ModelDeletion,
    ModelList,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelDownloadStatus,
)

router = APIRouter(tags=["models"])
models_service = ModelsService()
model_loader = ModelLoader()


def extract_model_id_from_path(request: Request) -> str:
    """Extract full model ID from request path"""
    path = request.url.path
    prefix = "/v1/models/" if "/v1/models/" in path else "/models/"
    return path[len(prefix) :]


def handle_model_error(e: Exception) -> None:
    """Handle model-related errors and raise appropriate HTTP exceptions"""
    if isinstance(e, ValueError):
        raise HTTPException(status_code=404, detail=str(e))
    print(f"Error processing request: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelList)
@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """
    Lists the currently available models, and provides basic information about each one
    such as the owner and availability.
    """
    try:
        return models_service.list_models()
    except Exception as e:
        handle_model_error(e)

@router.get("/models/rescan", response_model=ModelList)
@router.get("/v1/models/rescan", response_model=ModelList)
async def rescan_models() -> ModelList:
    """
    Rescan the local cache for models.
    """
    try:
        models_service._scan_models()
        return models_service.list_models()
    except Exception as e:
        handle_model_error(e)

@router.get("/models/{model_id:path}", response_model=Model)
@router.get("/v1/models/{model_id:path}", response_model=Model)
async def get_model(request: Request) -> Model:
    """
    Retrieves a model instance, providing basic information about the model such as
    the owner and permissioning.
    """
    try:
        model_id = extract_model_id_from_path(request)
        model = models_service.get_model(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found")
        return model
    except Exception as e:
        handle_model_error(e)


@router.delete("/models/{model_id:path}", response_model=ModelDeletion)
@router.delete("/v1/models/{model_id:path}", response_model=ModelDeletion)
async def delete_model(request: Request) -> ModelDeletion:
    """
    Delete a fine-tuned model from local cache.
    """
    try:
        model_id = extract_model_id_from_path(request)
        return models_service.delete_model(model_id)
    except Exception as e:
        handle_model_error(e)


@router.post("/models/load", response_model=ModelDownloadResponse)
@router.post("/v1/models/load", response_model=ModelDownloadResponse)
async def load_model(request: ModelDownloadRequest) -> ModelDownloadResponse:
    """Start background download of a model."""
    task_id = model_loader.start(request.model)
    return ModelDownloadResponse(id=task_id, status="in_progress")


@router.get("/models/load/{task_id}", response_model=ModelDownloadStatus)
@router.get("/v1/models/load/{task_id}", response_model=ModelDownloadStatus)
async def get_load_status(task_id: str) -> ModelDownloadStatus:
    """Get status of a background model download."""
    status = model_loader.get_status(task_id)
    return ModelDownloadStatus(id=task_id, **status)
