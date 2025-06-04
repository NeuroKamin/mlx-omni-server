from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Model(BaseModel):
    """Model information as per OpenAI API specification"""

    id: str = Field(..., description="The model identifier")
    object: str = Field(default="model", description="The object type (always 'model')")
    created: int = Field(
        ..., description="Unix timestamp of when the model was created"
    )
    owned_by: str = Field(..., description="Organization that owns the model")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Model configuration from config.json"
    )


class ModelList(BaseModel):
    """Response format for list of models"""

    object: str = Field(default="list", description="The object type (always 'list')")
    data: List[Model] = Field(..., description="List of model objects")


class ModelDeletion(BaseModel):
    """Response format for model deletion"""

    id: str = Field(..., description="The ID of the deleted model")
    object: str = Field(default="model", description="The object type (always 'model')")
    deleted: bool = Field(..., description="Whether the model was deleted")


class ModelDownloadRequest(BaseModel):
    """Request body for starting a model download."""

    model: str = Field(..., description="Model ID from Hugging Face")


class ModelDownloadResponse(BaseModel):
    """Response for a started model download."""

    id: str = Field(..., description="Download task identifier")
    status: str = Field(..., description="Current status of the task")


class ModelDownloadStatus(BaseModel):
    """Status information about a download task."""

    id: str = Field(..., description="Download task identifier")
    status: str = Field(..., description="Current status")
    error: Optional[str] = Field(default=None, description="Error message if any")
