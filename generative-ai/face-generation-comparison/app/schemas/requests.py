from pydantic import BaseModel, Field
from typing import Optional, List
from pydantic.config import ConfigDict

class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str = Field(..., description="Тип модели: 'gan' или 'vae'")
    num_images: int = Field(1, ge=1, le=25, description="Количество изображений для генерации")
    latent_vector: Optional[List[float]] = Field(None, description="Опциональный латентный вектор")

class InterpolationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str = Field(..., description="Тип модели: 'gan' или 'vae'")
    start_vector: Optional[List[float]] = Field(None, description="Начальный латентный вектор")
    end_vector: Optional[List[float]] = Field(None, description="Конечный латентный вектор")
    steps: int = Field(10, ge=2, le=20, description="Количество шагов интерполяции")

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str
    latent_dim: int
    image_size: List[int]
    status: str