from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict

from .schemas.requests import GenerationRequest, InterpolationRequest, ModelInfoResponse
from .services.generation_service import generation_service
from .services.training_service import training_service
from .config.settings import settings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Generation API",
    description="API для генерации изображений лиц с помощью GAN и VAE моделей",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для отслеживания тренировки
training_status = {
    "gan": {"status": "not_started", "progress": 0},
    "vae": {"status": "not_started", "progress": 0}
}

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Face Generation API", "status": "running"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Проверка здоровья API и моделей"""
    try:
        model_info = generation_service.get_model_info()
        return {
            "status": "healthy",
            "models": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/generate", tags=["Generation"])
async def generate_images(request: GenerationRequest):
    """Генерация изображений лиц"""
    try:
        print(f"Received generation request: model_type={request.model_type}, num_images={request.num_images}")
        
        results = generation_service.generate_images(
            model_type=request.model_type,
            num_images=request.num_images,
            latent_vector=request.latent_vector
        )
        
        return {
            "model_type": request.model_type,
            "num_generated": len(results),
            "images": results
        }
        
    except ValueError as e:
        logger.error(f"Value error in generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/interpolate", tags=["Generation"])
async def interpolate_images(request: InterpolationRequest):
    """Интерполяция между двумя точками в латентном пространстве"""
    try:
        result = generation_service.interpolate_images(
            model_type=request.model_type,
            start_vector=request.start_vector,
            end_vector=request.end_vector,
            steps=request.steps
        )
        
        return {
            "model_type": request.model_type,
            "steps": request.steps,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка интерполяции: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models/{model_type}", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_type: str):
    """Получение информации о конкретной модели"""
    try:
        model_info = generation_service.get_model_info(model_type)
        
        if model_type not in model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        info = model_info[model_type]
        
        # Проверяем наличие необходимых полей
        if info["status"] == "loaded":
            if "latent_dim" not in info or "image_size" not in info:
                raise HTTPException(status_code=500, detail="Model information incomplete")
        
        return ModelInfoResponse(
            model_type=model_type,
            latent_dim=info.get("latent_dim", 0),
            image_size=info.get("image_size", [0, 0, 0]),
            status=info["status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", tags=["Models"])
async def get_all_models():
    """Получение информации о всех доступных моделях"""
    try:
        return generation_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/gan", tags=["Training"])
async def train_gan(background_tasks: BackgroundTasks):
    """Запуск обучения GAN модели в фоновом режиме"""
    if training_status["gan"]["status"] == "training":
        raise HTTPException(status_code=400, detail="GAN already training")
    
    training_status["gan"] = {"status": "training", "progress": 0}
    
    def gan_training_task():
        try:
            training_service.train_gan()
            training_status["gan"] = {"status": "completed", "progress": 100}
            # Перезагрузка модели после обучения
            generation_service.load_models()
        except Exception as e:
            training_status["gan"] = {"status": "failed", "progress": 0, "error": str(e)}
            logger.error(f"GAN training failed: {e}")
    
    background_tasks.add_task(gan_training_task)
    return {"message": "GAN training started", "status": "training"}

@app.post("/train/vae", tags=["Training"])
async def train_vae(background_tasks: BackgroundTasks):
    """Запуск обучения VAE модели в фоновом режиме"""
    if training_status["vae"]["status"] == "training":
        raise HTTPException(status_code=400, detail="VAE already training")
    
    training_status["vae"] = {"status": "training", "progress": 0}
    
    def vae_training_task():
        try:
            # Обучаем модель
            training_service.train_vae()
            training_status["vae"] = {"status": "completed", "progress": 100}
            
            # Принудительно перезагружаем модели
            logger.info("Перезагрузка моделей после обучения VAE...")
            generation_service.load_models()
            
            # Проверяем загрузку
            model_info = generation_service.get_model_info("vae")
            if model_info["vae"]["status"] == "loaded":
                logger.info("VAE модель успешно перезагружена")
            else:
                logger.error("VAE модель не загрузилась после обучения")
                
        except Exception as e:
            training_status["vae"] = {"status": "failed", "progress": 0, "error": str(e)}
            logger.error(f"VAE training failed: {e}")
    
    background_tasks.add_task(vae_training_task)
    return {"message": "VAE training started", "status": "training"}

@app.post("/train/all", tags=["Training"])
async def train_all_models(background_tasks: BackgroundTasks):
    """Запуск обучения всех моделей"""
    # Проверяем, существуют ли уже модели
    from app.config.settings import settings
    
    if settings.GAN_GENERATOR_PATH.exists() and settings.VAE_ENCODER_PATH.exists():
        raise HTTPException(
            status_code=400, 
            detail="Модели уже обучены. Используйте /generate для генерации изображений."
        )
    
    if training_status["gan"]["status"] == "training" or training_status["vae"]["status"] == "training":
        raise HTTPException(status_code=400, detail="Models are already training")
    
    training_status["gan"] = {"status": "training", "progress": 0}
    training_status["vae"] = {"status": "training", "progress": 0}
    
    def all_training_task():
        try:
            training_service.train_all_models()
            training_status["gan"] = {"status": "completed", "progress": 100}
            training_status["vae"] = {"status": "completed", "progress": 100}
            # Перезагрузка моделей после обучения
            generation_service.load_models()
        except Exception as e:
            training_status["gan"] = {"status": "failed", "progress": 0, "error": str(e)}
            training_status["vae"] = {"status": "failed", "progress": 0, "error": str(e)}
            logger.error(f"All models training failed: {e}")
    
    background_tasks.add_task(all_training_task)
    return {"message": "All models training started", "status": "training"}

@app.get("/train/status", tags=["Training"])
async def get_training_status():
    """Получение статуса обучения моделей"""
    return training_status

# Остальные файлы (schemas, image_utils, base_model) остаются такими же как в предыдущем ответе

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )