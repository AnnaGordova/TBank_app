from fastapi import FastAPI, File, UploadFile, HTTPException
from app.models import DetectionResponse, ErrorResponse
from app.detector import init_detector
from app.utils import upload_file_to_image
import logging
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="T-Bank Logo Detection API",
    description="API для детекции логотипов Т-Банка на изображениях с автоматическим выбором CPU/GPU",
    version="1.0.0"
)

#глобальная переменная для хранения детектора
detector = None


#инициализация модели при запуске
@app.on_event("startup")
async def startup_event():
    global detector

    try:
        model_path = "wheights/best.pt"


        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file {model_path} not found")


        logger.info("Detecting available devices...")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA GPU(s)")
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Found Apple MPS (Metal) support")
        else:
            logger.info("Using CPU (no GPU available)")

        #инициализируем детектор
        detector = init_detector(model_path)
        logger.info("Model loaded successfully")


        device_info = detector.get_device_info()
        logger.info(f"Running on: {device_info['device_type']}")
        if device_info['device'] == "cuda":
            logger.info(f"GPU: {device_info['gpu_name']}")
            logger.info(f"GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.post("/detect",
          response_model=DetectionResponse,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    Поддерживаемые форматы: JPEG, PNG, BMP, WEBP
    """
    global detector

    #проверка типа файла
    allowed_content_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_content_types)}"
        )

    try:
        if detector is None:
            raise HTTPException(
                status_code=500,
                detail="Model not initialized. Please check server logs."
            )


        image = await upload_file_to_image(file)


        detections = detector.detect(image)

        return DetectionResponse(detections=detections)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint с информацией об устройстве"""
    global detector
    status = "healthy" if detector is not None else "unhealthy"

    response = {
        "status": status,
        "model_loaded": detector is not None
    }

    if detector is not None:
        device_info = detector.get_device_info()
        response.update({
            "device": device_info["device"],
            "device_type": device_info["device_type"],
            "performance": "high" if device_info["device"] != "cpu" else "standard"
        })

        if device_info["device"] == "cuda":
            response.update({
                "gpu_name": device_info["gpu_name"],
                "gpu_memory_gb": round(device_info["gpu_memory_gb"], 1)
            })

    return response


@app.get("/device-info")
async def device_info():
    """Информация о вычислительном устройстве"""
    global detector

    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    return detector.get_device_info()


@app.get("/supported-formats")
async def supported_formats():
    """Возвращает поддерживаемые форматы изображений"""
    return {
        "supported_formats": ["JPEG", "PNG", "BMP", "WEBP"],
        "mime_types": ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    }


@app.post("/validate")
async def validate_model():
    """
    Запуск валидации модели на тестовом датасете
    """
    global detector

    try:
        if detector is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # Запускаем валидацию
        result = detector.validate_model("my_val_dataset_labeled/data.yaml")

        return result

    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validation-config")
async def validation_config():
    """
    Проверка конфигурации валидационного датасета
    """
    config_path = "my_val_dataset_labeled/data.yaml"

    config_exists = os.path.exists(config_path)
    response = {
        "config_path": config_path,
        "config_exists": config_exists,
        "validation_data_exists": os.path.exists("my_val_dataset_labeled")
    }

    if config_exists:
        # Проверяем структуру датасета
        response["images_exist"] = os.path.exists("my_val_dataset_labeled/images/")
        response["labels_exist"] = os.path.exists("my_val_dataset_labeled/labels/")

        try:
            with open(config_path, 'r') as f:
                content = f.read()
                response["config_preview"] = content[:200] + "..." if len(content) > 200 else content
        except:
            response["config_error"] = "Cannot read config file"

    return response

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о API"""
    global detector

    device_type = "Unknown"
    if detector is not None:
        device_info = detector.get_device_info()
        device_type = device_info["device_type"]

    return {
        "message": "T-Bank Logo Detection API",
        "version": "1.0.0",
        "status": "ready" if detector is not None else "initializing",
        "device": device_type,
        "supported_formats": ["JPEG", "PNG", "BMP", "WEBP"],
        "endpoints": {
            "docs": "/docs",
            "detect": "/detect",
            "health": "/health",
            "device_info": "/device-info",
            "supported_formats": "/supported-formats",
            "validate": "/validate",
            "validation-config": "/validation-config"
        }
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    #альтернативный запуск из корневой директории
    #uvicorn app.main:app --reload --port 8000
    #http://127.0.0.1:8000/docs
