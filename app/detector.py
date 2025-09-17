from ultralytics import YOLO
from app.models import BoundingBox, Detection
import torch
import os
from PIL import Image
from typing import List, Dict, Any


class LogoDetector:
    def __init__(self, model_path: str):
        """Инициализация YOLO модели с автоматическим выбором устройства"""
        #выбираем вычислитель
        self.device = self._get_available_device()
        print(f"Using device: {self.device}")

        #загружаем модель
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        print(f"Model loaded from: {model_path} on {self.device}")

    def _get_available_device(self):
        """Автоматически определяет доступное устройство"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Found {gpu_count} GPU(s): {gpu_name}")

                #память GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                print(f"GPU Memory: {gpu_memory:.1f} GB")

                return "cuda"

            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Found Apple MPS (Metal Performance Shaders)")
                return "mps"

            else:
                print("Using CPU (GPU not available)")
                return "cpu"

        except Exception as e:
            print(f"Error detecting device: {e}, falling back to CPU")
            return "cpu"

    def get_device_info(self):
        """Возвращает информацию об устройстве"""
        info = {
            "device": self.device,
            "device_type": "GPU" if self.device == "cuda" else "MPS" if self.device == "mps" else "CPU"
        }

        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
                "cuda_version": torch.version.cuda
            })

        return info

    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Детекция логотипов Т-Банка на изображении
        """
        try:
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )

            detections = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

                        bbox = BoundingBox(
                            x_min=int(x_min),
                            y_min=int(y_min),
                            x_max=int(x_max),
                            y_max=int(y_max)
                        )
                        detections.append(Detection(bbox=bbox))

            print(f"Found {len(detections)} detections on {self.device}")
            return detections

        except Exception as e:
            print(f"Detection error on {self.device}: {e}")
            raise

    def validate_model(self, data_config: str = "../validation_data/data.yaml") -> Dict[str, Any]:
        """
        Валидация модели на тестовом датасете
        """
        try:
            if not os.path.exists(data_config):
                return {
                    "status": "error",
                    "error": f"Data config not found: {data_config}",
                    "config_exists": False
                }

            print(f"Starting validation on {self.device} with config: {data_config}")


            results = self.model.val(
                data=data_config,
                split='test',
                verbose=False,
                device=self.device
            )

            # Форматируем метрики
            metrics = {
                "status": "success",
                "metrics": {
                    "precision (IoU=0.5)": float(results.box.mp),
                    "recall (IoU=0.5)": float(results.box.mr),
                    "f1_score (IoU=0.5)": float(results.box.f1[0]),
                    "map50": float(results.box.map50),
                    "map50_95": float(results.box.map)
                },
                "device_used": self.device,
                "config_path": data_config
            }

            print("Validation completed")
            return metrics

        except Exception as e:
            print(f"Validation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "config_path": data_config
            }


#глобальный экземпляр
_detector_instance = None


def get_detector():
    """Получить экземпляр детектора"""
    global _detector_instance
    if _detector_instance is None:
        raise RuntimeError("Detector not initialized. Call init_detector() first.")
    return _detector_instance


def init_detector(model_path: str):
    """Инициализация детектора"""
    global _detector_instance
    try:
        _detector_instance = LogoDetector(model_path)
        print("Detector initialized successfully")
        return _detector_instance
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        raise