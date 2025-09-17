from ultralytics import YOLO
import torch
import os


def setup_environment():
    os.environ['NUM_WORKERS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '1'


def train_yolo():
    try:
        model = YOLO('yolov8n.pt')


        train_args = {
            'data': 'yolo_dataset/data.yaml',
            'epochs': 100,
            'patience' : 50,
            'imgsz': 640,
            'batch': 16,
            'workers': 2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'verbose': True,
            'cache': False,
            'close_mosaic': 5,
            'save' : True,
            'name' : 'exp2 (with augs)',
            'val' : True,
            'split' : 'val',
            'project' : 'checks',
            'plots' : True,
            'show_labels' : True,
            'show_conf' : True,
            'visualize' : False,  
            'pretrained' : True,
            'save_period' : -1,
            # НАСТРОЙКИ АУГМЕНТАЦИИ:
            'degrees' : 15.0,  #максимальный угол поворота ±15 градусов
            'translate' : 0.2,  #сдвиг на до 20% от размера изображения
            'scale' : 0.3,  #масштабирование от 70% до 130%
            'shear' : 15.0,  #перспектива
            'perspective' : 0.001,  #коэффициент перспективного искажения (малое значение -> сильное искажение)
            'flipud' : 0.0,
            'fliplr' : 0.0,
            'mosaic' : 1.0,  #mosaic аугментация (100% на первых эпохах)
            'mixup' : 0.2,  #mixUp аугментация 20%
        }

        results = model.train(**train_args)
        return results

    except Exception as e:
        print(f"Training error: {e}")
        return None


if __name__ == '__main__':
    print("Setting up environment for Windows...")
    setup_environment()

    print("Starting training...")
    results = train_yolo()

    if results:
        print("Training completed successfully!")
    else:
        print("Training failed!")

