from ultralytics import YOLO
import os
from pathlib import Path
from PIL import Image
import cv2


def main():
    model = YOLO('../wheights/best.pt')

    images_dir = '../my_val_dataset_labeled/images'
    output_dir = '../draw_my_val_dataset_labeled/'
    os.makedirs(output_dir, exist_ok=True)

    image_paths = list(Path(images_dir).glob('*.*'))

    for img_path in image_paths:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue


        results = model(str(img_path))
        result = results[0]


        annotated_image_bgr = result.plot()  #BGR format


        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)


        output_path = Path(output_dir) / img_path.name
        pil_image = Image.fromarray(annotated_image_rgb)
        pil_image.save(output_path)

        print(f"Сохранено: {output_path}")


if __name__ == '__main__':
    main()