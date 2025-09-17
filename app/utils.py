from fastapi import UploadFile
from PIL import Image
import io


async def upload_file_to_image(file: UploadFile):
    """
    Конвертирует UploadFile в PIL Image
    """
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    return image