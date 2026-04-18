from __future__ import annotations

import random
from PIL import Image, ImageEnhance, ImageFilter


def augment_document_crop(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.35))
    if random.random() < 0.35:
        image = image.rotate(
            random.uniform(-2.0, 2.0),
            expand=True,
            fillcolor=(255, 255, 255),
        )
    return image


def resize_preserve_aspect(image: Image.Image, height: int = 32) -> Image.Image:
    width, old_height = image.size
    if old_height <= 0:
        return image
    new_width = max(1, round(width * (height / old_height)))
    return image.resize((new_width, height), Image.Resampling.BICUBIC)

