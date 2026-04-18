from __future__ import annotations

from pathlib import Path

from utils import image_files


def list_simulated_noisy_office_images(root: Path, limit: int | None = None) -> list[Path]:
    images = image_files(root)
    if limit is not None:
        return images[: max(0, limit)]
    return images


def image_group(image_path: Path, root: Path) -> str:
    rel = image_path.relative_to(root)
    return rel.parts[0] if rel.parts else ""

