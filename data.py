from __future__ import annotations

from pathlib import Path

from datasets.sroie import PreparedSROIE, prepare_sroie_recognition_dataset
from datasets.simulated_noisy_office import list_simulated_noisy_office_images


def prepare_sroie(sroie_dir: Path, limit: int | None = None) -> PreparedSROIE:
    return prepare_sroie_recognition_dataset(sroie_dir, limit=limit)


def list_noisy_office(root: Path, limit: int | None = None) -> list[Path]:
    return list_simulated_noisy_office_images(root, limit=limit)

