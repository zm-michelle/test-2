from __future__ import annotations

from pathlib import Path

from datasets.sroie import PreparedSROIE, prepare_sroie_recognition_dataset


def prepare_training_data(sroie_dir: Path, limit: int | None = None) -> PreparedSROIE:
    return prepare_sroie_recognition_dataset(sroie_dir, limit=limit)

