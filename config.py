from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    sroie_dir: Path = PROJECT_ROOT / "data" / "SROIE"
    simulated_noisy_office_dir: Path = PROJECT_ROOT / "SimulatedNoisyOffice"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    logs_dir: Path = PROJECT_ROOT / "logs"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    paddleocr_repo_dir: Path = PROJECT_ROOT / "third_party" / "PaddleOCR"
    pretrain_dir: Path = PROJECT_ROOT / "pretrain_models"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    device: str = "cpu"
    freeze_backbone: bool = False
    limit: int | None = None
    validation_ratio: float = 0.15


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


PADDLEOCR_GIT_URL = "https://github.com/PaddlePaddle/PaddleOCR.git"
SROIE_GIT_ZIP_URL = "https://github.com/zzzDavid/ICDAR-2019-SROIE/archive/refs/heads/master.zip"
EN_REC_PRETRAIN_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
PADDLEOCR_REC_CONFIG = "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml"
PADDLEOCR_REC_CONFIG_FALLBACKS = (
    "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
    "configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml",
    "configs/rec/multi_language/rec_en_number_lite_train.yml",
)
