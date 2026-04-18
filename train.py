from __future__ import annotations

import subprocess
from pathlib import Path

from config import Paths, TrainingConfig
from datasets.sroie import prepare_sroie_recognition_dataset
from model import ensure_paddleocr_repo, ensure_pretrained_recognizer, export_recognizer, rec_config_path
from utils import DependencyMissing, ensure_dirs, env_with_paddle_flags, python_executable, require_modules


PADDLEOCR_TRAINING_MODULES = (
    "paddle",
    "skimage",
    "shapely",
    "pyclipper",
    "lmdb",
    "tqdm",
    "rapidfuzz",
    "cv2",
    "yaml",
    "albumentations",
    "albucore",
    "packaging",
)


def train_recognizer(paths: Paths, cfg: TrainingConfig, sroie_dir: Path | None = None) -> Path:
    try:
        require_modules(*PADDLEOCR_TRAINING_MODULES)
    except DependencyMissing as exc:
        print(exc)
        raise SystemExit(2) from None

    ensure_dirs([paths.checkpoints_dir, paths.logs_dir, paths.data_dir])
    prepared = prepare_sroie_recognition_dataset(
        sroie_dir or paths.sroie_dir,
        limit=cfg.limit,
        validation_ratio=cfg.validation_ratio,
    )
    repo = ensure_paddleocr_repo(paths)
    pretrained = ensure_pretrained_recognizer(paths)
    save_dir = paths.checkpoints_dir / "rec"

    overrides = [
        f"Global.epoch_num={cfg.epochs}",
        f"Global.save_model_dir={save_dir}",
        f"Global.pretrained_model={pretrained}",
        f"Global.use_gpu={str(cfg.device == 'gpu')}",
        f"Optimizer.lr.warmup_epoch={cfg.warmup_epochs}",
        f"Train.dataset.data_dir={prepared.data_dir}",
        f"Train.dataset.label_file_list=[{prepared.train_label_file}]",
        f"Train.loader.batch_size_per_card={cfg.batch_size}",
        f"Eval.dataset.data_dir={prepared.data_dir}",
        f"Eval.dataset.label_file_list=[{prepared.eval_label_file}]",
    ]
    if cfg.freeze_backbone:
        print("freeze_backbone requested; PaddleOCR config support varies, so only recognition-head fine tuning is not forced automatically.")

    cmd = [python_executable(), "tools/train.py", "-c", str(rec_config_path(paths)), "-o", *overrides]
    subprocess.run(cmd, cwd=str(repo), env=env_with_paddle_flags(cfg.device), check=True)

    checkpoint_prefix = save_dir / "best_accuracy"
    if not checkpoint_prefix.with_suffix(".pdparams").exists():
        raise RuntimeError(
            "Training finished without producing checkpoints/rec/best_accuracy.pdparams. "
            "Check PaddleOCR logs above; common causes are an empty dataset after filtering or a batch size larger than the usable sample count."
        )
    inference_dir = paths.checkpoints_dir / "inference_rec"
    export_recognizer(checkpoint_prefix, inference_dir, paths)
    print(f"Prepared {prepared.crop_count} SROIE recognition crops from {prepared.image_count} images.")
    print(f"Exported recognizer to {inference_dir}")
    return inference_dir
