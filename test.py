from __future__ import annotations

import subprocess
from pathlib import Path

from config import Paths, TrainingConfig
from datasets.sroie import prepare_sroie_recognition_dataset
from model import normalize_checkpoint, rec_config_path, ensure_paddleocr_repo
from train import PADDLEOCR_TRAINING_MODULES
from utils import DependencyMissing, env_with_paddle_flags, python_executable, require_modules


def evaluate_sroie(paths: Paths, cfg: TrainingConfig, checkpoint: str | Path | None, sroie_dir: Path | None = None) -> None:
    try:
        require_modules(*PADDLEOCR_TRAINING_MODULES)
    except DependencyMissing as exc:
        print(exc)
        raise SystemExit(2) from None

    prepared = prepare_sroie_recognition_dataset(sroie_dir or paths.sroie_dir, limit=cfg.limit)
    repo = ensure_paddleocr_repo(paths)
    checkpoint_path = normalize_checkpoint(checkpoint, paths)
    pretrained = checkpoint_path if checkpoint_path.is_file() or checkpoint_path.with_suffix(".pdparams").exists() else checkpoint_path
    cmd = [
        python_executable(),
        "tools/eval.py",
        "-c",
        str(rec_config_path(paths)),
        "-o",
        f"Global.pretrained_model={pretrained}",
        f"Global.use_gpu={str(cfg.device == 'gpu')}",
        f"Eval.dataset.data_dir={prepared.data_dir}",
        f"Eval.dataset.label_file_list=[{prepared.eval_label_file}]",
        f"Eval.loader.batch_size_per_card={cfg.batch_size}",
    ]
    subprocess.run(cmd, cwd=str(repo), env=env_with_paddle_flags(cfg.device), check=True)
