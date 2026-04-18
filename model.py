from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from config import (
    EN_REC_PRETRAIN_URL,
    PADDLEOCR_GIT_URL,
    PADDLEOCR_REC_CONFIG,
    PADDLEOCR_REC_CONFIG_FALLBACKS,
    Paths,
)
from utils import download_file, extract_tar, python_executable, run_checked


def ensure_paddleocr_repo(paths: Paths) -> Path:
    repo_dir = paths.paddleocr_repo_dir
    if (repo_dir / "tools" / "train.py").exists():
        patch_paddleocr_parallel_env(repo_dir)
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch PaddleOCR training scripts.")
    run_checked(["git", "clone", "--depth", "1", PADDLEOCR_GIT_URL, str(repo_dir)])
    patch_paddleocr_parallel_env(repo_dir)
    return repo_dir


def patch_paddleocr_parallel_env(repo_dir: Path) -> None:
    program_path = repo_dir / "tools" / "program.py"
    if not program_path.exists():
        return
    text = program_path.read_text(encoding="utf-8")
    old = "dist.ParallelEnv().dev_id"
    if old not in text:
        return
    helper = (
        "\n\ndef _paddleocr_parallel_device_id():\n"
        "    env = dist.ParallelEnv()\n"
        "    return getattr(env, 'dev_id', getattr(env, 'device_id', 0))\n"
    )
    if "def _paddleocr_parallel_device_id():" not in text:
        marker = "\ndef check_device(\n"
        text = text.replace(marker, helper + marker, 1)
    text = text.replace(old, "_paddleocr_parallel_device_id()")
    program_path.write_text(text, encoding="utf-8")


def ensure_pretrained_recognizer(paths: Paths) -> Path:
    target_dir = paths.pretrain_dir / "en_PP-OCRv3_rec_train"
    prefix = target_dir / "best_accuracy"
    if prefix.with_suffix(".pdparams").exists():
        return prefix
    archive = paths.pretrain_dir / "en_PP-OCRv3_rec_train.tar"
    download_file(EN_REC_PRETRAIN_URL, archive)
    extract_tar(archive, paths.pretrain_dir)
    if prefix.with_suffix(".pdparams").exists():
        return prefix
    candidates = sorted(paths.pretrain_dir.rglob("*.pdparams"))
    if candidates:
        return candidates[0].with_suffix("")
    raise RuntimeError("Downloaded PaddleOCR recognizer, but no .pdparams file was found.")


def rec_config_path(paths: Paths) -> Path:
    repo = ensure_paddleocr_repo(paths)
    candidates = [PADDLEOCR_REC_CONFIG, *PADDLEOCR_REC_CONFIG_FALLBACKS]
    for rel_path in candidates:
        config = repo / rel_path
        if config.exists():
            return config

    discovered = sorted((repo / "configs" / "rec").rglob("*en*rec*.yml"))
    if discovered:
        return discovered[0]

    raise RuntimeError(
        "PaddleOCR recognition config not found. Checked: "
        + ", ".join(str(repo / rel_path) for rel_path in candidates)
    )


def normalize_checkpoint(checkpoint: str | Path | None, paths: Paths) -> Path:
    if checkpoint is None:
        default = paths.checkpoints_dir / "inference_rec"
        return default
    path = Path(checkpoint)
    if not path.is_absolute():
        path = paths.root / path
    if path.is_dir():
        return path
    if path.with_suffix(".pdparams").exists():
        return path.with_suffix("")
    return path


def export_recognizer(checkpoint_prefix: Path, output_dir: Path, paths: Paths) -> Path:
    repo = ensure_paddleocr_repo(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable(),
        "tools/export_model.py",
        "-c",
        str(rec_config_path(paths)),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix}",
        f"Global.save_inference_dir={output_dir}",
    ]
    subprocess.run(cmd, cwd=str(repo), check=True)
    return output_dir
