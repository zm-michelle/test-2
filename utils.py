from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from config import SUPPORTED_IMAGE_SUFFIXES


class DependencyMissing(RuntimeError):
    pass


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def require_modules(*module_names: str) -> None:
    missing: list[str] = []
    import_names = {
        "scikit-image": "skimage",
        "opencv-python": "cv2",
        "opencv-contrib-python": "cv2",
        "pillow": "PIL",
        "pyyaml": "yaml",
    }
    for module_name in module_names:
        try:
            importlib.import_module(import_names.get(module_name, module_name))
        except Exception:
            missing.append(module_name)
    if missing:
        install_names = {
            "paddle": "paddlepaddle",
            "paddleocr": "paddleocr",
            "PIL": "pillow",
            "cv2": "opencv-python",
            "numpy": "numpy",
            "skimage": "scikit-image",
        }
        packages = " ".join(install_names.get(name, name) for name in missing)
        raise DependencyMissing(
            "Missing OCR dependencies: "
            + ", ".join(missing)
            + f"\nInstall them with: python3 -m pip install {packages}"
        )


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def safe_relative_txt_path(image_path: Path, base_dir: Path) -> Path:
    rel = image_path.relative_to(base_dir)
    safe_parts = [sanitize_path_part(part) for part in rel.parts]
    return Path(*safe_parts).with_suffix(".txt")


def sanitize_path_part(part: str) -> str:
    keep = []
    for ch in part:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def choose_device(requested: str) -> tuple[str, bool]:
    requested = (requested or "cpu").lower()
    if requested == "gpu":
        try:
            import paddle

            if paddle.device.is_compiled_with_cuda():
                return "gpu", True
        except Exception:
            pass
        print("GPU requested but unavailable; using CPU.", flush=True)
    return "cpu", False


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)


def extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(destination)


def copytree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def python_executable() -> str:
    return sys.executable or "python3"


def env_with_paddle_flags(device: str) -> dict[str, str]:
    env = os.environ.copy()
    if device.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env
