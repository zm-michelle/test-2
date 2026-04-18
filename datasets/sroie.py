from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import SROIE_GIT_ZIP_URL, SUPPORTED_IMAGE_SUFFIXES
from utils import copytree_contents, download_file, extract_zip, image_files


@dataclass(frozen=True)
class SROIEBox:
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    text: str


@dataclass(frozen=True)
class PreparedSROIE:
    data_dir: Path
    train_label_file: Path
    eval_label_file: Path
    image_count: int
    crop_count: int


def ensure_sroie_dataset(sroie_dir: Path) -> Path:
    if discover_sroie_pairs(sroie_dir):
        return sroie_dir

    archive = sroie_dir.parent / "ICDAR-2019-SROIE-master.zip"
    tmp_extract = sroie_dir.parent / "_sroie_download"
    download_file(SROIE_GIT_ZIP_URL, archive)
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    extract_zip(archive, tmp_extract)

    candidates = [p for p in tmp_extract.rglob("*") if p.is_dir() and discover_sroie_pairs(p)]
    if not candidates:
        raise RuntimeError(
            "Downloaded SROIE archive, but no OCR annotation layout was found. "
            "Place SROIE manually under data/SROIE with images and bbox .txt files."
        )
    if sroie_dir.exists():
        shutil.rmtree(sroie_dir)
    copytree_contents(candidates[0], sroie_dir)
    shutil.rmtree(tmp_extract)
    return sroie_dir


def prepare_sroie_recognition_dataset(
    sroie_dir: Path,
    limit: int | None = None,
    validation_ratio: float = 0.15,
) -> PreparedSROIE:
    sroie_dir = ensure_sroie_dataset(sroie_dir)
    pairs = discover_sroie_pairs(sroie_dir)
    if limit is not None:
        pairs = pairs[: max(0, limit)]
    if not pairs:
        raise RuntimeError(f"No SROIE image/annotation pairs found in {sroie_dir}")

    processed_dir = sroie_dir / "processed"
    crops_dir = processed_dir / "rec_images"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for image_path, annotation_path in pairs:
        boxes = parse_sroie_annotation(annotation_path)
        if not boxes:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for idx, box in enumerate(boxes):
                crop = crop_box(image, box)
                if crop.width <= 0 or crop.height <= 0:
                    continue
                rel_crop = Path("rec_images") / f"{image_path.stem}_{idx:04d}.png"
                crop.save(processed_dir / rel_crop)
                rows.append((rel_crop.as_posix(), box.text))

    if not rows:
        raise RuntimeError("SROIE annotations were found, but no recognition crops were produced.")

    rng = random.Random(1337)
    rng.shuffle(rows)
    split = max(1, round(len(rows) * (1 - validation_ratio)))
    if split >= len(rows):
        split = max(1, len(rows) - 1)
    train_rows = rows[:split]
    eval_rows = rows[split:] or rows[-1:]

    train_label = processed_dir / "rec_gt_train.txt"
    eval_label = processed_dir / "rec_gt_eval.txt"
    write_paddle_rec_labels(train_label, train_rows)
    write_paddle_rec_labels(eval_label, eval_rows)

    return PreparedSROIE(
        data_dir=processed_dir,
        train_label_file=train_label,
        eval_label_file=eval_label,
        image_count=len(pairs),
        crop_count=len(rows),
    )


def discover_sroie_pairs(root: Path) -> list[tuple[Path, Path]]:
    canonical_img_dir = root / "data" / "img"
    canonical_box_dir = root / "data" / "box"
    if canonical_img_dir.exists() and canonical_box_dir.exists():
        canonical_pairs = []
        box_by_stem = {path.stem: path for path in canonical_box_dir.glob("*.csv")}
        for image_path in image_files(canonical_img_dir):
            annotation = box_by_stem.get(image_path.stem)
            if annotation and _looks_like_sroie_annotation(annotation):
                canonical_pairs.append((image_path, annotation))
        if canonical_pairs:
            return sorted(canonical_pairs)

    images = image_files(root)
    annotations = [
        path
        for pattern in ("*.csv", "*.txt")
        for path in root.rglob(pattern)
        if _is_candidate_annotation(path)
    ]
    annotation_by_stem: dict[str, list[Path]] = {}
    for annotation in annotations:
        if _looks_like_sroie_annotation(annotation):
            annotation_by_stem.setdefault(annotation.stem, []).append(annotation)

    pairs: list[tuple[Path, Path]] = []
    for image_path in images:
        candidates = annotation_by_stem.get(image_path.stem)
        if candidates:
            pairs.append((image_path, sorted(candidates, key=lambda p: len(p.parts))[0]))
    return sorted(pairs)


def _is_candidate_annotation(path: Path) -> bool:
    excluded_parts = {"processed", "test_result", "__pycache__"}
    if any(part in excluded_parts for part in path.parts):
        return False
    if path.name in {"rec_gt_train.txt", "rec_gt_eval.txt"}:
        return False
    return True


def parse_sroie_annotation(annotation_path: Path) -> list[SROIEBox]:
    boxes: list[SROIEBox] = []
    with annotation_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = raw_line.split(",", 8)
            if len(row) != 9:
                row = next(csv.reader([raw_line]), [])
            if len(row) < 9:
                continue
            try:
                coords = [int(float(value.strip())) for value in row[:8]]
            except ValueError:
                continue
            text = row[8].replace("\t", " ").replace("\r", " ").replace("\n", " ").strip().strip('"')
            if not text:
                continue
            points = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            )
            boxes.append(SROIEBox(points=points, text=text))
    return boxes


def crop_box(image: Image.Image, box: SROIEBox) -> Image.Image:
    xs = [point[0] for point in box.points]
    ys = [point[1] for point in box.points]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(image.width, max(xs))
    bottom = min(image.height, max(ys))
    return image.crop((left, top, right, bottom))


def write_paddle_rec_labels(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rel_path, text in rows:
            fh.write(f"{rel_path}\t{text}\n")


def _has_sroie_annotations(root: Path) -> bool:
    return root.exists() and any(_looks_like_sroie_annotation(path) for path in root.rglob("*.txt"))


def _looks_like_sroie_annotation(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 9:
                    continue
                try:
                    [int(float(value.strip())) for value in row[:8]]
                    return True
                except ValueError:
                    continue
    except OSError:
        return False
    return False
