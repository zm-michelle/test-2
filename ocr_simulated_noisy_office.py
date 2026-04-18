from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Paths
from datasets.simulated_noisy_office import image_group, list_simulated_noisy_office_images
from model import normalize_checkpoint
from utils import DependencyMissing, choose_device, ensure_dirs, require_modules, safe_relative_txt_path


@dataclass
class OCRLine:
    box: list[list[float]]
    text: str
    confidence: float | None


def run_simulated_noisy_office_ocr(
    paths: Paths,
    checkpoint: str | Path | None,
    output_dir: Path | None = None,
    device: str = "cpu",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    try:
        require_modules("paddleocr", "paddle", "PIL", "numpy")
    except DependencyMissing as exc:
        print(exc)
        raise SystemExit(2) from None

    from paddleocr import PaddleOCR

    output_dir = output_dir or paths.outputs_dir
    txt_dir = output_dir / "simulated_noisy_office_txt"
    ensure_dirs([output_dir, txt_dir])

    checkpoint_dir = normalize_checkpoint(checkpoint, paths)
    _, use_gpu = choose_device(device)
    ocr = _build_paddle_ocr(PaddleOCR, checkpoint_dir, use_gpu)

    image_paths = list_simulated_noisy_office_images(paths.simulated_noisy_office_dir, limit=limit)
    results: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] OCR {image_path.relative_to(paths.root)}", flush=True)
        raw = run_paddle_ocr(ocr, image_path)
        lines = sort_ocr_lines(parse_paddle_result(raw))
        text = reconstruct_text(lines)
        item = {
            "image_path": image_path.relative_to(paths.root).as_posix(),
            "group": image_group(image_path, paths.simulated_noisy_office_dir),
            "text": text,
            "lines": [
                {
                    "text": line.text,
                    "confidence": line.confidence,
                    "box": line.box,
                }
                for line in lines
            ],
        }
        results.append(item)
        per_image_txt = txt_dir / safe_relative_txt_path(image_path, paths.simulated_noisy_office_dir)
        per_image_txt.parent.mkdir(parents=True, exist_ok=True)
        per_image_txt.write_text(text + ("\n" if text else ""), encoding="utf-8")

    (output_dir / "simulated_noisy_office_ocr.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_combined_txt(output_dir / "simulated_noisy_office_ocr.txt", results)
    return results


def _build_paddle_ocr(paddle_ocr_cls: Any, checkpoint_dir: Path, use_gpu: bool) -> Any:
    common = {"lang": "en", "use_angle_cls": True}
    rec_model_name = _read_exported_rec_model_name(checkpoint_dir)
    attempts = [
        {
            **common,
            "det": True,
            "rec": True,
            "use_gpu": use_gpu,
            "rec_model_dir": str(checkpoint_dir),
            "show_log": False,
        },
        {
            "ocr_version": "PP-OCRv3",
            "lang": "en",
            "use_textline_orientation": True,
            "text_recognition_model_name": rec_model_name,
            "text_recognition_model_dir": str(checkpoint_dir),
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return paddle_ocr_cls(**kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize PaddleOCR with custom recognizer: {last_error}") from last_error


def run_paddle_ocr(ocr: Any, image_path: Path) -> Any:
    try:
        return ocr.predict(str(image_path), use_textline_orientation=True)
    except AttributeError:
        return ocr.ocr(str(image_path), det=True, rec=True, cls=True)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return ocr.ocr(str(image_path))


def _read_exported_rec_model_name(checkpoint_dir: Path) -> str:
    inference_yml = checkpoint_dir / "inference.yml"
    if not inference_yml.exists():
        return "en_PP-OCRv3_mobile_rec"
    for line in inference_yml.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("model_name:"):
            return stripped.split(":", 1)[1].strip() or "en_PP-OCRv3_mobile_rec"
    return "en_PP-OCRv3_mobile_rec"


def parse_paddle_result(raw: Any) -> list[OCRLine]:
    if raw is None:
        return []

    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return _parse_v3_result(raw[0])
    if isinstance(raw, dict):
        return _parse_v3_result(raw)

    page = raw[0] if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) else raw
    lines: list[OCRLine] = []
    if not isinstance(page, list):
        return lines
    for entry in page:
        if not entry or len(entry) < 2:
            continue
        box = _as_box(entry[0])
        rec = entry[1]
        if isinstance(rec, (list, tuple)) and rec:
            text = str(rec[0])
            confidence = float(rec[1]) if len(rec) > 1 and rec[1] is not None else None
        else:
            text = str(rec)
            confidence = None
        lines.append(OCRLine(box=box, text=text, confidence=confidence))
    return lines


def _parse_v3_result(result: dict[str, Any]) -> list[OCRLine]:
    boxes = result.get("dt_polys") or result.get("rec_boxes") or result.get("boxes") or []
    texts = result.get("rec_texts") or result.get("texts") or []
    scores = result.get("rec_scores") or result.get("scores") or []
    lines = []
    for idx, text in enumerate(texts):
        box = _as_box(boxes[idx]) if idx < len(boxes) else []
        confidence = float(scores[idx]) if idx < len(scores) and scores[idx] is not None else None
        lines.append(OCRLine(box=box, text=str(text), confidence=confidence))
    return lines


def sort_ocr_lines(lines: list[OCRLine]) -> list[OCRLine]:
    if not lines:
        return []

    def y_center(line: OCRLine) -> float:
        return sum(point[1] for point in line.box) / len(line.box) if line.box else 0.0

    def x_min(line: OCRLine) -> float:
        return min((point[0] for point in line.box), default=0.0)

    heights = []
    for line in lines:
        ys = [point[1] for point in line.box]
        if ys:
            heights.append(max(ys) - min(ys))
    tolerance = (sorted(heights)[len(heights) // 2] if heights else 12.0) * 0.6
    tolerance = max(tolerance, 8.0)

    buckets: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda item: (y_center(item), x_min(item))):
        placed = False
        for bucket in buckets:
            if abs(y_center(bucket[0]) - y_center(line)) <= tolerance:
                bucket.append(line)
                placed = True
                break
        if not placed:
            buckets.append([line])

    ordered: list[OCRLine] = []
    for bucket in buckets:
        ordered.extend(sorted(bucket, key=x_min))
    return ordered


def reconstruct_text(lines: list[OCRLine]) -> str:
    return "\n".join(line.text for line in lines if line.text).strip()


def write_combined_txt(path: Path, results: list[dict[str, Any]]) -> None:
    chunks: list[str] = []
    for item in results:
        chunks.append(f"### {item['image_path']}")
        chunks.append(item.get("text", ""))
    path.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def _as_box(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(not isinstance(x, (list, tuple)) for x in value):
        x1, y1, x2, y2 = [float(x) for x in value]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points = []
    for point in value:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append([float(point[0]), float(point[1])])
    return points
