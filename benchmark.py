from __future__ import annotations

import json
import time
from pathlib import Path

from config import Paths, TrainingConfig
from ocr_simulated_noisy_office import run_simulated_noisy_office_ocr
from test import evaluate_sroie


def benchmark_sroie(paths: Paths, cfg: TrainingConfig, checkpoint: str | Path | None, sroie_dir: Path | None = None) -> None:
    start = time.perf_counter()
    evaluate_sroie(paths, cfg, checkpoint, sroie_dir=sroie_dir)
    elapsed = time.perf_counter() - start
    print(json.dumps({"dataset": "SROIE", "elapsed_seconds": elapsed}, indent=2))


def benchmark_simulated_noisy_office(
    paths: Paths,
    checkpoint: str | Path | None,
    output_dir: Path | None = None,
    device: str = "cpu",
    limit: int | None = None,
) -> dict:
    start = time.perf_counter()
    results = run_simulated_noisy_office_ocr(
        paths=paths,
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        limit=limit,
    )
    elapsed = time.perf_counter() - start
    confidences = [
        line["confidence"]
        for item in results
        for line in item.get("lines", [])
        if line.get("confidence") is not None
    ]
    summary = {
        "dataset": "SimulatedNoisyOffice",
        "image_count": len(results),
        "total_ocr_seconds": elapsed,
        "average_ocr_seconds": elapsed / len(results) if results else 0.0,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "empty_output_count": sum(1 for item in results if not item.get("text")),
    }
    print(json.dumps(summary, indent=2))
    return summary

