from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import benchmark_simulated_noisy_office, benchmark_sroie
from config import Paths, TrainingConfig
from ocr_simulated_noisy_office import run_simulated_noisy_office_ocr
from test import evaluate_sroie
from train import train_recognizer
from utils import ensure_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaddleOCR SROIE fine-tuning and SimulatedNoisyOffice OCR pipeline")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--train", action="store_true", help="Fine-tune the recognizer on SROIE")
    actions.add_argument("--test", action="store_true", help="Evaluate the recognizer on SROIE")
    actions.add_argument("--benchmark", action="store_true", help="Run benchmark reporting")
    actions.add_argument("--ocr_simulated_noisy_office", action="store_true", help="Run full OCR on SimulatedNoisyOffice")
    parser.add_argument("--checkpoint", default=None, help="Exported recognizer dir or Paddle checkpoint prefix")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=float, default=1.0, help="Learning-rate warmup epochs for PaddleOCR training")
    parser.add_argument("--eval_every_steps", type=int, default=200, help="Validation/checkpoint frequency during training")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sroie_dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit images for smoke tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = Paths()
    ensure_dirs([paths.data_dir, paths.checkpoints_dir, paths.logs_dir, paths.outputs_dir])
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        eval_every_steps=args.eval_every_steps,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        limit=args.limit,
    )
    sroie_dir = Path(args.sroie_dir).resolve() if args.sroie_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if args.train:
        train_recognizer(paths, cfg, sroie_dir=sroie_dir)
    elif args.test:
        evaluate_sroie(paths, cfg, args.checkpoint, sroie_dir=sroie_dir)
    elif args.benchmark:
        if args.checkpoint:
            benchmark_sroie(paths, cfg, args.checkpoint, sroie_dir=sroie_dir)
            benchmark_simulated_noisy_office(paths, args.checkpoint, output_dir=output_dir, device=args.device, limit=args.limit)
        else:
            benchmark_simulated_noisy_office(paths, args.checkpoint, output_dir=output_dir, device=args.device, limit=args.limit)
    elif args.ocr_simulated_noisy_office:
        run_simulated_noisy_office_ocr(paths, args.checkpoint, output_dir=output_dir, device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
