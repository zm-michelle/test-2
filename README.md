# PaddleOCR SROIE + SimulatedNoisyOffice Pipeline

This project fine-tunes a native PaddleOCR English recognizer on SROIE, then runs full OCR on every image in `SimulatedNoisyOffice` with PaddleOCR detection plus the fine-tuned recognizer.

## Setup

Use `python3`; this machine does not expose `python` on PATH.

```bash
python3 -m pip install -r requirements.txt
```

If PaddleOCR has already been cloned under `third_party/PaddleOCR`, this project still uses the same dependency set listed in PaddleOCR's own `requirements.txt`, including `scikit-image`, `pyclipper`, `lmdb`, `rapidfuzz`, and `albumentations`.

## Commands

```bash
python3 main.py --train --epochs 10
python3 main.py --test --checkpoint checkpoints/inference_rec
python3 main.py --benchmark --checkpoint checkpoints/inference_rec
python3 main.py --ocr_simulated_noisy_office --checkpoint checkpoints/inference_rec
```

For faster CPU feedback, lower warmup:

```bash
python3 main.py --train --epochs 3 --batch_size 16 --warmup_epochs 0.25
```

Useful smoke-test option:

```bash
python3 main.py --ocr_simulated_noisy_office --checkpoint checkpoints/inference_rec --limit 3
```

## Data

- `SimulatedNoisyOffice` is already present and is inference-only.
- SROIE defaults to `data/SROIE`. If it is not present, the pipeline downloads the public ICDAR-2019-SROIE mirror and prepares recognition crops.

## Outputs

- `outputs/simulated_noisy_office_ocr.json`
- `outputs/simulated_noisy_office_ocr.txt`
- `outputs/simulated_noisy_office_txt/<relative_path>.txt`
