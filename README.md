# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation of construction defects using fine-tuned CLIPSeg.

## Task

Given an image and a natural-language prompt, produce a binary mask for:
- `"segment crack"` — wall/surface cracks
- `"segment taping area"` — drywall joint/tape regions

## Results

| Prompt | Dice | IoU |
|---|---|---|
| segment crack | 0.5503 | 0.4051 |
| segment taping area | 0.6128 | 0.4581 |
| **Mean** | **0.5816** | **0.4316** |

- **Train time:** 108.5 minutes (20 epochs, Kaggle T4 GPU)
- **Avg inference:** ~40.9 ms/image
- **Model size:** 150.7M parameters

## Model

**CLIPSeg** (`CIDAS/clipseg-rd64-refined`) — a vision-language model that conditions segmentation on free-form text prompts via CLIP embeddings. Selected for its open-vocabulary capability allowing multi-prompt defect detection without separate models per class.

## Reproducibility

| Setting | Value |
|---|---|
| Seeds | `torch=42, random=42, numpy=42` |
| Optimizer | AdamW (`lr=1e-5` → `5e-6`, `weight_decay=1e-2`) |
| Loss | `0.5 × BCE + 0.5 × Dice` |
| Epochs | 20 (10 Stage 1 + 10 Stage 2) |
| Batch size | 4 |
| Input resolution | 352×352 |
| Hardware | Kaggle T4 GPU |

## Datasets

- [Drywall-Join-Detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) — 820 train, 202 val
- [Cracks-3ii36](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) — 4,027 train, 268 test

## Prediction Masks

- Format: single-channel PNG, values `{0, 255}`
- Filename: `{image_id}__{prompt_slug}.png`
- Example: `251__segment_crack.png`

## Setup

```bash
pip install torch torchvision transformers pycocotools tqdm Pillow
```

## Usage

```python
# Train
python src/train.py

# Inference
python src/inference.py --model_path /path/to/model --image_dir /path/to/images --prompt "segment crack"
```

## Report

See `report/report.pdf` for full methodology, results, visual examples, and failure analysis.

## Repo Structure

```
├── README.md
├── src/
│   ├── train.py          # Full training pipeline
│   └── inference.py      # Inference + mask saving
└── report/
|    ├── report.pdf        # Full PDF report
|    ├── report.tex        # LaTeX source
|    └── visuals_*.png     # Visual examples
|___ weights  #           model weights
```
