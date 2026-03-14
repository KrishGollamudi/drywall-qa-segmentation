
import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from pycocotools.coco import COCO

device    = "cuda" if torch.cuda.is_available() else "cpu"
threshold = 0.5


MODEL_PATH = "/kaggle/working/clipseg_final"  # update path as needed
model     = CLIPSegForImageSegmentation.from_pretrained(MODEL_PATH).to(device)
processor = CLIPSegProcessor.from_pretrained(MODEL_PATH)
model.eval()


def run_inference(image_dir, prompt, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    prompt_slug = prompt.replace(" ", "_")
    times = []

    for image_file in tqdm(image_files, desc=f"Inference: {prompt}"):
        orig_stem  = os.path.splitext(image_file)[0]
        orig_image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")
        orig_w, orig_h = orig_image.size

        inputs = processor(text=prompt, images=orig_image, return_tensors="pt",
                           padding="max_length", max_length=77, truncation=True).to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits
        times.append(time.time() - start)

        pred = torch.sigmoid(logits[0])
        pred_resized = F.interpolate(
            pred.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        ).squeeze()

        mask = (pred_resized > threshold).cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(output_dir, f"{orig_stem}__{prompt_slug}.png"))

    avg_ms = np.mean(times) * 1000
    print(f"Avg inference time: {avg_ms:.1f} ms/image")
    return avg_ms


def evaluate(mask_dir, pred_dir, prompt, coco_ann_file):
    coco = COCO(coco_ann_file)
    id_to_file  = {str(img_id): info['file_name'] for img_id, info in coco.imgs.items()}
    prompt_slug = prompt.replace(" ", "_")
    total_dice = total_iou = 0.0
    count = 0

    for mask_file in tqdm(os.listdir(mask_dir)):
        coco_id = mask_file.split("__")[0]
        if coco_id not in id_to_file:
            continue
        pred_path = os.path.join(pred_dir, f"{coco_id}__{prompt_slug}.png")
        if not os.path.exists(pred_path):
            continue

        gt   = np.array(Image.open(os.path.join(mask_dir, mask_file)).convert("L")).astype(np.float32) / 255.0
        pred = np.array(Image.open(pred_path).convert("L")).astype(np.float32) / 255.0

        if gt.shape != pred.shape:
            pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)).astype(np.float32) / 255.0

        intersection = (pred * gt).sum()
        dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        iou  = (intersection + 1e-6) / (pred.sum() + gt.sum() - intersection + 1e-6)
        total_dice += dice
        total_iou  += iou
        count += 1

    print(f"\n[{prompt}] Dice: {total_dice/count:.4f} | IoU: {total_iou/count:.4f} | Count: {count}")
    return total_dice / count, total_iou / count


if __name__ == "__main__":
    OUTPUT_DIR = "/kaggle/working/predictions"

    # Run inference
    t1 = run_inference(
        "/kaggle/input/datasets/krishnasaigollamudi/cracks/test",
        "segment crack", OUTPUT_DIR
    )
    t2 = run_inference(
        "/kaggle/input/datasets/krishnasaigollamudi/drywall/valid",
        "segment taping area", OUTPUT_DIR
    )

    # Evaluate
    crack_dice, crack_iou = evaluate(
        "/kaggle/working/crack_test_masks", OUTPUT_DIR,
        "segment crack",
        "/kaggle/input/datasets/krishnasaigollamudi/cracks/test/_annotations.coco.json"
    )
    drywall_dice, drywall_iou = evaluate(
        "/kaggle/working/drywall_valid_masks", OUTPUT_DIR,
        "segment taping area",
        "/kaggle/input/datasets/krishnasaigollamudi/drywall/valid/_annotations.coco.json"
    )

    print(f"\n{'='*45}")
    print(f"FINAL RESULTS")
    print(f"{'='*45}")
    print(f"Crack       — Dice: {crack_dice:.4f} | IoU: {crack_iou:.4f}")
    print(f"Taping Area — Dice: {drywall_dice:.4f} | IoU: {drywall_iou:.4f}")
    print(f"mDice: {(crack_dice+drywall_dice)/2:.4f} | mIoU: {(crack_iou+drywall_iou)/2:.4f}")
    print(f"Avg inference: {(t1+t2)/2:.1f} ms/image")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
