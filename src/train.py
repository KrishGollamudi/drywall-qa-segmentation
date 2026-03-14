
import os
import time
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


CRACK_IMAGE_DIR   = "/kaggle/input/datasets/krishnasaigollamudi/cracks/train"
CRACK_MASK_DIR    = "/kaggle/working/crack_train_masks"
CRACK_ANN_FILE    = "/kaggle/input/datasets/krishnasaigollamudi/cracks/train/_annotations.coco.json"

DRYWALL_IMAGE_DIR = "/kaggle/input/datasets/krishnasaigollamudi/drywall/train"
DRYWALL_MASK_DIR  = "/kaggle/working/drywall_train_masks"
DRYWALL_ANN_FILE  = "/kaggle/input/datasets/krishnasaigollamudi/drywall/train/_annotations.coco.json"

CHECKPOINT_DIR    = "/kaggle/working"
FINAL_MODEL_PATH  = "/kaggle/working/clipseg_final"

def generate_masks(image_dir, prompt, output_dir, coco_ann_file):
    os.makedirs(output_dir, exist_ok=True)
    coco = COCO(coco_ann_file)
    id_to_file = {str(img_id): info['file_name'] for img_id, info in coco.imgs.items()}
    ann_by_image = {}
    for ann_id, ann in coco.anns.items():
        img_id = str(ann['image_id'])
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    for img_id, img_file in id_to_file.items():
        img_path = os.path.join(image_dir, img_file)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)
        if img_id in ann_by_image:
            for ann in ann_by_image[img_id]:
                rle = coco.annToMask(ann)
                mask = np.maximum(mask, rle * 255)
        prompt_slug = prompt.replace(" ", "_")
        Image.fromarray(mask).save(os.path.join(output_dir, f"{img_id}__{prompt_slug}.png"))
    print(f"✓ Generated masks: {output_dir}")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prompts, coco_ann_file, size=(352, 352)):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.prompts   = prompts
        self.files     = os.listdir(mask_dir)
        self.size      = size
        coco = COCO(coco_ann_file)
        self.id_to_file = {str(img_id): info['file_name'] for img_id, info in coco.imgs.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mask_file  = self.files[idx]
        image_id   = mask_file.split("__")[0]
        image_file = self.id_to_file[image_id]
        image = Image.open(os.path.join(self.image_dir, image_file)).convert("RGB").resize(self.size)
        mask  = Image.open(os.path.join(self.mask_dir, mask_file)).convert("L").resize(self.size, Image.NEAREST)

        # Synchronized spatial augmentations
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask  = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask  = transforms.functional.vflip(mask)
        angle = random.uniform(-15, 15)
        image = transforms.functional.rotate(image, angle)
        mask  = transforms.functional.rotate(mask, angle)

        # Color jitter — image only
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)

        prompt = random.choice(self.prompts)
        inputs = processor(text=prompt, images=image, return_tensors="pt",
                           padding="max_length", max_length=77, truncation=True, do_resize=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(np.array(mask)).float() / 255.0
        return inputs


def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if k in ('input_ids', 'attention_mask'):
            max_len = max(b[k].shape[0] for b in batch)
            padded = []
            for b in batch:
                t = b[k]
                pad_val = processor.tokenizer.pad_token_id if k == 'input_ids' else 0
                padded.append(torch.cat([t, torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)]))
            collated[k] = torch.stack(padded)
        else:
            collated[k] = torch.stack([b[k] for b in batch])
    return collated


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

def combined_loss(logits, labels):
    bce = torch.nn.BCEWithLogitsLoss()(logits, labels)
    dl  = dice_loss(logits, labels)
    return 0.5 * bce + 0.5 * dl


def dice_coef(pred, target, eps=1e-6):
    pred, target = pred.view(-1), target.view(-1)
    return (2 * (pred * target).sum() + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred, target = pred.view(-1), target.view(-1)
    intersection = (pred * target).sum()
    return (intersection + eps) / (pred.sum() + target.sum() - intersection + eps)


if __name__ == "__main__":

    # Model
    model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Generate masks
    print("Generating masks...")
    generate_masks(CRACK_IMAGE_DIR, "segment crack", CRACK_MASK_DIR, CRACK_ANN_FILE)
    generate_masks(DRYWALL_IMAGE_DIR, "segment taping area", DRYWALL_MASK_DIR, DRYWALL_ANN_FILE)

    # Dataset
    crack_dataset = SegmentationDataset(
        CRACK_IMAGE_DIR, CRACK_MASK_DIR,
        ["segment crack", "segment wall crack"], CRACK_ANN_FILE
    )
    drywall_dataset = SegmentationDataset(
        DRYWALL_IMAGE_DIR, DRYWALL_MASK_DIR,
        ["segment taping area", "segment joint/tape", "segment drywall seam"], DRYWALL_ANN_FILE
    )
    dataset = ConcatDataset([crack_dataset, drywall_dataset])
    loader  = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    print(f"Total dataset size: {len(dataset)}")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    threshold   = 0.5
    train_start = time.time()

    def run_epochs(num_epochs, stage="S1"):
        for epoch in range(num_epochs):
            model.train()
            total_loss = total_dice = total_iou = 0.0
            progress = tqdm(loader, desc=f"{stage} Epoch {epoch+1}/{num_epochs}", leave=True)
            for i, batch in enumerate(progress, 1):
                labels = batch.pop("labels").to(device)
                batch  = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits  = outputs.logits
                h, w    = logits.shape[-2], logits.shape[-1]
                labels_resized = F.interpolate(labels.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                loss = combined_loss(logits, labels_resized)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pred_mask   = (torch.sigmoid(logits) > threshold).float()
                    total_loss += loss.item()
                    total_dice += dice_coef(pred_mask, labels_resized)
                    total_iou  += iou_score(pred_mask, labels_resized)
                progress.set_postfix({
                    "Loss": f"{total_loss/i:.4f}",
                    "Dice": f"{total_dice/i:.4f}",
                    "IoU":  f"{total_iou/i:.4f}"
                })
            n = len(loader)
            print(f"{stage} Epoch {epoch+1}/{num_epochs} Loss: {total_loss/n:.4f} "
                  f"Dice: {total_dice/n:.4f} IoU: {total_iou/n:.4f}")
            model.save_pretrained(os.path.join(CHECKPOINT_DIR, f"{stage}_ckpt_ep{epoch+1}"))
            processor.save_pretrained(os.path.join(CHECKPOINT_DIR, f"{stage}_ckpt_ep{epoch+1}"))
            print(f"✓ Saved {stage}_ckpt_ep{epoch+1}")

    print("\n=== Stage 1: lr=1e-5, 10 epochs ===")
    run_epochs(10, stage="S1")

    print("\n=== Stage 2: lr=5e-6, 10 epochs ===")
    for pg in optimizer.param_groups:
        pg['lr'] = 5e-6
    run_epochs(10, stage="S2")

    total_time = (time.time() - train_start) / 60
    print(f"\n✓ Total training time: {total_time:.1f} minutes")
    model.save_pretrained(FINAL_MODEL_PATH)
    processor.save_pretrained(FINAL_MODEL_PATH)
    print("✓ Final model saved")
