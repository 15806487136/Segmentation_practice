"""
segmentation_improvements.py

轻量脚本：在本地复现 notebook 的合成数据流程，加入数据增强、Dice+CE 复合损失、冻结 backbone 选项、学习率调度与梯度累积设置。
用法示例（在 med_phys 环境下运行）:
  /home/zhouyang/miniconda3/envs/med_phys/bin/python medical_physics/segmentation_improvements.py --quick

快速模式 (--quick) 会用很小的数据集和 1 个 epoch 做 smoke test。
"""
import argparse
import random
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer,
)


PRETRAINED_SEGFORMER_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


def make_synthetic_example(size=(128, 128), seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    mask = Image.new("L", size, 255)  # 255 as ignore/background
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # draw 1-3 random shapes with labels 0,1,2
    num_shapes = random.randint(1, 3)
    for i in range(num_shapes):
        label = i % 3
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        x1 = random.randint(w // 2, w - 1)
        y1 = random.randint(h // 2, h - 1)
        color = [(120, 160, 255), (180, 180, 180), (255, 200, 120)][label]
        draw.ellipse([x0, y0, x1, y1], fill=color)
        mask_draw.ellipse([x0, y0, x1, y1], fill=label)

    return img, mask


class SimpleSegDataset(Dataset):
    def __init__(self, size=128, n=100, processor=None, transforms=None, seed=0):
        self.size = (size, size)
        self.n = n
        self.processor = processor
        self.transforms = transforms
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, mask = make_synthetic_example(size=self.size, seed=(self.seed + idx))
        if self.transforms:
            img, mask = self.transforms(img, mask)
        # processor expects list/sequence of images for batched processing
        inputs = self.processor(images=img, return_tensors="pt")
        # processor returns pixel_values shape (1, C, H, W)
        pixel_values = inputs.pixel_values.squeeze(0)
        label_arr = torch.from_numpy(np.array(mask, dtype=np.int64))
        # leave background/ignore as 255
        return {"pixel_values": pixel_values, "labels": label_arr}


class RandomAug:
    def __init__(self, prob_flip=0.5, prob_rotate=0.3):
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate

    def __call__(self, image, mask):
        # random horizontal flip
        if random.random() < self.prob_flip:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        # random 90-degree rotation
        if random.random() < self.prob_rotate:
            k = random.choice([1, 2, 3])
            image = image.rotate(90 * k, expand=False)
            mask = mask.rotate(90 * k, expand=False)
        return image, mask


def dice_loss(pred, target, ignore_index=255, eps=1e-6):
    # pred: B x C x H x W (logits)
    # target: B x H x W
    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    loss = 0.0
    total = 0
    for c in range(num_classes):
        # mask out ignore
        t_c = (target == c).float()
        mask = (target != ignore_index).float()
        p_c = pred_soft[:, c, :, :] * mask
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        if union.item() == 0:
            continue
        loss += 1.0 - (2.0 * inter + eps) / (union + eps)
        total += 1
    if total == 0:
        return torch.tensor(0.0, device=pred.device)
    return loss / total


def focal_loss(logits, target, ignore_index=255, gamma=2.0, alpha=0.25):
    valid_mask = target != ignore_index
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=logits.device)

    target_clamped = target.clone()
    target_clamped[~valid_mask] = 0
    ce = F.cross_entropy(logits, target_clamped, reduction="none")
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce
    focal = focal * valid_mask.float()
    denom = valid_mask.float().sum().clamp_min(1.0)
    return focal.sum() / denom


class CustomTrainer(Trainer):
    def __init__(self, dice_weight=1.0, ce_weight=1.0, focal_weight=1.0, loss_mode="dice_ce", focal_gamma=2.0, focal_alpha=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.loss_mode = loss_mode
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # resize logits to labels spatial size
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        # logits: B x C x H x W, labels: B x H x W
        d = dice_loss(logits, labels.to(logits.device), ignore_index=255)
        if self.loss_mode == "dice_focal":
            fl = focal_loss(logits, labels.to(logits.device), ignore_index=255, gamma=self.focal_gamma, alpha=self.focal_alpha)
            loss = self.dice_weight * d + self.focal_weight * fl
        else:
            ce = F.cross_entropy(logits, labels.to(logits.device), ignore_index=255)
            loss = self.ce_weight * ce + self.dice_weight * d
        if return_outputs:
            return loss, outputs
        return loss


def build_model(num_labels=3, use_pretrained=False):
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    if use_pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            PRETRAINED_SEGFORMER_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    config = SegformerConfig(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_sizes=[32, 64, 160, 256],
        hidden_dropout_prob=0.1,
    )
    return SegformerForSemanticSegmentation(config)


def build_processor(use_pretrained=False):
    if use_pretrained:
        processor = SegformerImageProcessor.from_pretrained(
            PRETRAINED_SEGFORMER_NAME,
            do_reduce_labels=False,
        )
    else:
        processor = SegformerImageProcessor(size={"height": 128, "width": 128})
    processor.size = {"height": 128, "width": 128}
    return processor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    processor = SegformerImageProcessor(size={"height": 128, "width": 128})
    aug = RandomAug()
    # small datasets for smoke test
    if args.quick:
        train_ds = SimpleSegDataset(size=128, n=40, processor=processor, transforms=aug, seed=42)
        val_ds = SimpleSegDataset(size=128, n=10, processor=processor, transforms=None, seed=999)
        epochs = 1
    else:
        train_ds = SimpleSegDataset(size=128, n=200, processor=processor, transforms=aug, seed=42)
        val_ds = SimpleSegDataset(size=128, n=50, processor=processor, transforms=None, seed=999)
        epochs = 3

    model = build_model(num_labels=3)
    model.to(device)

    # optional: freeze encoder/backbone except heads
    if getattr(args, "freeze_encoder", False):
        for name, param in model.named_parameters():
            # keep head/classifier params trainable, freeze others
            if any(k in name for k in ("classifier", "decode", "head", "norm")):
                param.requires_grad = True
            else:
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./outputs/seg_improve",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        learning_rate=5e-4,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dice_weight=1.0,
        ce_weight=1.0,
    )

    print("Starting training (quick=%s)..." % args.quick)
    trainer.train()
    print("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder/backbone parameters and train heads only")
    args = parser.parse_args()
    main(args)
