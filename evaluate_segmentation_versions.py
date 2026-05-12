"""Terminal-side evaluator for segmentation notebook versions.

This script trains and evaluates two configurations on the shared synthetic
segmentation task and prints a side-by-side metric comparison.

Example:
  /home/zhouyang/miniconda3/envs/med_phys/bin/python medical_physics/evaluate_segmentation_versions.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_physics.segmentation_improvements import (
    CustomTrainer,
    RandomAug,
    SimpleSegDataset,
    SegformerImageProcessor,
    build_model,
)
from transformers import TrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    if predictions.shape[-2:] != labels.shape[-2:]:
        prediction_tensor = torch.from_numpy(predictions[:, None]).float()
        prediction_tensor = torch.nn.functional.interpolate(
            prediction_tensor,
            size=labels.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        predictions = prediction_tensor.numpy()

    valid_mask = labels != 255
    pixel_accuracy = float((predictions[valid_mask] == labels[valid_mask]).mean()) if np.any(valid_mask) else 0.0

    ious = []
    num_labels = 3
    for class_id in range(num_labels):
        prediction_mask = predictions == class_id
        label_mask = labels == class_id
        union = np.logical_or(prediction_mask, label_mask)
        union = np.logical_and(union, valid_mask)
        union_sum = union.sum()
        if union_sum > 0:
            intersection = np.logical_and(prediction_mask, label_mask)
            ious.append(float(intersection.sum() / union_sum))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou}


@dataclass
class VersionConfig:
    name: str
    freeze_encoder: bool = False
    learning_rate: float = 6e-5


def build_datasets(quick: bool):
    processor = SegformerImageProcessor(do_reduce_labels=False)
    processor.size = {"height": 512, "width": 512}

    if quick:
        train_size = 20
        eval_size = 4
    else:
        train_size = 40
        eval_size = 8

    train_dataset = SimpleSegDataset(size=512, n=train_size, processor=processor, transforms=RandomAug(), seed=42)
    eval_dataset = SimpleSegDataset(size=512, n=eval_size, processor=processor, transforms=None, seed=999)
    return processor, train_dataset, eval_dataset


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_model_except_head(model):
    for name, param in model.named_parameters():
        if any(key in name for key in ("classifier", "decode", "head", "norm")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def train_and_evaluate(version: VersionConfig, quick: bool, seed: int):
    set_global_seed(seed)
    _, train_dataset, eval_dataset = build_datasets(quick)
    model = build_model(num_labels=3)

    # If this is the 'new' improved version, run a two-stage fine-tuning:
    # 1) freeze encoder and train short
    # 2) unfreeze and continue training with (optionally) lower LR / weight decay
    if version.name == "new":
        # Stage 1: freeze encoder
        freeze_model_except_head(model)
        stage1_args = TrainingArguments(
            output_dir=os.path.join("./outputs", f"segmentation_{version.name}_stage1"),
            learning_rate=version.learning_rate,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer = CustomTrainer(
            model=model,
            args=stage1_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        stage1_res = trainer.train()

        # Stage 2: unfreeze all and continue with weight decay and slightly lower LR
        for p in model.parameters():
            p.requires_grad = True
        stage2_args = TrainingArguments(
            output_dir=os.path.join("./outputs", f"segmentation_{version.name}_stage2"),
            learning_rate=version.learning_rate * 0.5,
            weight_decay=0.01,
            num_train_epochs=1 if quick else 2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer2 = CustomTrainer(
            model=model,
            args=stage2_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        stage2_res = trainer2.train()

        # Evaluate using the final model
        predictions = trainer2.predict(eval_dataset)
        metrics = compute_metrics((predictions.predictions, predictions.label_ids))
        # Aggregate training loss as last stage's reported training_loss
        training_loss = float(stage2_res.training_loss)
        train_runtime = float(stage1_res.metrics.get("train_runtime", 0.0)) + float(stage2_res.metrics.get("train_runtime", 0.0))
        return {
            "version": version.name,
            "freeze_encoder": True,
            "training_loss": training_loss,
            "train_runtime": train_runtime,
            "pixel_accuracy": metrics["pixel_accuracy"],
            "mean_iou": metrics["mean_iou"],
        }

    # Default (baseline) single-stage training
    if version.freeze_encoder:
        freeze_model_except_head(model)

    training_args = TrainingArguments(
        output_dir=os.path.join("./outputs", f"segmentation_{version.name}"),
        learning_rate=version.learning_rate,
        num_train_epochs=1 if quick else 3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    predictions = trainer.predict(eval_dataset)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    return {
        "version": version.name,
        "freeze_encoder": version.freeze_encoder,
        "training_loss": float(train_result.training_loss),
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou": metrics["mean_iou"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a very small dataset for a fast smoke test")
    parser.add_argument("--last-freeze-encoder", action="store_true", help="Freeze encoder for the last version")
    parser.add_argument("--new-freeze-encoder", action="store_true", help="Freeze encoder for the new version")
    parser.add_argument("--last-lr", type=float, default=6e-5)
    parser.add_argument("--new-lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=42, help="Global seed for both versions")
    args = parser.parse_args()

    set_global_seed(args.seed)

    versions = [
        VersionConfig(name="last", freeze_encoder=args.last_freeze_encoder, learning_rate=args.last_lr),
        VersionConfig(name="new", freeze_encoder=args.new_freeze_encoder, learning_rate=args.new_lr),
    ]

    results = [train_and_evaluate(version, quick=args.quick, seed=args.seed) for version in versions]

    print(json.dumps(results, ensure_ascii=False, indent=2))

    last, new = results
    print("\nComparison")
    print(f"- training_loss: last={last['training_loss']:.4f}, new={new['training_loss']:.4f}")
    print(f"- pixel_accuracy: last={last['pixel_accuracy']:.4f}, new={new['pixel_accuracy']:.4f}")
    print(f"- mean_iou: last={last['mean_iou']:.4f}, new={new['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
