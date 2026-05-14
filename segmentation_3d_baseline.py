"""3D segmentation baseline (synthetic volumes).

This script provides a minimal 3D training/evaluation loop so the project can
move from 2D experiments to 3D experiments with a reproducible baseline.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sphere_volume(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # one-channel image + one-channel binary label
    image = np.zeros((1, size, size, size), dtype=np.float32)
    label = np.zeros((size, size, size), dtype=np.int64)

    # random sphere center/radius
    radius = rng.integers(max(3, size // 10), max(4, size // 5))
    cx = rng.integers(radius, size - radius)
    cy = rng.integers(radius, size - radius)
    cz = rng.integers(radius, size - radius)

    zz, yy, xx = np.ogrid[:size, :size, :size]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    mask = dist2 <= radius**2

    label[mask] = 1
    image[0] = rng.normal(0.0, 0.08, size=(size, size, size)).astype(np.float32)
    image[0][mask] += 0.9
    image[0] = np.clip(image[0], 0.0, 1.0)

    return image, label


class Synthetic3DDataset(Dataset):
    def __init__(self, n: int, size: int, seed: int) -> None:
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)
        image, label = make_sphere_volume(self.size, rng)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }


class TinyUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, base: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv3d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        u = self.up(e2)
        x = torch.cat([u, e1], dim=1)
        x = self.dec(x)
        return self.head(x)


def dice_fg(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> float:
    pred = torch.argmax(logits, dim=1)
    pred_fg = (pred == 1).float()
    label_fg = (labels == 1).float()
    inter = (pred_fg * label_fg).sum().item()
    union = pred_fg.sum().item() + label_fg.sum().item()
    return float((2.0 * inter + eps) / (union + eps))


def dice_loss_fg(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = (labels == 1).float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_acc = 0.0
    total_dice = 0.0
    total = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = model(image)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).float().mean().item()
        total_acc += acc
        total_dice += dice_fg(logits, label)
        total += 1
    return {
        "voxel_accuracy": total_acc / max(total, 1),
        "dice_fg": total_dice / max(total, 1),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        train_n, val_n, epochs = 24, 8, 4
    else:
        train_n, val_n, epochs = 64, 16, args.epochs

    train_ds = Synthetic3DDataset(train_n, size=args.size, seed=args.seed)
    val_ds = Synthetic3DDataset(val_n, size=args.size, seed=args.seed + 10_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyUNet3D(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce_weight = torch.tensor([1.0, args.fg_weight], dtype=torch.float32, device=device)

    print(f"Device: {device}")
    print(f"Train samples: {train_n}, Val samples: {val_n}, Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = model(image)
            ce = F.cross_entropy(logits, label, weight=ce_weight)
            d = dice_loss_fg(logits, label)
            loss = ce + args.dice_weight * d
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_voxel_acc={metrics['voxel_accuracy']:.4f} val_dice_fg={metrics['dice_fg']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg-weight", type=float, default=8.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
