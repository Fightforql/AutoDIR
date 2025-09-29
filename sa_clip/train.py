from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from .augment import build_multi_view_augmentations
from .data import ImageTextCsvDataset
from .model import SAClipModel, saclip_loss
from .utils import set_seed, save_checkpoint


def collate_multi_view(batch):
    # batch: list of (views, text)
    views_list = [torch.stack(images, dim=0) for images, _ in batch]  # [V, C, H, W]
    # Stack into [B, V, C, H, W]
    images = torch.stack(views_list, dim=0)
    texts = [t for _, t in batch]
    return images, texts


def build_loader(cfg, split: str):
    aug_cfg = cfg["data"]["augment"]
    transforms = build_multi_view_augmentations(
        image_size=cfg["data"]["image_size"],
        aug_views=int(aug_cfg.get("aug_views", 2)),
        color_jitter=float(aug_cfg.get("color_jitter", 0.3)),
        gray_prob=float(aug_cfg.get("gray_prob", 0.1)),
        blur_prob=float(aug_cfg.get("blur_prob", 0.2)),
        crop_scale=tuple(aug_cfg.get("crop_scale", [0.8, 1.0])),
    )
    csv_path = cfg["data"][f"{split}_csv"]
    ds = ImageTextCsvDataset(csv_path=csv_path, multi_view_transforms=transforms)
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_multi_view,
        drop_last=True,
    )
    return loader


def evaluate(model: SAClipModel, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for images, texts in loader:
            steps += 1
            if steps > max_batches:
                break
            images = images.to(device, non_blocking=True)
            tokenized = model.tokenize(texts).to(device)
            image_features, text_features, logit_scale = model(images, tokenized)
            loss = saclip_loss(image_features, text_features, logit_scale)
            total_loss += loss.item()
    model.train()
    return total_loss / max(steps, 1)


def train(cfg):
    set_seed(int(cfg.get("seed", 42)))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAClipModel(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        text_ctx_len=cfg["model"].get("text_ctx_len", 77),
        freeze_vision=cfg["model"].get("freeze_vision", False),
        freeze_text=cfg["model"].get("freeze_text", True),
        device=device,
    )
    model.to(device)

    train_loader = build_loader(cfg, split="train")
    val_loader = build_loader(cfg, split="val")

    total_steps = math.ceil(len(train_loader) * cfg["train"]["epochs"])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg["train"]["lr"], betas=tuple(cfg["train"]["betas"]), weight_decay=cfg["train"]["wd"]) 

    scaler = GradScaler(enabled=(cfg["train"]["precision"] == "amp"))

    global_step = 0
    best_val = float("inf")
    model.train()

    for epoch in range(cfg["train"]["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for images, texts in pbar:
            global_step += 1
            images = images.to(device, non_blocking=True)
            tokenized = model.tokenize(texts).to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(cfg["train"]["precision"] == "amp")):
                image_features, text_features, logit_scale = model(images, tokenized)
                loss = saclip_loss(image_features, text_features, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if cfg["eval"]["every_n_steps"] and global_step % cfg["eval"]["every_n_steps"] == 0:
                val_loss = evaluate(model, val_loader, device, max_batches=cfg["eval"]["num_batches"])
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint({
                        "model": model.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }, cfg["output_dir"], global_step)

            if cfg["train"].get("save_interval", 0) and global_step % cfg["train"]["save_interval"] == 0:
                save_checkpoint({"model": model.state_dict(), "step": global_step}, cfg["output_dir"], global_step)

    # Final save
    save_checkpoint({"model": model.state_dict(), "step": global_step}, cfg["output_dir"], global_step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./sa_clip/configs/train.yaml",type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()