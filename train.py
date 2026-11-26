import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from preprocess import preprocess_image
from model import MobileNetV3Iris, CosFaceHead

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class IrisDataset(Dataset):
    def __init__(self, csv_path, img_size, use_rubber_sheet):
        self.df = pd.read_csv(csv_path)
        self.img_size = tuple(img_size)
        self.use_rubber_sheet = use_rubber_sheet

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tensor = preprocess_image(row["filepath"], self.img_size, self.use_rubber_sheet)
        label = int(row["label"]) if "label" in row else 0
        return tensor, label

def train_one_seed(seed, cfg):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = IrisDataset(
        os.path.join(cfg["paths"]["protocols_dir"], "train.csv"),
        cfg["preprocess"]["img_size"],
        cfg["preprocess"]["use_rubber_sheet"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    num_classes = train_ds.df["label"].nunique()

    model = MobileNetV3Iris(
        backbone_name=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        embedding_dim=cfg["model"]["embedding_dim"],
        freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)
    
    head = CosFaceHead(
        embedding_dim=cfg["model"]["embedding_dim"],
        num_classes=num_classes,
        s=cfg["train"]["scale_s"],
        m=cfg["train"]["margin_m"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        head.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embs = model(imgs)
            logits = head(embs, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[Seed {seed}] Epoch {epoch+1}/{cfg['train']['epochs']} - Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    save_path = os.path.join(cfg["paths"]["results_dir"], f"mobilenetv3_seed{seed}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    for s in cfg["seeds_repetition"]:
        train_one_seed(s, cfg)