"""02_singleModal_ImageOnly.py
单模态图像基线：ResNet18 图像分类
依赖：image_encoder.py
"""
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from 03_image_encoder import ImageEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 30
COMMON_DIM = 512
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-4

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BirdImgDataset(Dataset):
    def __init__(self, json_path, img_root):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)["annotations"]
        self.img_root = img_root
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.img_root, item["image_name"])).convert("RGB")
        img = img_transform(img)
        label = int(item["label"])
        return img, label

class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ImageEncoder(
            backbone="resnet18",
            pretrained=True,
            freeze=True,
            out_dim=COMMON_DIM,
            dropout=0.1
        )
        self.head = nn.Linear(COMMON_DIM, NUM_CLASSES)
    def forward(self, x):
        feat = self.enc(x, return_tokens=False)
        return self.head(feat)

if __name__ == "__main__":
    train_ds = BirdImgDataset("datasets/train.json", "datasets/images")
    val_ds   = BirdImgDataset("datasets/val.json",   "datasets/images")
    test_ds  = BirdImgDataset("datasets/test.json",  "datasets/images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = ImageOnlyModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        model.train()
        loss_sum, corr, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            pred = logits.argmax(1)
            corr += (pred == labels).sum().item()
            total += labels.shape[0]
        train_acc = corr / total

        model.eval()
        cv, tv = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs).argmax(1)
                cv += (pred == labels).sum().item()
                tv += labels.shape[0]
        val_acc = cv / tv
        print(f"Ep{ep+1} | Loss:{loss_sum/len(train_loader):.4f} | TrainAcc:{train_acc:.4f} | ValAcc:{val_acc:.4f}")

    all_pred, all_label = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            pred = logits.argmax(1)
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(labels.cpu().numpy())
    acc = accuracy_score(all_label, all_pred)
    pre = precision_score(all_label, all_pred, average="macro", zero_division=0)
    rec = recall_score(all_label, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_label, all_pred, average="macro", zero_division=0)
    print("\n==== 仅图像基线结果 ====")
    print(f"Acc:{acc:.4f}, P:{pre:.4f}, R:{rec:.4f}, F1:{f1:.4f}")
