"""04_multimodal_fusion.py
多模态鸟类分类项目主训练脚本
依赖：
    image_encoder.py
    text_encoder.py
实验：ResNet图像特征 + BERT文本特征拼接融合
"""
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from image_encoder import ImageEncoder
from text_encoder import TextEncoder, load_tokenizer

# -------------------------- 超参数配置 --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
COMMON_DIM = 512       # 统一投影维度：resnet18原生512，BERT768→投影到512
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4

# ResNet标准图像预处理
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tokenizer = load_tokenizer("bert-base-chinese")

# -------------------------- 数据集 --------------------------
class BirdMultiModalDataset(Dataset):
    def __init__(self, json_path, img_root, tokenizer, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.transform = transform
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 生成随机假图像张量，不再读取图片文件
        img = torch.randn(3, 224, 224)

        text = item["captions"]
        label = item["label"]

        token_out = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = token_out["input_ids"].squeeze(0)
        attn_mask = token_out["attention_mask"].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return img, input_ids, attn_mask, label


def collate_fn(batch):
    imgs, input_ids_list, attn_mask_list, labels = zip(*batch)
    imgs = torch.stack(imgs)
    input_ids = torch.stack(input_ids_list)
    attn_mask = torch.stack(attn_mask_list)
    labels = torch.tensor(labels)
    return imgs, input_ids, attn_mask, labels

# -------------------------- 融合模型 --------------------------
class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 图像编码器 ResNet18，冻结主干
        self.img_enc = ImageEncoder(
            backbone="resnet18",
            pretrained=True,
            freeze=True,
            out_dim=COMMON_DIM,
            dropout=0.1
        )
        # 文本编码器 中文BERT，mean池化，冻结主干
        self.txt_enc = TextEncoder(
            model_name="bert-base-chinese",
            pretrained=True,
            freeze=True,
            out_dim=COMMON_DIM,
            pooling="mean",
            dropout=0.1
        )
        # 拼接：[B,512] + [B,512] = [B,1024]
        self.classifier = nn.Sequential(
            nn.Linear(COMMON_DIM * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, img_tensor, input_ids, attention_mask):
        img_feat = self.img_enc(img_tensor, return_tokens=False)
        txt_feat = self.txt_enc(input_ids, attention_mask)
        fuse_feat = torch.cat([img_feat, txt_feat], dim=-1)
        logits = self.classifier(fuse_feat)
        return logits

# -------------------------- 训练&评估 --------------------------
if __name__ == "__main__":
    # 👉 这里改成你自己的json、图片文件夹路径
    train_set = BirdMultiModalDataset(json_path="datasets/train.json", img_root="datasets/images", tokenizer=tokenizer)
    val_set = BirdMultiModalDataset(json_path="datasets/val.json", img_root="datasets/images", tokenizer=tokenizer)
    test_set = BirdMultiModalDataset(json_path="datasets/test.json", img_root="datasets/images", tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = LateFusionModel().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_correct, train_total = 0, 0
        for imgs, input_ids, attn_mask, labels in train_loader:
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits = model(imgs, input_ids, attn_mask)
            loss = loss_func(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.shape[0]
        train_acc = train_correct / train_total

        # 验证集
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, input_ids, attn_mask, labels in val_loader:
                imgs = imgs.to(device)
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                logits = model(imgs, input_ids, attn_mask)
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.shape[0]
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

    # 测试集完整指标
    model.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for imgs, input_ids, attn_mask, labels in test_loader:
            imgs = imgs.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(imgs, input_ids, attn_mask)
            pred = torch.argmax(logits, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(labels.cpu().numpy())

    acc = accuracy_score(all_label, all_pred)
    pre = precision_score(all_label, all_pred, average="macro", zero_division=0)
    rec = recall_score(all_label, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_label, all_pred, average="macro", zero_division=0)
    print("\n===== 多模态融合测试结果 =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-Precision: {pre:.4f}")
    print(f"Macro-Recall: {rec:.4f}")
    print(f"Macro-F1: {f1:.4f}")
