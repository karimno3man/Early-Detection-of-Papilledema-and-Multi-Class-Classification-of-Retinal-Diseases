import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ==============================
# 1️⃣  Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

root = r"C:\Users\dell\Downloads\OCT2017\OCT2017"
train_dir = os.path.join(root, "train")
val_dir   = os.path.join(root, "val")
test_dir  = os.path.join(root, "test")

# ==============================
# 2️⃣  Data Augmentation & Normalization
# ==============================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 3️⃣  Datasets & Dataloaders
# ==============================
train_set = datasets.ImageFolder(train_dir, transform=train_tf)
val_set   = datasets.ImageFolder(val_dir, transform=test_tf)
test_set  = datasets.ImageFolder(test_dir, transform=test_tf)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

print(f"\n✅ Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

# ==============================
# 4️⃣  Class Distribution (Detailed)
# ==============================
train_counts = Counter([y for _, y in train_set.samples])
val_counts   = Counter([y for _, y in val_set.samples])
test_counts  = Counter([y for _, y in test_set.samples])

print("\n📊 Dataset Distribution:")
for i, cls in enumerate(train_set.classes):
    print(f"  {cls:<10} | Train: {train_counts[i]:5d} | Val: {val_counts[i]:5d} | Test: {test_counts[i]:5d}")

# ==============================
# 5️⃣  Model: ViT-B16 (Vision Transformer)
# ==============================
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Freeze feature extractor first
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.heads.head = nn.Linear(model.heads.head.in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.heads.parameters(), lr=1e-4)

# ==============================
# 6️⃣  Training
# ==============================
best_val = 0
train_acc_hist, val_acc_hist = [], []

epochs = 15
warmup_epochs = 5

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    if epoch == warmup_epochs:
     print("🔓 Unfreezing last few transformer blocks...")
    for name, param in model.named_parameters():
        if "encoder.layers.encoder_layer_10" in name or "encoder.layers.encoder_layer_11" in name:
            param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)


    # Save best model
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "vit_b16_best.pth")
        print("✅ Model saved!")

# ==============================
# 7️⃣  Evaluation on Test Set
# ==============================
print("\n=== Testing best model ===")
model.load_state_dict(torch.load("vit_b16_best.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print("\n=== Classification Report (Test) ===")
print(classification_report(y_true, y_pred, target_names=train_set.classes))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_set.classes,
            yticklabels=train_set.classes)
plt.title("ViT-B16 Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show(block=True)
