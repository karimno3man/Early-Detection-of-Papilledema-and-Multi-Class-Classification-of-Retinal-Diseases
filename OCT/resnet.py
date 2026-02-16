import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ====== PATHS ======
root = r"C:\Users\dell\Downloads\OCT2017\OCT2017"
train_dir = os.path.join(root, "train")
val_dir   = os.path.join(root, "val")
test_dir  = os.path.join(root, "test")

# ====== TRANSFORMS ======
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ====== DATASETS ======
train_set = datasets.ImageFolder(train_dir, transform=train_tf)
val_set   = datasets.ImageFolder(val_dir, transform=test_tf)
test_set  = datasets.ImageFolder(test_dir, transform=test_tf)

# ====== DATA LOADERS ======
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

# ====== CLASS WEIGHTS ======
class_counts = Counter([y for _,y in train_set.samples])
total = sum(class_counts.values())
class_weights = [total/class_counts[i] for i in range(len(class_counts))]
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class Weights:", class_weights)

# ====== MODEL ======
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

# ====== WARM-UP — freeze backbone first ======
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

best_val = 0

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

# ====== TRAINING ======
epochs = 20
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

        # Track training accuracy
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    val_acc = evaluate(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 🔥 Unfreeze full model after warm-up
    if epoch == warmup_epochs:
        print("🔓 Unfreezing entire model for full fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_resnet.pth")
        print("✅ Model saved!")

    scheduler.step(val_acc)


# ====== TEST ======
print("\n=== Testing best model ===")
model.load_state_dict(torch.load("best_resnet.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device)
        out = model(x)
        pred = out.argmax(1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())

print(classification_report(y_true, y_pred, target_names=train_set.classes))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
