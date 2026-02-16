import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import numpy as np
import os

# ==============================
# 1️⃣ Paths and Parameters
# ==============================
data_dir = r"C:\Users\dell\Downloads\OCT2017\OCT2017"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

batch_size = 32
epochs = 20
learning_rate = 0.001
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ==============================
# 2️⃣ Data Loading (NO AUGMENTATION)
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# Split training set into train + validation
val_size = int(0.1 * len(train_data))
train_size = len(train_data) - val_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"✅ Classes found: {train_data.classes}")
print(f"✅ Samples: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_data)}")

# ==============================
# 3️⃣ Define Basic CNN
# ==============================
class BasicCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = BasicCNN(num_classes=num_classes).to(device)

# ==============================
# 4️⃣ Training Setup
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stop_patience = 3
best_val_acc = 0
patience_counter = 0

train_acc_list, val_acc_list = [], []
train_prec_list, val_prec_list = [], []
train_rec_list, val_rec_list = [] , []

# ==============================
# 5️⃣ Training Loop (w/ Precision & Recall)
# ==============================
for epoch in range(epochs):
    # ---- Training ----
    model.train()
    y_true_train, y_pred_train = [], []
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds.cpu().numpy())

    train_acc = correct / total * 100
    train_prec = precision_score(y_true_train, y_pred_train, average="macro", zero_division=0)
    train_rec = recall_score(y_true_train, y_pred_train, average="macro", zero_division=0)

    # ---- Validation ----
    model.eval()
    y_true_val, y_pred_val = [], []
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(preds.cpu().numpy())

    val_acc = val_correct / val_total * 100
    val_prec = precision_score(y_true_val, y_pred_val, average="macro", zero_division=0)
    val_rec = recall_score(y_true_val, y_pred_val, average="macro", zero_division=0)

    # Save metrics
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_prec_list.append(train_prec)
    val_prec_list.append(val_prec)
    train_rec_list.append(train_rec)
    val_rec_list.append(val_rec)

    print(f"Epoch [{epoch+1}/{epochs}] ➤ "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
          f"Train Prec: {train_prec:.3f} | Val Prec: {val_prec:.3f} | "
          f"Train Rec: {train_rec:.3f} | Val Rec: {val_rec:.3f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("🛑 Early stopping triggered!")
            break

# ==============================
# 6️⃣ Plot Accuracy, Precision, Recall
# ==============================
plt.figure(figsize=(10,6))
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(val_acc_list, label="Validation Accuracy")
plt.plot(np.array(train_prec_list)*100, label="Train Precision")
plt.plot(np.array(val_prec_list)*100, label="Validation Precision")
plt.plot(np.array(train_rec_list)*100, label="Train Recall")
plt.plot(np.array(val_rec_list)*100, label="Validation Recall")
plt.xlabel("Epochs")
plt.ylabel("Score (%)")
plt.title("Training vs Validation Metrics (Accuracy, Precision, Recall)")
plt.legend()
plt.tight_layout()
plt.show(block=True)

# ==============================
# 7️⃣ Evaluation on Test Set
# ==============================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== Classification Report on Test Set ===")
print(classification_report(y_true, y_pred, target_names=train_data.classes))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.title("Confusion Matrix (No Augmentation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show(block=True)
print("✅ Evaluation complete.")