import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import numpy as np
from collections import Counter

def main():
    # ---------------------------
    # 1) Paths & Hyperparameters
    # ---------------------------
    data_dir = r"C:\Users\dell\Downloads\OCT2017\OCT2017"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    batch_size = 32
    epochs = 20
    lr = 1e-3
    num_classes = 4
    early_stop_patience = 4
    input_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ---------------------------
    # 2) Transforms (with Augmentation)
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ---------------------------
    # 3) Dataset & Dataloader
    # ---------------------------
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✅ Classes found: {train_dataset.classes}")
    print(f"✅ Samples: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}")
    train_targets = [train_subset[i][1] for i in range(len(train_subset))]
    print("Train class counts (subset):", dict(Counter(train_targets)))

    # ---------------------------
    # 4) Define Custom CNN
    # ---------------------------
    class CustomCNN(nn.Module):
        def __init__(self, num_classes=4):
            super(CustomCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.BatchNorm2d(128),
                nn.Dropout(0.25)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * (input_size // 8) * (input_size // 8), 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = CustomCNN(num_classes=num_classes).to(device)
    print(model)

    # ---------------------------
    # 5) Loss, Optimizer
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler without verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # ---------------------------
    # 6) Training Loop
    # ---------------------------
    train_acc_hist, val_acc_hist = [], []
    train_prec_hist, val_prec_hist = [], []
    train_rec_hist, val_rec_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        y_true_train, y_pred_train = [], []
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        train_loss = running_loss / len(train_subset)
        train_acc = (np.array(y_true_train) == np.array(y_pred_train)).mean() * 100
        train_prec = precision_score(y_true_train, y_pred_train, average="macro", zero_division=0)
        train_rec = recall_score(y_true_train, y_pred_train, average="macro", zero_division=0)

        # Validation
        model.eval()
        y_true_val, y_pred_val = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_subset)
        val_acc = (np.array(y_true_val) == np.array(y_pred_val)).mean() * 100
        val_prec = precision_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        val_rec = recall_score(y_true_val, y_pred_val, average="macro", zero_division=0)

        scheduler.step(val_acc)

        # Record
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_prec_hist.append(train_prec)
        val_prec_hist.append(val_prec)
        train_rec_hist.append(train_rec)
        val_rec_hist.append(val_rec)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Prec: {train_prec:.3f} | Val Prec: {val_prec:.3f} | "
              f"Train Rec: {train_rec:.3f} | Val Rec: {val_rec:.3f}")

        # Early stopping
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("🛑 Early stopping triggered!")
                break

    # ---------------------------
    # 7) Final Evaluation + Confusion Matrix
    # ---------------------------
    print("\n=== Final evaluation on test set ===")
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    print(f"Number of test samples: {len(y_true_test)}")
    print("\n=== Classification Report (test) ===")
    print(classification_report(y_true_test, y_pred_test, target_names=train_dataset.classes))

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix (test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show(block=True)

# ---------------------------
# MAIN GUARD (fixes Windows multiprocessing)
# ---------------------------
if __name__ == "__main__":
    main()
