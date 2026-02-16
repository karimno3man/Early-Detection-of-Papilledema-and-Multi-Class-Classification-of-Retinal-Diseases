import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import numpy as np

def main():

    # ================================
    # 1) Paths & Hyperparameters
    # ================================
    data_dir = r"C:\Users\dell\Downloads\OCT2017\OCT2017"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    batch_size = 32
    epochs = 20
    lr = 1e-4
    input_size = 224   # ✅ ResNet uses 224x224 images
    num_classes = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ================================
    # 2) Data Transforms
    # ================================
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # ================================
    # 3) Datasets & Loaders
    # ================================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✅ Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}")

    # ================================
    # 4) Load ResNet-18
    # ================================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers first (fine-tune classifier only)
    for param in model.parameters():
        param.requires_grad = False

    # Replace FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # ================================
    # 5) Train Loop
    # ================================
    best_val_acc = 0
    train_acc_hist = []; val_acc_hist = []

    for epoch in range(1, epochs+1):
        model.train()
        y_true_train=[]; y_pred_train=[]
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*imgs.size(0)
            _, preds = torch.max(outputs,1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        train_acc = (np.array(y_true_train)==np.array(y_pred_train)).mean()*100
        train_acc_hist.append(train_acc)

        # ----- Validation -----
        model.eval()
        y_true_val=[]; y_pred_val=[]
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs,1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_acc = (np.array(y_true_val)==np.array(y_pred_val)).mean()*100
        val_acc_hist.append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "resnet18_best.pth")

    # ================================
    # 6) Test Evaluation
    # ================================
    print("\n✅ Loading best model...")
    model.load_state_dict(torch.load("resnet18_best.pth"))
    model.eval()

    y_true_test=[]; y_pred_test=[]
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs,1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    print("\n=== Test Report ===")
    print(classification_report(y_true_test, y_pred_test, target_names=train_dataset.classes))

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("ResNet-18 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show(block=True)


# Windows guard
if __name__ == "__main__":
    main()
