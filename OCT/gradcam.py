import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ===== Load trained model =====
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model.load_state_dict(torch.load("best_resnet.pth", map_location=device))
model = model.to(device)
model.eval()

# ===== Grad-CAM Hook =====
gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

for name, module in model.named_modules():
    if name == "layer4":  # last conv block
        target_layer = module

def hook_layer(module, inp, out):
    activations.append(out)
    out.register_hook(save_gradient)

target_layer.register_forward_hook(hook_layer)

# ===== Preprocessing =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

classes = ["CNV", "DME", "DRUSEN", "NORMAL"]

# ===== Grad-CAM Function =====
def generate_gradcam(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # forward
    gradients.clear()
    activations.clear()
    out = model(x)
    pred = out.argmax(dim=1).item()

    # backward: focus on predicted class
    model.zero_grad()
    one_hot = torch.zeros((1, out.shape[-1]), device=device)
    one_hot[0, pred] = 1
    out.backward(gradient=one_hot)

    # CAM computation
    grads = gradients[0].mean(dim=[2,3], keepdim=True)
    cams = (grads * activations[0]).sum(dim=1).squeeze().cpu().detach().numpy()
    cams = np.maximum(cams, 0)
    cams = cams / cams.max() if cams.max() > 0 else cams

    # Read and resize image
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224,224))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    
    # Create heatmap and resize to match image dimensions
    heatmap = cv2.applyColorMap(np.uint8(255 * cams), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224, 224))  # Ensure same size
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # FIX: Convert heatmap to float and ensure same data type
    superimposed = cv2.addWeighted(
        img_cv.astype(np.float32), 0.6, 
        heatmap.astype(np.float32), 0.4, 
        0
    ).astype(np.uint8)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Heatmap"); plt.imshow(heatmap); plt.axis('off')
    plt.subplot(1,3,3); plt.title(f"Overlay — {classes[pred]}"); plt.imshow(superimposed); plt.axis('off')
    plt.tight_layout()
    plt.show()

# ===== Try Example =====
if __name__ == "__main__":
    test_img = r"C:\Users\dell\Downloads\OCT2017\OCT2017\test\Normal\NORMAL-2416187-1.jpeg"  # <-- change here
    generate_gradcam(test_img)
