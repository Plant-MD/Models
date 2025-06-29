import os
import time
import sys
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# ==== CONFIG ====
DATA_DIR = r"C:\Users\suyog\tests\plant-detection-training-model-resnet34\plantvillage dataset\color"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.67
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("========== DEVICE INFO ==========")
print(f"[DEBUG] Using device: {DEVICE}")
print(f"[DEBUG] CUDA Available: {torch.cuda.is_available()}")
print(f"[DEBUG] CUDA Version: {torch.version.cuda}")
if DEVICE.type == 'cuda':
    print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
print("=================================")

# ==== TRANSFORMS ====
print("[DEBUG] Initializing image transforms...")
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== LOAD DATA ====
print("[DEBUG] Loading dataset...")
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"[DEBUG] Total images found: {len(dataset)}")
print(f"[DEBUG] Classes: {dataset.classes}")
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size
print(f"[DEBUG] Splitting dataset into {train_size} training and {val_size} validation samples...")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
print("[DEBUG] Data loaders initialized.")

# ==== MODEL ====
print("[DEBUG] Loading ResNet34 model...")
model = models.resnet34(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(DEVICE)
print(f"[DEBUG] Model moved to {DEVICE}")
print(model)

# ==== LOSS & OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("[DEBUG] Loss function and optimizer set.")

# ==== TRAINING LOOP ====
print("========== TRAINING START ==========")
start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n[DEBUG] Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss, correct = 0, 0
    total_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_correct = (outputs.argmax(1) == labels).sum().item()
        correct += batch_correct

        # Progress bar update
        progress = (batch_idx + 1) / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rEpoch {epoch+1} Progress: [{bar}] {batch_idx + 1}/{total_batches} batches')
        sys.stdout.flush()

    train_acc = correct / len(train_loader.dataset)
    print()  # move to next line after progress bar
    print(f"[DEBUG] Epoch {epoch+1} Completed - Total Loss: {total_loss:.4f}, Training Accuracy: {train_acc:.4f}")

print("========== TRAINING COMPLETE ==========")
print(f"[DEBUG] Total training time: {time.time() - start_time:.2f} seconds")

# ==== VALIDATION ====
print("========== VALIDATION START ==========")
model.eval()
correct = 0
with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        batch_correct = (outputs.argmax(1) == labels).sum().item()
        correct += batch_correct
        # Optional: add a validation progress bar if you want

val_acc = correct / len(val_loader.dataset)
print(f"[DEBUG] Validation Accuracy: {val_acc:.4f}")
print("========== VALIDATION COMPLETE ==========")

# ==== SAVE MODEL ====
save_path = "resnet34_plant_disease-5epoch.pth"
torch.save(model.state_dict(), save_path)
print(f"[DEBUG] Model saved as: {save_path}")
