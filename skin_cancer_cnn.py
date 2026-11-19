
import os
import glob
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Extract zip dataset if not extracted
zip_path = '/content/archive (1).zip'
extract_dir = '/content/extracted_data'
if os.path.exists(zip_path) and not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# 2. Gather image paths and classes
base_dir = '/content/extracted_data/train_cancer'
classes = ['benign', 'malignant']
image_data = []

for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    if os.path.exists(cls_dir):
        files = glob.glob(os.path.join(cls_dir, '*.*'))
        for f in files:
            try:
                with Image.open(f) as img:
                    width, height = img.size
                image_data.append({
                    'path': f,
                    'class': cls
                })
            except:
                pass

df = pd.DataFrame(image_data)

# 3. Train-Val-Test Stratified Split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['class'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['class'])

# 4. Dataset class & Transforms
class CancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_map = {'benign': 0, 'malignant': 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'path']
        label = self.dataframe.loc[idx, 'class']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.class_map[label]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CancerDataset(train_df, transform=train_transform)
test_dataset = CancerDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

# Calculate Class Weights to handle imbalance
total_train = len(train_df)
num_benign = len(train_df[train_df['class'] == 'benign'])
num_malignant = len(train_df[train_df['class'] == 'malignant'])
class_weights = torch.tensor([total_train / (2.0 * num_benign), total_train / (2.0 * num_malignant)], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. Model Training (3 Epochs)
print("Training CNN model...")
model.train()
for epoch in range(3):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/3 - Loss: {running_loss / len(train_dataset):.4f}")

# 7. Model Evaluation
print("Evaluating CNN model on testing set...")
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.numpy())

print(f"Test Accuracy: {accuracy_score(test_labels, test_preds) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=['benign', 'malignant'], zero_division=0))
