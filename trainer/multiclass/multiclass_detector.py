# multiclass_detector.py

import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import timm
import cv2
import numpy as np
from collections import defaultdict

# Define the model class (must match the training code)
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load label names from the training CSV
base_dir = os.path.dirname(os.path.dirname(__file__))  # Adjust to point to the 'trainer' directory
csv_file = os.path.join(base_dir, 'config_files', 'multiclass', '_classes.csv')
df = pd.read_csv(csv_file)
labels = df.columns[1:]  # Skip 'filename' column
num_classes = len(labels)

# Function to determine category based on label
def get_category(label):
    # Common colors
    colors = {'white', 'black', 'red', 'blue', 'green', 'grey', 'silver', 'yellow', 'orange', 'pink', 'purple', 'brown', 'beige', 'tan', 'bronze', 'green-yellow', 'gray'}
    # Common vehicle types
    vehicle_types = {'car', 'truck', 'suv', 'van', 'pick-up', 'bus', '4_wheel_truck', 'texi_business', 'texi_personal', 'texi', 'car_personal'}
    # Sides
    sides = {'rear_side', 'front_side', 'left_side', 'right_side', 'side'}

    if label.lower() in colors:
        return 'color'
    elif label.lower() in vehicle_types:
        return 'vehicle_type'
    elif label.lower() in sides or label.lower().endswith('_side'):
        return 'side'
    else:
        return 'mark'  # Assuming remaining are brands/models like toyota_prius

# Initialize model and load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiLabelModel(num_classes=num_classes).to(device)
model_path = os.path.join(base_dir, 'saved_models', 'multiclass', 'multilabel_vehicle_model_epch_70K.pth')
model.load_state_dict(torch.load(model_path))

# Inference function
def predict_image(image_pil, model, transform, device):
    model.eval()
    image = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs.cpu().numpy()[0]

def detect_vehicle_attributes(img: np.ndarray) -> dict:
    # Convert cv2 image (BGR) to PIL Image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Run prediction
    predictions = predict_image(image_pil, model, transform, device)
    
    # Collect predictions by category
    categories = defaultdict(list)
    for label, prob in zip(labels, predictions):
        if prob > 0.5:
            category = get_category(label)
            categories[category].append((label, prob))
    
    # For each category, select the label with the highest probability
    attributes = {}
    for category, items in categories.items():
        if items:
            items.sort(key=lambda x: x[1], reverse=True)
            top_label, top_prob = items[0]
            attributes[category] = top_label
            attributes[f"{category}_p"] = int(top_prob * 100)
    
    return attributes