import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from model import Model
from utils import CTCLabelConverter
import yaml
import argparse
from easyocr import Reader
import sys
import numpy as np

# Define get_config function
def get_config(config_path):
    with open(config_path, 'r', encoding="utf8") as stream:
        config_dict = yaml.safe_load(stream)
    opt = argparse.Namespace(**config_dict)
    return opt

# Accept image path from command line
if len(sys.argv) < 2:
    print("Please provide an image path: python3 test_plate.py <image_path>")
    sys.exit(1)
image_path = sys.argv[1]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
config_path = "config_files/mn_filtered_config_v5.yaml"
opt = get_config(config_path)

# Ensure opt.character is set
if not hasattr(opt, 'character'):
    opt.character = opt.number + opt.symbol + opt.lang_char
opt.character = ''.join(sorted(set(opt.character)))
opt.num_class = len(opt.character) + 1
print("Character set:", opt.character)

# Load recognition model
model = Model(opt).to(device)
state_dict = torch.load('best_accuracy.pth', map_location=device)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((opt.imgH, opt.imgW)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize EasyOCR Reader
reader = Reader(['ru', 'mn'], gpu=torch.cuda.is_available())

# Load the image
image = Image.open(image_path)
image_width, image_height = image.size
print(f"Input image size: {image_width}x{image_height}")

# Detect text regions
detection_results = reader.detect(image_path, min_size=15, text_threshold=0.7, low_text=0.3)
print("Detection results:", detection_results)

# Initialize CTC label converter
converter = CTCLabelConverter(opt.character)

# Visualize bounding boxes on the original image
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Process each detected region
if detection_results[0]:  # Check if any regions were detected
    for bbox in detection_results[0]:
        # Extract axis-aligned bounding box
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        padding = 50  # Increased padding
        # Compute AABB with corrected min/max
        x1 = max(min(x_coords) - padding, 0)
        x2 = min(max(x_coords) + padding, image_width)
        y1 = max(min(y_coords) - padding, 0)
        y2 = min(max(y_coords) + padding, image_height)
        
        # Swap y1 and y2 if y1 > y2 to ensure valid box
        if y1 > y2:
            y1, y2 = y2, y1
            print(f"Swapped y1 and y2: y1={y1}, y2={y2}")
        
        # Validate bounding box
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bounding box: x1={x1}, x2={x2}, y1={y1}, y2={y2}, skipping...")
            continue
        
        print(f"Cropping region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Draw bounding box on the image
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        
        # Crop the image
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Save the cropped image for debugging
        cropped_filename = f"cropped_{image_path.split('/')[-1]}"
        cropped_image.save(cropped_filename)
        print(f"Saved cropped image: {cropped_filename}")
        
        # Preprocess the cropped image
        cropped_image = cropped_image.convert('L')
        cropped_image = transform(cropped_image).unsqueeze(0).to(device)
        print(f"Cropped image shape: {cropped_image.shape}")

        dummy_text = torch.zeros(1, opt.batch_max_length, dtype=torch.long).to(device)
        
        # Make prediction
        with torch.no_grad():
            preds = model(cropped_image, dummy_text, is_train=False)
            print("Preds shape:", preds.shape)
            print("Raw predictions:", preds.max(dim=2)[1])
            
            preds = preds.log_softmax(2).permute(1, 0, 2)
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(0, 1)
            
            text_index = preds_index.view(-1)
            length = [preds_index.size(1)]
            
            preds_str = converter.decode_greedy(text_index, length)
            print(f"Predicted License Plate: {preds_str[0]}")
else:
    print("No text regions detected, using entire image as fallback...")
    # Fallback: Use the entire image if detection fails
    cropped_image = image
    cropped_image.save(f"cropped_{image_path.split('/')[-1]}")
    cropped_image = cropped_image.convert('L')
    cropped_image = transform(cropped_image).unsqueeze(0).to(device)
    print(f"Cropped image shape: {cropped_image.shape}")

    dummy_text = torch.zeros(1, opt.batch_max_length, dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        preds = model(cropped_image, dummy_text, is_train=False)
        print("Preds shape:", preds.shape)
        print("Raw predictions:", preds.max(dim=2)[1])
        
        preds = preds.log_softmax(2).permute(1, 0, 2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(0, 1)
        
        text_index = preds_index.view(-1)
        length = [preds_index.size(1)]
        
        preds_str = converter.decode_greedy(text_index, length)
        print(f"Predicted License Plate: {preds_str[0]}")

# Save the image with bounding boxes
bbox_image_filename = f"bbox_{image_path.split('/')[-1]}"
image_with_boxes.save(bbox_image_filename)
print(f"Saved image with bounding boxes: {bbox_image_filename}")