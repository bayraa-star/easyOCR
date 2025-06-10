import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import Model
from utils import CTCLabelConverter, AttrDict
import yaml
import os
import re
import time

start_time = time.time()

num_threads = 2
torch.set_num_threads(num_threads)
cv2.setNumThreads(num_threads)

# Define frame size for consistency
frame_width, frame_height = 1280, 720  # Adjust as needed


def load_config(file_path):
    """Load the configuration from a YAML file and prepare the character set."""
    with open(file_path, "r", encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    return opt


def resize_with_pad(img, target_height, target_width):
    """Resize the image while maintaining aspect ratio and pad to target size."""
    w, h = img.size
    ratio = target_height / h
    new_w = int(w * ratio)
    if new_w > target_width:
        ratio = target_width / w
        new_h = int(h * ratio)
        img = img.resize((target_width, new_h), Image.BICUBIC)
        padding = (0, 0, 0, target_height - new_h)  # Pad bottom
        img = ImageOps.expand(img, padding, fill=0)  # Black padding
    else:
        img = img.resize((new_w, target_height), Image.BICUBIC)
        padding = (0, 0, target_width - new_w, 0)  # Pad right
        img = ImageOps.expand(img, padding, fill=0)  # Black padding
    return img


class LicensePlateRecognizer:
    def __init__(self, model_path, config_path):
        """Initialize the recognizer with model and config paths."""
        self.opt = load_config(config_path)
        if "CTC" in self.opt.Prediction:
            self.opt.num_class = len(self.opt.character) + 1  # +1 for CTC blank
        else:
            self.opt.num_class = len(self.opt.character)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.opt).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.converter = CTCLabelConverter(self.opt.character)

    def predict(self, image_array):
        """Predict text from a NumPy image array."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(gray)  # Convert to PIL Image
        img = resize_with_pad(img, self.opt.imgH, self.opt.imgW)
        img = transforms.ToTensor()(img)  # To tensor [0, 1]
        img = transforms.Normalize(mean=[0.5], std=[0.5])(img)  # Normalize [-1, 1]
        img = img.unsqueeze(0).to(self.device)  # Add batch dimension

        dummy_text = torch.zeros(1, self.opt.batch_max_length, dtype=torch.long).to(
            self.device
        )

        with torch.no_grad():
            preds = self.model(img, dummy_text).log_softmax(2)

        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        pred_str = self.converter.decode_greedy(preds_index[0].cpu(), preds_size.cpu())

        if "[blank]" in pred_str:
            pred_str = pred_str.replace("[blank]", "")
            cleaned = []
            for char in pred_str:
                if not cleaned or char != cleaned[-1]:
                    cleaned.append(char)
            pred_str = "".join(cleaned)

        return pred_str


# Load video
cap = cv2.VideoCapture("../videos/tollgate.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Resize the first frame
first_frame = cv2.resize(first_frame, (frame_width, frame_height))

# Select ROI interactively
print("Select the ROI on the first frame. Press Enter to confirm, Esc to cancel.")
roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)

# Check if a valid ROI was selected
if roi[2] == 0 or roi[3] == 0:
    print("No ROI selected. Exiting.")
    exit()

# Extract ROI coordinates
ROI_x1, ROI_y1, ROI_w, ROI_h = roi
ROI_x2 = ROI_x1 + ROI_w
ROI_y2 = ROI_y1 + ROI_h

# Close the selectROI window
cv2.destroyWindow("Select ROI")

# Initialize previous ROI for motion detection
prev_roi = first_frame[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
prev_gray_roi = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)

# Load Plate detection model
yolo_model = YOLO("../models/best_last_version.pt")

# Load OCR model
ocr_model = LicensePlateRecognizer(
    model_path="./saved_models/mn_filtered_v16-1200image/ocr.pth",
    config_path="./config_files/mn_filtered_config_v6m1000.yaml",
)

# Create directory to save cropped plates
save_dir = "detected_plates"
os.makedirs(save_dir, exist_ok=True)

# Frame counter for naming saved images
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Draw ROI on the frame for visualization
    cv2.rectangle(frame, (ROI_x1, ROI_y1), (ROI_x2, ROI_y2), (255, 0, 0), 2)

    # Extract current ROI and convert to grayscale
    roi = frame[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between current and previous ROI
    diff = cv2.absdiff(gray_roi, prev_gray_roi)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    non_zero = cv2.countNonZero(thresh)

    # Calculate motion threshold (e.g., 1% of ROI pixels)
    roi_pixels = (ROI_x2 - ROI_x1) * (ROI_y2 - ROI_y1)
    motion_threshold = 0.01 * roi_pixels

    if non_zero > motion_threshold:
        print(f"Motion detected in ROI at frame {frame_count}")
        results = yolo_model.predict(frame)
        detections = results[0].boxes.data.cpu().numpy()

        for det_index, det in enumerate(detections):
            x1, y1, x2, y2, confidence = (
                int(det[0]),
                int(det[1]),
                int(det[2]),
                int(det[3]),
                det[4],
            )
            if confidence < 0.5:
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if not (ROI_x1 <= center_x <= ROI_x2 and ROI_y1 <= center_y <= ROI_y2):
                continue

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            plate_img = frame[y1:y2, x1:x2]

            text = ocr_model.predict(plate_img)[0]
            text = text.strip() or "unknown"

            sanitized_text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
            filename = f"frame{frame_count}_plate_{det_index}_{sanitized_text}.jpg"
            filepath = os.path.join(save_dir, filename)

            cv2.imwrite(filepath, plate_img)
            print(f"Saved: {filepath}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    else:
        print(f"No motion in ROI at frame {frame_count}")

    prev_gray_roi = gray_roi.copy()

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
