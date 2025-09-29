# app.py
# Updated app.py: Integrates YOLOv8 for license plate detection, OCR, Basic Authentication, returns base64 images, vehicle multi-class detection, and sends training results to Django
# Now returns per-character details from OCR prediction
# Added handling for empty uploads to prevent OpenCV errors
# Returns "TRAIN" if plate_number is empty

# Prerequisites:
# - Install Ultralytics for YOLOv8: pip install ultralytics
# - Install FastAPI security dependencies: pip install fastapi[all]
# - Install python-dotenv: pip install python-dotenv
# - Install requests: pip install requests
# - Run the app: uvicorn App:app --host 0.0.0.0 --port 8000 --reload

import os
import base64
import logging
import requests
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, BackgroundTasks, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from license_plate_recognizer import LicensePlateRecognizer
from ultralytics import YOLO
import numpy as np
import cv2
import io
from multiclass.multiclass_detector import detect_vehicle_attributes
from typing import Dict, Any
import pandas as pd
from utils import AttrDict
from train import train

app = FastAPI()
security = HTTPBasic()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded credentials for FastAPI
USERNAME = "admin"
PASSWORD = "Admin@123"

# Django endpoint and credentials
DJANGO_URL = 'http://localhost:8008/api/ocr-create-trained/'
DJANGO_AUTH = ('raspberrypi', 'Admin@zxcasdqwe')

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    logger.info(f"Received credentials: username={credentials.username}")
    logger.info(f"Expected credentials: username={USERNAME}")
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        logger.warning("Invalid credentials attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Initialize the recognizer and detector
MODEL_PATH = 'saved_models/mn_filtered_v24-85Kimage/best_accuracy.pth'  # OCR model
CONFIG_PATH = 'config_files/mn_filtered_config_v6m1000.yaml'  # OCR config
DETECTION_MODEL_PATH = 'saved_models/plate_detection/plate_detection.pt'

recognizer = LicensePlateRecognizer(MODEL_PATH, CONFIG_PATH)
detector = YOLO(DETECTION_MODEL_PATH)

def get_config_from_dict(config_dict: Dict[str, Any]):
    opt = AttrDict(config_dict)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt.select_data.split('-'):
            csv_path = os.path.join(opt.train_data, data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character = ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    opt.character = ''.join(sorted(set(opt.character)))
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

def parse_log_train(log_content: str) -> dict:
    lines = log_content.strip().split('\n')
    last_loss_line = None
    last_current_line = None
    last_best_line = None
    for line in lines:
        line = line.strip()
        if line.startswith('[') and 'Train loss:' in line:
            last_loss_line = line
        elif line.startswith('Current_accuracy'):
            last_current_line = line
        elif line.startswith('Best_accuracy'):
            last_best_line = line

    metrics = {}
    if last_loss_line:
        parts = last_loss_line.split(', ')
        metrics['train_loss'] = float(parts[0].split('Train loss: ')[1].strip())
        metrics['valid_loss'] = float(parts[1].split('Valid loss: ')[1].strip())
        metrics['elapsed_time'] = float(parts[2].split('Elapsed_time: ')[1].strip())
    if last_current_line:
        parts = last_current_line.split(', ')
        metrics['current_accuracy'] = float(parts[0].split(' : ')[1].strip())
        metrics['current_norm_ed'] = float(parts[1].split(' : ')[1].strip())
    if last_best_line:
        parts = last_best_line.split(', ')
        metrics['best_accuracy'] = float(parts[0].split(' : ')[1].strip())
        metrics['best_norm_ed'] = float(parts[1].split(' : ')[1].strip())
    return metrics

def parse_log_dataset(log_content: str) -> dict:
    metrics = {}
    lines = log_content.strip().split('\n')
    for line in lines:
        if 'num total samples of' in line:
            metrics['num_train_samples'] = int(line.split(': ')[1].split(' x')[0].strip())
        if 'sub-directory: /. num samples:' in line:
            metrics['num_valid_samples'] = int(line.split('num samples: ')[1].strip())
    return metrics

def send_to_django(opt, metrics, dataset_metrics, model_path):
    data = {
        "experiment_name": opt.experiment_name,
        "elapsed_time": str(metrics.get('elapsed_time', '')),
        "train_loss": str(metrics.get('train_loss', '')),
        "valid_loss": str(metrics.get('valid_loss', '')),
        "current_accuracy": str(metrics.get('current_accuracy', '')),
        "current_norm_ed": str(metrics.get('current_norm_ed', '')),
        "best_accuracy": str(metrics.get('best_accuracy', '')),
        "best_norm_ed": str(metrics.get('best_norm_ed', '')),
        "num_train_samples": str(dataset_metrics.get('num_train_samples', '')),
        "num_valid_samples": str(dataset_metrics.get('num_valid_samples', ''))
    }

    files = {}
    if os.path.exists(model_path):
        files['model_file'] = ('best_accuracy.pth', open(model_path, 'rb'), 'application/octet-stream')
    else:
        logger.warning(f"Model file {model_path} not found")

    try:
        response = requests.post(DJANGO_URL, data=data, files=files, auth=DJANGO_AUTH)
        logger.info(f"Django response: {response.status_code} {response.text}")
        if not response.ok:
            logger.error(f"Failed to send to Django: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending to Django: {str(e)}\n{traceback.format_exc()}")
    finally:
        for file_key, file_tuple in files.items():
            file_tuple[1].close()  # Close file handle

def run_training(opt):
    try:
        train(opt)
        
        # After training, collect and send data to Django
        experiment_dir = f'./saved_models/{opt.experiment_name}'
        
        # Read logs
        log_train_path = f'{experiment_dir}/log_train.txt'
        log_dataset_path = f'{experiment_dir}/log_dataset.txt'
        model_path = f'{experiment_dir}/best_accuracy.pth'
        
        metrics = {}
        dataset_metrics = {}
        
        if os.path.exists(log_train_path):
            with open(log_train_path, 'r', encoding='utf8') as f:
                log_train_content = f.read()
            metrics = parse_log_train(log_train_content)
        else:
            logger.warning(f"Log file {log_train_path} not found")
        
        if os.path.exists(log_dataset_path):
            with open(log_dataset_path, 'r', encoding='utf8') as f:
                log_dataset_content = f.read()
            dataset_metrics = parse_log_dataset(log_dataset_content)
        else:
            logger.warning(f"Log file {log_dataset_path} not found")
        
        # Send to Django
        send_to_django(opt, metrics, dataset_metrics, model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")

@app.post("/predict")
async def predict(full_photo: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        logger.info("Received request to /predict")
        
        # Read the uploaded file
        contents = await full_photo.read()
        logger.info(f"Uploaded file: {full_photo.filename}, size: {len(contents)} bytes")
        
        # Handle empty upload
        if len(contents) == 0:
            logger.error("Uploaded file is empty")
            return {"error": "Uploaded file is empty or invalid"}
        
        # Convert to base64 for full_photo
        full_photo_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Convert to NumPy array and decode as image
        nparr = np.frombuffer(contents, np.uint8)
        try:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except cv2.error as e:
            logger.error(f"OpenCV decode error: {e}")
            return {"error": "Failed to process image"}
        
        if img is None:
            logger.error("Failed to decode image")
            return {"error": "Invalid image file"}
        
        # Run multi-class vehicle detection on the full image
        vehicle_attributes = detect_vehicle_attributes(img)
        
        # Print the multi-class predictions
        print("Predicted labels for image:")
        for category, value in vehicle_attributes.items():
            print(f"{category}: {value}")
        
        results = detector(img, verbose=False)

        plates = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:
                    conf = box.conf.item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plates.append((conf, x1, y1, x2, y2))
                    logger.info(f"Detected plate with confidence: {conf}")

        if not plates:
            logger.warning("No license plate detected")
            return {"error": "No license plate detected"}

        # Select the detection with the highest confidence
        plates.sort(reverse=True)  # Sort by confidence descending
        _, x1, y1, x2, y2 = plates[0]
        logger.info(f"Selected plate bounding box: ({x1}, {y1}, {x2}, {y2})")

        # Crop the detected plate
        padding = 1
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        cropped_img = img[y1:y2, x1:x2]
        logger.info(f"Cropped plate dimensions: {cropped_img.shape}")

        # Convert cropped_img to base64
        _, buffer = cv2.imencode('.jpg', cropped_img)  # Encode as JPEG
        cropped_img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Updated: Unpack OCR results to include char_details
        ocr_result = recognizer.predict(cropped_img)
        plate_number, precision, char_details = ocr_result
        logger.info(f"Recognized plate number: {plate_number}")
        
        # Handle empty plate_number
        if not plate_number:
            plate_number = "TRAIN"
            precision = 0.0
            char_details = []
        
        # Format char_details as list of dicts for JSON serialization
        formatted_char_details = [{"character": char, "confidence": conf} for char, conf in char_details]
        
        response = {
            "plate_number": plate_number,
            "precision": precision,
            "char_details": formatted_char_details,
            "vehicle_attributes": vehicle_attributes,
            #"full_photo_base64": full_photo_base64,
            "cropped_img_base64": cropped_img_base64,
        }
        print(response["plate_number"])

        return response

    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Exception occurred: {error_msg}\n{stack_trace}")
        return {"error": error_msg}

@app.post("/predict-segment")
async def predict_segment(full_photo: UploadFile = File(...), segment_photo: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        logger.info("Received request to /predict-segment")
        
        # Read the full_photo file
        full_contents = await full_photo.read()
        logger.info(f"Uploaded full_photo: {full_photo.filename}, size: {len(full_contents)} bytes")
        
        # Handle empty full_photo
        if len(full_contents) == 0:
            logger.error("Uploaded full_photo is empty")
            return {"error": "Uploaded full_photo is empty or invalid"}
        
        # Convert to base64 for full_photo (commented out in response as in original)
        full_photo_base64 = base64.b64encode(full_contents).decode('utf-8')
        
        # Convert to NumPy array and decode as image
        full_nparr = np.frombuffer(full_contents, np.uint8)
        try:
            full_img = cv2.imdecode(full_nparr, cv2.IMREAD_COLOR)
        except cv2.error as e:
            logger.error(f"OpenCV decode error for full image: {e}")
            return {"error": "Failed to process full image"}
        
        if full_img is None:
            logger.error("Failed to decode full image")
            return {"error": "Invalid full image file"}
        
        # Run multi-class vehicle detection on the full image
        vehicle_attributes = detect_vehicle_attributes(full_img)
        
        # Print the multi-class predictions
        print("Predicted labels for image:")
        for category, value in vehicle_attributes.items():
            print(f"{category}: {value}")
        
        # Read the segment_photo file
        segment_contents = await segment_photo.read()
        logger.info(f"Uploaded segment_photo: {segment_photo.filename}, size: {len(segment_contents)} bytes")
        
        # Handle empty segment_photo
        if len(segment_contents) == 0:
            logger.error("Uploaded segment_photo is empty")
            return {"error": "Uploaded segment_photo is empty or invalid"}
        
        # Convert to NumPy array and decode as image
        segment_nparr = np.frombuffer(segment_contents, np.uint8)
        try:
            segment_img = cv2.imdecode(segment_nparr, cv2.IMREAD_COLOR)
        except cv2.error as e:
            logger.error(f"OpenCV decode error for segment image: {e}")
            return {"error": "Failed to process segment image"}
        
        if segment_img is None:
            logger.error("Failed to decode segment image")
            return {"error": "Invalid segment image file"}
        
        # Convert segment_img to base64
        _, buffer = cv2.imencode('.jpg', segment_img)  # Encode as JPEG
        segment_img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Updated: Unpack OCR results to include char_details
        ocr_result = recognizer.predict(segment_img)
        plate_number, precision, char_details = ocr_result
        logger.info(f"Recognized plate number: {plate_number}")
        
        # Handle empty plate_number
        if not plate_number:
            plate_number = "TRAIN"
            precision = 0.0
            char_details = []
        
        # Format char_details as list of dicts for JSON serialization
        formatted_char_details = [{"character": char, "confidence": conf} for char, conf in char_details]
        
        response = {
            "plate_number": plate_number,
            "precision": precision,
            "char_details": formatted_char_details,
            "vehicle_attributes": vehicle_attributes,
            #"full_photo_base64": full_photo_base64,
            "cropped_img_base64": segment_img_base64,
        }
        print(response["plate_number"])

        return response

    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Exception occurred: {error_msg}\n{stack_trace}")
        return {"error": error_msg}

@app.post("/train")
async def train_api(background_tasks: BackgroundTasks, config: Dict[str, Any] = Body(...), credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        logger.info("Received request to /train")
        experiment_name = config.get('experiment_name')
        if not experiment_name:
            raise HTTPException(status_code=400, detail="experiment_name is required")
        saved_dir = f'./saved_models/{experiment_name}'
        if os.path.exists(saved_dir):
            raise HTTPException(status_code=409, detail="Experiment name already exists")
        opt = get_config_from_dict(config)
        background_tasks.add_task(run_training, opt)
        return {"message": "Training started in the background"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Exception occurred: {error_msg}\n{stack_trace}")
        raise HTTPException(status_code=500, detail=error_msg)