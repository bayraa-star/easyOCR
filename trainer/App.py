# app.py
# Updated app.py: Integrates YOLOv8 for license plate detection, OCR, Basic Authentication, returns base64 images, and vehicle multi-class detection

# Prerequisites:
# - Install Ultralytics for YOLOv8: pip install ultralytics
# - Install FastAPI security dependencies: pip install fastapi[all]
# - Install python-dotenv: pip install python-dotenv
# - Run the app: uvicorn App:app --host 0.0.0.0 --port 8000 --reload

import os
import base64
import logging  # Added for more error logging
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from license_plate_recognizer import LicensePlateRecognizer
from ultralytics import YOLO
import numpy as np
import cv2
import io
import traceback  # Added for detailed stack traces
from multiclass.multiclass_detector import detect_vehicle_attributes  # Import the multi-class function

app = FastAPI()
security = HTTPBasic()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded credentials for testing
USERNAME = "admin"
PASSWORD = "Admin@123"

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
MODEL_PATH = 'saved_models/mn_filtered_v24-85Kimage/best_accuracy.pth'  # OCR model (fix path if needed)
CONFIG_PATH = 'config_files/mn_filtered_config_v6m1000.yaml'  # OCR config
DETECTION_MODEL_PATH = 'saved_models/plate_detection/plate_detection.pt'

recognizer = LicensePlateRecognizer(MODEL_PATH, CONFIG_PATH)
detector = YOLO(DETECTION_MODEL_PATH)

@app.post("/predict")
async def predict(full_photo: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        logger.info("Received request to /predict")
        
        # Read the uploaded file
        contents = await full_photo.read()
        logger.info(f"Uploaded file: {full_photo.filename}, size: {len(contents)} bytes")
        
        # Convert to base64 for full_photo
        full_photo_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Convert to NumPy array and decode as image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
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

        plate_number = recognizer.predict(cropped_img)
        logger.info(f"Recognized plate number: {plate_number}")
        
        response = {
            "plate_number": plate_number,
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
        
        # Convert to base64 for full_photo (commented out in response as in original)
        full_photo_base64 = base64.b64encode(full_contents).decode('utf-8')
        
        # Convert to NumPy array and decode as image
        full_nparr = np.frombuffer(full_contents, np.uint8)
        full_img = cv2.imdecode(full_nparr, cv2.IMREAD_COLOR)
        
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
        
        # Convert to NumPy array and decode as image
        segment_nparr = np.frombuffer(segment_contents, np.uint8)
        segment_img = cv2.imdecode(segment_nparr, cv2.IMREAD_COLOR)
        
        if segment_img is None:
            logger.error("Failed to decode segment image")
            return {"error": "Invalid segment image file"}
        
        # Convert segment_img to base64
        _, buffer = cv2.imencode('.jpg', segment_img)  # Encode as JPEG
        segment_img_base64 = base64.b64encode(buffer).decode('utf-8')

        plate_number = recognizer.predict(segment_img)
        logger.info(f"Recognized plate number: {plate_number}")
        
        response = {
            "plate_number": plate_number,
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
