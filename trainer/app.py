# Updated app.py: Integrates YOLOv8 for license plate detection before recognition
# Prerequisites:
# - Install Ultralytics for YOLOv8: pip install ultralytics
# - Ensure you have the plate detection model file: 'plate_detection.pt' (YOLOv8 format)
# - Keep the existing LicensePlateRecognizer in license_plate_recognizer.py
# - Update MODEL_PATH, CONFIG_PATH, and add DETECTION_MODEL_PATH
# - Run the app: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from license_plate_recognizer import LicensePlateRecognizer
from ultralytics import YOLO
import numpy as np
import cv2
import io

app = FastAPI()
security = HTTPBasic()

# USERNAME = os.getenv("BASIC_AUTH_USERNAME")
# PASSWORD = os.getenv("BASIC_AUTH_PASSWORD")
USERNAME = "admin"
PASSWORD = "Admin@123"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Initialize the recognizer with your model and config paths
MODEL_PATH = 'saved_models/mn_filtered_v24-10k<image/best_accuracy.pth' # OCR model
CONFIG_PATH = 'config_files/mn_filtered_config_v6m1000.yaml'  # OCR config
DETECTION_MODEL_PATH = 'saved_models/plate_detection/plate_detection.pt'

recognizer = LicensePlateRecognizer(MODEL_PATH, CONFIG_PATH)
detector = YOLO(DETECTION_MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to NumPy array and decode as image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image file"}
        
        results = detector(img, verbose=False)

        plates = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:

                    conf = box.conf.item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plates.append((conf, x1, y1, x2, y2))

        if not plates:
            return { "error": "No license plate detected"}

        # Select the detection with the highest confidence
        plates.sort(reverse=True) # Sort by confidence descending
        _, x1, y1, x2, y2 = plates[0]

        # Crop the detected plate
        padding = 1
        h, w = img.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        cropped_img =  img[y1:y2, x1:x2]

        plate_number = recognizer.predict(cropped_img)
        
        response = {"plate_number": plate_number}
        print(response["plate_number"])

        return response

    except Exception as e:
        return {"error": str(e)}