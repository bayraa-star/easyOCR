import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from lpr_result import LicensePlateRecognizer
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("license_plate_recognition.log"),
        logging.StreamHandler(),
    ],
)

# Load custom YOLOv8 model trained on license plates
try:
    logging.info("Loading YOLOv8 model...")
    model = YOLO("../models/best_last_version.pt")
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model: {e}")
    raise

# Create output folder
os.makedirs("easy", exist_ok=True)
logging.info("Output folder 'easy' is ready.")

# Initialize the recognizer
try:
    logging.info("Initializing LicensePlateRecognizer...")
    recognizer = LicensePlateRecognizer(
        model_path="./saved_models/mn_filtered_v16-1200image/ocr.pth",
        config_path="./config_files/mn_filtered_config_v6m1000.yaml",
    )
    logging.info("LicensePlateRecognizer initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize LicensePlateRecognizer: {e}")
    raise

# Open video stream
try:
    logging.info("Opening video stream from '../videos'...")
    cap = cv2.VideoCapture("../videos/mvideo.mp4")
    if not cap.isOpened():
        raise IOError("Cannot open video file.")
    logging.info("Video stream opened successfully.")
except Exception as e:
    logging.error(f"Failed to open video stream: {e}")
    raise

frame_id = 0
total_frames_processed = 0
total_plates_detected = 0

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream reached.")
            break

        total_frames_processed += 1
        logging.debug(f"Processing frame {frame_id}")

        # YOLOv8 inference
        logging.debug("Running YOLOv8 inference...")
        results = model(frame)
        logging.debug(f"YOLOv8 inference completed. Found {len(results)} result(s).")

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                logging.debug(
                    f"Detection: Confidence={conf:.2f}, Coordinates=({x1},{y1})-({x2},{y2})"
                )

                if conf < 0.5:
                    logging.debug("Skipping detection due to low confidence.")
                    continue

                # Crop license plate region
                try:
                    plate_crop = frame[y1:y2, x1:x2]
                    logging.debug("License plate region cropped successfully.")
                except Exception as e:
                    logging.error(f"Failed to crop license plate: {e}")
                    continue

                # Predict plate text
                try:
                    plate_text = recognizer.predict(plate_crop)
                    logging.debug(f"Predicted plate text: {plate_text}")
                except Exception as e:
                    logging.error(f"Failed to predict plate text: {e}")
                    plate_text = "unreadable"

                # Annotate detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                # Save cropped plate with metadata if readable
                if plate_text != "unreadable":
                    total_plates_detected += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"easy/{timestamp}_{plate_text}.jpg"
                    try:
                        cv2.imwrite(filename, plate_crop)
                        logging.info(f"Saved cropped plate: {filename}")
                    except Exception as e:
                        logging.error(f"Failed to save cropped plate: {e}")

                    try:
                        with open("easy/plate_log.csv", "a") as log_file:
                            log_file.write(
                                f"{timestamp},{plate_text},{x1},{y1},{x2},{y2}\n"
                            )
                        logging.debug("Logged plate information to CSV.")
                    except Exception as e:
                        logging.error(f"Failed to write to log file: {e}")

        cv2.imshow("License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("User interrupted the process.")
            break

        frame_id += 1

    except Exception as e:
        logging.error(f"Error processing frame {frame_id}: {e}")
        logging.debug(traceback.format_exc())

cap.release()
cv2.destroyAllWindows()

logging.info(f"Total frames processed: {total_frames_processed}")
logging.info(f"Total license plates detected: {total_plates_detected}")
logging.info("License plate recognition completed.")
