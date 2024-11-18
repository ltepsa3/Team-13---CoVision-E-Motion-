import cv2
import torch
from ultralytics import YOLO
import os
import time

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model and move it to the GPU if available
model = YOLO('yolov8n.pt')

# Blur ratio
blur_ratio = 50

# Detection flag and file-open flag
detected = False
files_opened_once = False  # Global flag to prevent multiple openings of files

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def preprocess_frame(frame):
    # Resize frame to 640x640 and normalize for YOLOv8
    resized_frame = cv2.resize(frame, (640, 640))
    frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return frame_tensor.to(device)


def blur_and_annotate_humans(frame):
    global detected, files_opened_once

    # Perform inference on the frame using YOLOv8
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()

    # Detect person class with confidence > 0.9
    person_detected = any(int(cls) == 0 and conf > 0.9 for cls, conf in zip(clss, confs))

    if person_detected and not files_opened_once:
        try:
            # Open photo once
            os.startfile(r'C:\Users\jeanj\Downloads\50f.png')
            print("Photo opened.")
        except Exception as e:
            print(f"Failed to open photo: {e}")

        try:
            # Open video once
            os.startfile(r'C:\Users\jeanj\Downloads\Vine boom sound effect.mp4')
            print("Video opened.")
        except Exception as e:
            print(f"Failed to open video: {e}")

        files_opened_once = True  # Prevent further openings during the runtime

    if not person_detected and detected:
        # Close applications when no person is detectedq
        os.system(r'TASKKILL /F /IM Microsoft.Photos.exe')
        os.system(r'TASKKILL /F /IM wmplayer.exe')
        time.sleep(1)
        detected = False  # Detection reset without affecting file open status

    # Process each detected bounding box
    for box, cls, conf in zip(boxes, clss, confs):
        if int(cls) == 0 and conf > 0.9:  # 0 is the class ID for 'person'
            x1, y1, x2, y2 = map(int, box)
            person_region = frame[y1:y2, x1:x2]
            blurred_person = cv2.blur(person_region, (blur_ratio, blur_ratio))
            frame[y1:y2, x1:x2] = blurred_person
            label = f"Person: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Process and blur humans in the frame
        blurred_frame = blur_and_annotate_humans(frame)

        # Display only the blurred frame
        cv2.imshow("Person Detection", blurred_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()