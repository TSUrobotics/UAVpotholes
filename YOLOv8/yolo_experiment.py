import cv2
import time
from ultralytics import YOLO

# Load a model
model = YOLO("runs/segment/yolov8Potholes10/weights/best.pt")  # load a custom model

video_path = "../uav/potholeB_30deg_10m.mp4"

if False:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    start_time = time.time()

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            results = model(frame)
        else:
            break
            
if False:
    start_time = time.time()

    results = model(video_path)
    
if True:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            frames.append(frame)
        else:
            break
    
    start_time = time.time()
    results = model(frames)
        
total_time = time.time() - start_time

print(f"total time spent: {total_time:.5f}s")