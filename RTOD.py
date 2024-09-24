import cv2
from ultralytics import YOLO
import time

# Step 1: Load the trained YOLO model
model_path = '/home/haroon-raza/Documents/Models/ver_3/output_v3/yolo_experiment/weights/best.pt'
model = YOLO(model_path)

# Set the confidence threshold
confidence_threshold = 0.85

# Set the target resize dimension (largest dimension will be resized to this)
target_size = None

# Step 2: Initialize the video capture with IP camera URL
ip_camera_url = 'http://192.168.43.1:8080/video'  # Replace with your IP camera URL
cap = cv2.VideoCapture(ip_camera_url)

# Check if the video capture is opened
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Function to resize the frame while keeping the aspect ratio
def resize_frame_with_aspect_ratio(frame, target_size):
    if target_size is None:
        return frame
    
    h, w = frame.shape[:2]
    # Determine which dimension is the largest
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    # Resize the frame to the new dimensions
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

# Step 3: Start capturing video and performing inference
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not read correctly, try to reinitialize the camera
    if not ret:
        print("Warning: Could not read frame, attempting to reinitialize.")
        cap.release()
        time.sleep(1)  # Wait for a short period before trying again
        cap = cv2.VideoCapture(ip_camera_url)
        if not cap.isOpened():
            print("Error: Could not reinitialize video capture.")
            break
        continue
    
    # Resize the frame while maintaining aspect ratio
    frame = resize_frame_with_aspect_ratio(frame, target_size)
    
    # Perform inference
    results = model(frame)
    
    # Initialize an empty frame for annotations
    annotated_frame = frame.copy()

    # Process each result
    for result in results:
        if result.boxes is not None:
            # Access bounding boxes, confidences, and classes
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Filter detections based on confidence
            for i in range(len(confidences)):
                if confidences[i] >= confidence_threshold:
                    box = boxes[i]
                    conf = confidences[i]
                    cls = int(classes[i])
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Detection', annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
