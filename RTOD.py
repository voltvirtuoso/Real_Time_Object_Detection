import cv2
from ultralytics import YOLO
import time
import numpy as np

# Load the trained YOLO model
model_path = "./"
model_name = "yolov8n.pt"

model = YOLO(model_path + model_name)

# Set the confidence threshold
confidence_threshold = 0.75

# Set the target resize dimension (largest dimension will be resized to this)
target_size = 640    # Use None if no resizing is required

# Set the FPS option ('auto' or hard-coded integer like 30)
fps_option = 'auto'  # Change to a number to set manually (e.g., 30)

# Initialize the video capture with IP camera URL
ip_camera_url = 'http://192.168.4.105:8080/video'  # Replace with your IP camera URL

cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# If source FPS is set manually, it will override the auto calculation
source_fps = 'auto'  # Set to a specific value (e.g., 25) if you know the camera's FPS

# Function to resize the frame while keeping the aspect ratio
def resize_frame_with_aspect_ratio(frame, target_size):
    if target_size is None:
        return frame
    h, w = frame.shape[:2]
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    return cv2.resize(frame, (new_w, new_h))

# Function to draw FPS on the frame
def draw_fps(frame, fps):
    h, w = frame.shape[:2]
    font_scale = min(w, h) / 1000  # Scale font size based on frame size
    font_color = (255, 0, 0)
    thickness = 2
    position = (w - 150, 50)  # Top-right corner
    cv2.putText(frame, f'FPS: {int(fps)}', position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

# Function to calculate FPS based on inference time
def calculate_fps(inference_times):
    mean_time = np.mean(inference_times)
    fps = int(1 / mean_time) if mean_time > 0 else 1
    return max(1, fps)  # Ensure FPS is at least 1

# Function to determine frame skipping pattern based on target FPS
def get_frame_skip_pattern(source_fps, target_fps):
    skip_pattern = []
    if source_fps <= target_fps:
        return skip_pattern  # No skipping needed if source FPS is less or equal
    skip_interval = max(1, source_fps // target_fps)
    for i in range(1, int(source_fps) + 1):
        if (i - 1) % skip_interval != 0:
            skip_pattern.append(i - 1)
    return skip_pattern

# Step 3: Start capturing video and performing inference
inference_times = []
frame_index = 0
prev_time = time.time()

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("Warning: Could not read frame, attempting to reinitialize.")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(ip_camera_url)
        if not cap.isOpened():
            print("Error: Could not reinitialize video capture.")
            break
        continue

    # Calculate dynamic FPS based on frame capture time if source FPS is set to 'auto'
    if source_fps == 'auto':
        current_time = time.time()
        time_between_frames = current_time - prev_time
        prev_time = current_time
        dynamic_fps = int(1 / time_between_frames) if time_between_frames > 0 else 1
        dynamic_fps = max(dynamic_fps, 1)  # Ensure dynamic FPS is at least 1
    else:
        # Use the hardcoded source FPS directly (minimize CPU load)
        dynamic_fps = max(int(source_fps), 1)

    frame = resize_frame_with_aspect_ratio(frame, target_size)
    results = model(frame)

    annotated_frame = frame.copy()
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(confidences)):
                if confidences[i] >= confidence_threshold:
                    box = boxes[i]
                    conf = confidences[i]
                    cls = int(classes[i])
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS if in 'auto' mode
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    
    if fps_option == 'auto':
        fps = calculate_fps(inference_times[-20:])  # Calculate based on the last 20 inferences
    else:
        fps = fps_option

    # Generate frame skip pattern dynamically based on the dynamic FPS
    frame_skip_pattern = get_frame_skip_pattern(dynamic_fps, int(fps))
    
    # Skip frames based on the pattern (handle edge cases)
    if frame_skip_pattern and frame_index % dynamic_fps in frame_skip_pattern:
        frame_index += 1
        continue

    frame_index += 1

    draw_fps(annotated_frame, fps)
    cv2.imshow('Real-time Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
