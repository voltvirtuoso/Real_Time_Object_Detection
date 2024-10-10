import cv2
from ultralytics import YOLO
import time
import os
import glob

# Step 1: Load the trained YOLO model
model_path = './best.pt'
model = YOLO(model_path)

# Set the confidence threshold
confidence_threshold = 0.5

# Parameters
input_path = "./"  # Directory or file path
export_dir = "./output"  # Directory for saving outputs (None if you don't want to save)
output_filename = None  # Set to None if you want to use original names
metadata_filename = "metadata.txt"
show_video = True  # Set to True if you want to see the video being processed

# Resize parameters
target_size = None  # Largest dimension size for resizing

# Function to resize keeping aspect ratio
def resize_with_aspect_ratio(frame, target_size):
    if target_size is None:
        return frame, (frame.shape[1], frame.shape[0])
    h, w = frame.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(target_size * w / h)
    else:
        new_w = target_size
        new_h = int(target_size * h / w)
    return cv2.resize(frame, (new_w, new_h)), (new_w, new_h)

# Function to process a video file
def process_video(video_path, output_filename, show_video):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    # Initialize video writer
    out = None
    frame_size = None
    detected_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        resized_frame, frame_size = resize_with_aspect_ratio(frame, target_size)

        if out is None and export_dir is not None:
            os.makedirs(export_dir, exist_ok=True)
            output_path = os.path.join(export_dir, output_filename or os.path.basename(video_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), frame_size)

        results = model(resized_frame)
        annotated_frame = resized_frame.copy()

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

                        detected_objects.append({'label': model.names[cls], 'confidence': conf})

        if out is not None:
            out.write(annotated_frame)

        if show_video:
            cv2.imshow('Processed Video', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Save metadata
    metadata_path = os.path.join(export_dir or "./", metadata_filename)
    with open(metadata_path, 'w') as f:
        f.write("Detected Objects Metadata:\n")
        for obj in detected_objects:
            f.write(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}\n")

    print(f"Metadata saved to {metadata_path}")

# Function to process a single image file
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}.")
        return

    resized_image, _ = resize_with_aspect_ratio(image, target_size)
    results = model(resized_image)
    annotated_image = resized_image.copy()
    detected_objects = []

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

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detected_objects.append({'label': model.names[cls], 'confidence': conf})

    # Save output image
    output_path = os.path.join(export_dir or "./", output_filename or os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    print(f"Processed and saved {output_path}")

# Function to process all files (videos and images) in a directory
def process_files_in_directory(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.png']

    # Process all video files
    for ext in video_extensions:
        video_paths = glob.glob(os.path.join(directory, f'*{ext}'))
        for video_path in video_paths:
            process_video(video_path, output_filename, show_video)

    # Process all image files
    for ext in image_extensions:
        image_paths = glob.glob(os.path.join(directory, f'*{ext}'))
        for image_path in image_paths:
            process_image(image_path)

# Main processing logic
if os.path.isfile(input_path):  # Check if it's a single file
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check for video formats
        process_video(input_path, output_filename, show_video)
    elif input_path.lower().endswith(('.jpg', '.png')):  # Check for image formats
        process_image(input_path)
    else:
        print("Error: Unsupported file format. Please provide a valid video or image file.")
elif os.path.isdir(input_path):  # Check if it's a directory
    process_files_in_directory(input_path)
else:
    print("Error: The provided path is neither a valid directory nor a file.")
