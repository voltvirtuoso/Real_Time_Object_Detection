# Real-Time Object Detection with YOLOv8

This repository demonstrates a real-time object detection system using the YOLOv8 model and OpenCV to capture live video from an IP camera. The detection is performed frame-by-frame, with bounding boxes drawn around detected objects, and their classes and confidence scores displayed. It also includes functionality to resize video frames while maintaining the aspect ratio.

## Features
- **YOLOv8 Model Integration:** Leverages the YOLOv8 model for object detection.
- **IP Camera Support:** Streams video from an IP camera for real-time processing.
- **Frame Resizing with Aspect Ratio:** Allows resizing the largest frame dimension to a target size while maintaining the original aspect ratio.
- **Confidence Filtering:** Filters detected objects based on a configurable confidence threshold.
- **Real-Time Display:** Shows the detection results with bounding boxes and labels in real time.

## Requirements
- Python 3.x
- [Ultralytics YOLO package](https://github.com/ultralytics/ultralytics)
- OpenCV (`cv2`)

You can install the dependencies using the following command:
```bash
pip install ultralytics opencv-python
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/voltvirtuoso/Real_Time_Object_Detection.git
   cd Real_Time_Object_Detection
   ```

2. **Set the YOLOv8 model path:**
   Update the `model_path` in the code to point to your trained YOLOv8 model weights. Example:
   ```python
   model_path = '..\best.pt'
   ```

3. **Set IP camera URL:**
   Update the `ip_camera_url` in the code to your IP camera stream:
   ```python
   ip_camera_url = 'http://192.168.43.1:8080/video'
   ```

## Usage
1. **Run the detection script:**
   To start real-time detection, simply run the Python script:
   ```bash
   python object_detection.py
   ```

2. **Adjust the confidence threshold:**
   The confidence threshold is set at 0.85 by default, but you can modify it:
   ```python
   confidence_threshold = 0.85
   ```

3. **Frame resizing with aspect ratio:**
   Set the `target_size` to resize frames while keeping the aspect ratio. Example:
   ```python
   target_size = 640  # Resize the largest dimension to 640 pixels
   ```

4. **Exit:**
   Press the `q` key to exit the video stream and stop the detection.

## Code Overview
The script is structured as follows:

- **Model Loading:** Loads the trained YOLOv8 model for inference.
- **IP Camera Stream:** Captures video frames from the IP camera using OpenCV.
- **Frame Resizing:** Adjusts the size of the video frames while maintaining the aspect ratio.
- **Object Detection:** Detects objects using the YOLOv8 model, filters results based on confidence, and draws bounding boxes and labels on the frame.
- **Display Results:** Displays the video with annotated detection results in real time.

### Key Functions:
- `resize_frame_with_aspect_ratio(frame, target_size)`: Resizes the frame while preserving the aspect ratio.

## Example Output
In this example, the detected objects in the frame will be highlighted with bounding boxes, and each object’s label and confidence score will be displayed:

```
Real-time Detection:
- Person 0.89
- Car 0.92
- Dog 0.78
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
