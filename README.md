# Multi-Cam Object Tracking Using YOLOv8

## Overview
This repository demonstrates real-time object detection and tracking using YOLOv8. It supports both video files and live streams, offering high-speed inference and accurate object detection. The project is split into two primary scripts:

1. `object_detection.py`: Handles processing of videos and images for object detection.
2. `RTOD.py`: Facilitates real-time object detection from an IP camera feed.

## Features
- **Real-time Object Detection**: Processes video frames in real-time using YOLOv8.
- **Batch Processing**: Handles multiple video/image files from a directory.
- **Resizing with Aspect Ratio**: Ensures input frames maintain their original aspect ratio.
- **Metadata Generation**: Saves detection results in a metadata file.
- **Dynamic FPS Handling**: Adjusts frame skipping based on FPS settings.
- **Live Stream Integration**: Processes video streams from IP cameras.

## Requirements
To run this project, you need:

- Python 3.7+
- OpenCV
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- NumPy

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## File Structure
```
.
├── object_detection.py   # Processes video and image files
├── RTOD.py               # Real-time object detection from IP camera
├── requirements.txt      # Python dependencies
├── best.pt               # Trained YOLO model weights
├── output/               # Output directory for processed files
└── README.md             # Project documentation
```

## Usage

### 1. Object Detection
To process videos or images, use `object_detection.py`.

#### Command:
```bash
python object_detection.py
```
#### Key Parameters:
- `model_path`: Path to the YOLO model weights (default: `./best.pt`).
- `input_path`: Directory or file to process (default: `./`).
- `export_dir`: Directory to save processed files (default: `./output`).
- `show_video`: Set to `True` to display processed video in real-time.
- `confidence_threshold`: Confidence threshold for detection (default: `0.5`).

### 2. Real-Time Object Detection
To run real-time object detection from an IP camera, use `RTOD.py`.

#### Command:
```bash
python RTOD.py
```
#### Key Parameters:
- `ip_camera_url`: URL of the IP camera (e.g., `http://192.168.4.105:8080/video`).
- `model_name`: YOLO model name (default: `yolov8n.pt`).
- `confidence_threshold`: Confidence threshold for detection (default: `0.75`).
- `target_size`: Resize dimension for input frames (default: `640`).
- `fps_option`: Set FPS mode (`'auto'` or an integer, e.g., `30`).

## Outputs
- **Processed Video/Image**: Saved in the `output/` directory.
- **Metadata**: Detection results saved in `metadata.txt`, including object labels and confidence scores.

## Example
### Detecting Objects in a Video File
```bash
python object_detection.py --input_path ./videos/sample.mp4 --export_dir ./output --show_video True
```

### Running Real-Time Detection
```bash
python RTOD.py --ip_camera_url http://192.168.4.105:8080/video --fps_option auto
```

## Notes
- Ensure the YOLO model weights (`best.pt` or `yolov8n.pt`) are available in the repository.
- For live stream processing, verify the IP camera URL is accessible.
- Press `q` to stop video processing or real-time detection.

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection framework.
- OpenCV for image and video processing.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
Happy coding! 🚀

