# Pedestrian Detector

Real-time pedestrian detection in video using YOLOv4-tiny and OpenCV's DNN module.

## Overview

This project runs a YOLOv4-tiny object detector over video frames, filters detections down to the `person` class, and draws bounding boxes around every pedestrian found — frame by frame, live.

## How it works

1. Loads a pretrained YOLOv4-tiny network through `cv2.dnn.readNetFromDarknet`
2. Converts each frame into a blob and runs a forward pass through the network
3. Keeps only detections classified as `person`, above a confidence threshold
4. Applies Non-Maximum Suppression to remove duplicate, overlapping boxes
5. Draws the final bounding boxes on the frame and displays it live

## Built with

- Python
- OpenCV (`cv2.dnn`)
- YOLOv4-tiny (Darknet)
- NumPy, imutils

## Setup

```bash
git clone https://github.com/akshiit02/Pedestrian-Detector.git
cd Pedestrian-Detector
pip install opencv-python numpy imutils
```

You'll also need the YOLOv4-tiny weights file (not included here due to size) — download `yolov4-tiny.weights` from the [official Darknet releases](https://github.com/AlexeyAB/darknet/releases) and place it in this folder alongside `Trainer.cfg` (the network config) and `jl.names` (COCO class names). Update the file paths in `Pedestrian_detection.py` to match, point `cap = cv2.VideoCapture(...)` at your own video, then run:

```bash
python Pedestrian_detection.py
```

## What I'd add next

- Swap in YOLOv8 for better accuracy
- Add an FPS counter and basic benchmarking
- Log detection counts to a file instead of just displaying live video
