# AI-Powered Driver-Aware ADAS Simulator

It is an AI-powered Advanced Driver Assistance System (ADAS) simulator built using CARLA Simulator, Python, YOLOv8, MediaPipe, and OpenCV. It combines real-time road perception with driver monitoring to improve driving safety through intelligent vehicle assistance.

## Features

- Adaptive Cruise Control (ACC)
- Automatic Emergency Braking (AEB)
- Lane Keeping Assist (LKA)
- Vehicle & Pedestrian Detection
- Traffic Light Detection
- Speed Limit Assist
- Driver Drowsiness Detection
- Yawning Detection
- AI Risk Score
- Safe Stop Assist

## Technologies Used

- Python
- CARLA Simulator
- OpenCV
- MediaPipe
- YOLOv8
- Pygame
- NumPy

## Requirements

- Python 3.10
- CARLA Simulator 0.9.16

## Installation
Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Pre-trained Models

Download the required YOLOP files and model weights from:

1.https://drive.google.com/drive/folders/1Tq_mOvXqLKHsJOzXH_jR6nI1PmLsoLVx?usp=sharing

2.https://drive.google.com/drive/folders/1g2lJyTQLlS3mIXWOVuGDvPwMxM8-mqvr?usp=sharing

Extract them into:

YOLOP/

weights/

## Running the Simulator

1. Launch **CARLA Simulator**.
2. Connect the simulator.
3. Run the project:

```bash
python camera.py 
```
OR
```bash
py -3.10 camera.py 
```
## Screenshots

### Front Camera (Main ADAS Dashboard)

![Front Camera (Main ADAS Dashboard)](SCREENSHOTS/Front_Camera_(Main_ADAS_Dashboard).png)

### Lane Departure Warning System

![Lane Departure Warning System](SCREENSHOTS/Lane%20Departure%20Warning%20System.png)

### Lane Keeping Assist (LKA)

![Lane Keeping Assist (LKA)](SCREENSHOTS/Lane%20Keeping%20Assist%20(LKA).png)

### Left Camera Monitoring

![Left Camera Monitoring](SCREENSHOTS/Left%20Camera%20Monitoring.png)

### Rear Camera Monitoring

![Rear Camera Monitoring](SCREENSHOTS/Rear%20Camera%20Monitoring.png)

### Right Camera Monitoring (Right Blind Spot Detection)

![Right Camera Monitoring](SCREENSHOTS/Right%20Camera%20Monitoring%20(Right%20Blind%20Spot%20Detection).png)


## Author
**Ishita Das**
