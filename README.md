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
# Screenshots

## Main ADAS Dashboard

![Front Camera-Main ADAS Dashboard](SCREENSHOTS/Front Camera (Main ADAS Dashboard.png)

---

## Lane Departure Warning

![Lane Departure Warning](<img width="959" height="563" alt="Lane Departure Warning System" src="https://github.com/user-attachments/assets/5ae7510d-fcfc-479d-8f59-89029edffed4" />
)

---

## Lane Keeping Assist (LKA)

![Lane Keeping Assist](<img width="959" height="560" alt="Lane Keeping Assist (LKA)" src="https://github.com/user-attachments/assets/bb6fe0ea-77bf-47c3-bcc7-14aa23a7542e" />
)

---

## Rear Camera Monitoring

![Rear Camera](<img width="959" height="563" alt="Rear Camera Monitoring" src="https://github.com/user-attachments/assets/aa90800e-21bb-420c-8b37-8b64e3ad5c71" />
)

---

## Right Blind Spot Detection

![Right Camera Monitoring-Right Blind Spot Detection](<img width="959" height="560" alt="Right Camera Monitoring (Right Blind Spot Detection)" src="https://github.com/user-attachments/assets/b34bd53a-c644-492c-aedc-5d33b42ccf44" />
)

---

## Rear Vehicle Detection

![Rear Vehicle Detection](screenshots/rear_vehicle_detection.png)

---

## Left Side Camera View

![Left Camera Monitoring](<img width="959" height="563" alt="Left Camera Monitoring" src="https://github.com/user-attachments/assets/a91fc618-5029-484c-9228-a4e3ea93c077" />
)

## Author
**Ishita Das**
