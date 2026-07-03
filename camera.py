from carla_modules.keyboard_control import control_vehicle
import carla_modules.adas_state as adas
import threading
import carla
import cv2
import numpy as np
import random
import time
from datetime import datetime
import traceback
import time
from driver_monitor import start_driver_monitor
from perception.lane_detection import detect_lanes
from ultrafastLaneDetector.ultrafastLaneDetector import (
    UltrafastLaneDetector,
    ModelType
)

from ultralytics import YOLO
import torch
import onnxruntime as ort
#from ultralytics.utils.ops import non_max_suppression
# =====================================================
# PROFESSIONAL LOGGER
# =====================================================

def log(level, message):
    """
    Professional terminal logger with timestamp.
    Example:
    [14:42:18.215] [ACC] Adaptive Cruise Control ACTIVE
    """

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(f"[{timestamp}] [{level}] {message}")

# -------------------------
# Connect to CARLA
# -------------------------
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
print("Current Map:", world.get_map().name)

blueprint_library = world.get_blueprint_library()

# -------------------------
# Spawn Tesla
# -------------------------
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    print("Vehicle could not be spawned.")
    exit()

log("INFO", "Vehicle spawned!")

# -------------------------
# Load AI Lane Detector
# -------------------------
lane_detector = UltrafastLaneDetector(
    "weights/tusimple_18.onnx",
    ModelType.TUSIMPLE
)
import onnxruntime as ort

yolop_session = ort.InferenceSession(
    "YOLOP/weights/yolop-640-640.onnx"
)

log("SYSTEM", "YOLOP Loaded!")

log("SYSTEM", "AI Lane Detector Loaded!")

yolo_model = YOLO("yolov8n.pt")
yolo_model = YOLO("yolov8n.pt")

# -------------------------
# Traffic Sign Recognition
# -------------------------
# traffic_model = YOLO("weights/traffic_signs.pt")

# print("Traffic Sign Model Loaded!")

# -------------------------
# Blind Spot Flags
# -------------------------
left_blind_spot = False
right_blind_spot = False
# -------------------------
# Rear Collision Warning
# -------------------------
rear_collision = False
rear_distance = "SAFE"
# -------------------------
# Pedestrian Detection
# -------------------------
pedestrian_detected = False
pedestrian_warning = "SAFE"
# -------------------------
# AI Risk Heatmap
# -------------------------
front_risk = "GREEN"
left_risk = "GREEN"
right_risk = "GREEN"
rear_risk = "GREEN"
# -------------------------
# Automatic Emergency Braking
# -------------------------
adas.aeb_active = False
aeb_status = "OFF"
# -------------------------
# Create RGB Camera
# -------------------------
camera_bp = blueprint_library.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)
# -------------------------
# Left Camera
# -------------------------
left_camera_transform = carla.Transform(
    carla.Location(x=0.3, y=-0.9, z=1.6),
    carla.Rotation(yaw=-90)
)
left_camera = world.spawn_actor(
    camera_bp,
    left_camera_transform,
    attach_to=vehicle
)
# -------------------------
# Right Camera
# -------------------------
right_camera_transform = carla.Transform(
    carla.Location(x=0.3, y=0.9, z=1.6),
    carla.Rotation(yaw=90)
)

right_camera = world.spawn_actor(
    camera_bp,
    right_camera_transform,
    attach_to=vehicle
)
# -------------------------
# Rear Camera
# -------------------------
rear_camera_transform = carla.Transform(
    carla.Location(x=-2.0, z=1.6),
    carla.Rotation(yaw=180)
)

rear_camera = world.spawn_actor(
    camera_bp,
    rear_camera_transform,
    attach_to=vehicle
)
def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

    dw = pad_w // 2  # divide padding into 2 sides
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)

# -------------------------
# AI Risk Heatmap Color
# -------------------------
def risk_color(level):

        if level == "RED":
            return (0,0,255)

        elif level == "YELLOW":
            return (0,255,255)

        else:
            return (0,255,0)
# -------------------------
# Camera Callback
# -------------------------
def process_image(image):

    global front_risk
    global left_risk
    global right_risk
    global rear_risk

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((720, 1280, 4))
    img = img[:, :, :3].copy()

    # -------------------------
    # YOLOP Lane Segmentation
    # -------------------------

# Resize image to YOLOP input size
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(
        img[:, :, ::-1],
        (640, 640)
    )

    input_img = canvas.astype(np.float32)
    input_img /= 255.0

    input_img[:, :, 0] = (input_img[:, :, 0] - 0.485) / 0.229
    input_img[:, :, 1] = (input_img[:, :, 1] - 0.456) / 0.224
    input_img[:, :, 2] = (input_img[:, :, 2] - 0.406) / 0.225

    input_img = input_img.transpose(2, 0, 1)
    input_img = np.expand_dims(input_img, axis=0)

    outputs = yolop_session.run(
        None,
        {"images": input_img}
    )
    det_out, drive_area_seg, lane_line_seg = outputs
    print("Drive Area min/max:",
      np.min(drive_area_seg),
      np.max(drive_area_seg))

    print("Lane Line min/max:",
      np.min(lane_line_seg),
      np.max(lane_line_seg))
    # -------------------------
# Remove padding
# -------------------------
    lane_line_seg = lane_line_seg[
        :,
        :,
        dh:dh + new_unpad_h,
        dw:dw + new_unpad_w
    ]

    # -------------------------
    # Get lane mask
    # -------------------------
    lane_mask = np.argmax(lane_line_seg, axis=1)[0]

    print("Lane pixels:", np.sum(lane_mask > 0))
    print("Max value:", np.max(lane_mask))

    # Convert to 0-255
    lane_mask = (lane_mask * 255).astype(np.uint8)

    # Make lanes thicker
    kernel = np.ones((5, 5), np.uint8)
    lane_mask = cv2.dilate(lane_mask, kernel, iterations=3)

    # Resize to camera resolution
    lane_mask = cv2.resize(
        lane_mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    # =====================================
# Find lane centers from YOLOP mask
# =====================================

    height, width = lane_mask.shape

    # Look near the bottom of the image
    y = int(height * 0.80)

    # Take a horizontal slice
    row = lane_mask[y]

    # Find all white pixels
    lane_pixels = np.where(row > 0)[0]

    lane_overlay = np.zeros_like(img)
    lane_overlay[:, :, 1] = lane_mask


    output_img = cv2.addWeighted(
        img,
        1.0,
        lane_overlay,
        0.5,
        0
    )

    left_lane_x = None
    right_lane_x = None

    if len(lane_pixels) > 20:

        center = width // 2

        left_pixels = lane_pixels[lane_pixels < center]
        right_pixels = lane_pixels[lane_pixels > center]

        if len(left_pixels):
            left_lane_x = int(np.mean(left_pixels))

        if len(right_pixels):
            right_lane_x = int(np.mean(right_pixels))

    print("Left Lane :", left_lane_x)
    print("Right Lane:", right_lane_x)

    if left_lane_x is not None:
        cv2.circle(
        output_img,
        (left_lane_x, y),
        8,
        (0,0,255),
        -1
    )

    if right_lane_x is not None:
        cv2.circle(
        output_img,
        (right_lane_x, y),
        8,
        (255,0,0),
        -1
    )
        
    # Vehicle center is always the center of the camera
    vehicle_center = img.shape[1] // 2

    cv2.circle(
    output_img,
    (vehicle_center, y),
    8,
    (255,255,255),
    -1
    )

    offset = None
    lane_center = None
    correction = ""



    if left_lane_x is not None and right_lane_x is not None:

        lane_center = (left_lane_x + right_lane_x) // 2

        cv2.circle(
            output_img,
            (lane_center, y),
            10,
            (0,255,255),
            -1
        )

        cv2.line(
        output_img,
        (vehicle_center, y),
        (lane_center, y),
        (255,255,0),
        3
        )
        offset = lane_center - vehicle_center

        print("Offset:", offset)

        # Determine drift direction and steering correction
        if offset > 0:
            drift = "LEFT"
            correction = "STEER RIGHT"
        else:
            drift = "RIGHT"
            correction = "STEER LEFT"

        # Lane Departure Warning
    if offset is not None:

        if abs(offset) < 40:
            lane_status = "CENTERED"
            lane_color = (0,255,0)

        elif abs(offset) < 120:

            if offset > 0:
                lane_status = "DRIFTING LEFT"
            else:
                lane_status = "DRIFTING RIGHT"

            lane_color = (0,255,255)

        else:

            if offset > 0:
                lane_status = "LANE DEPARTURE LEFT"
            else:
                lane_status = "LANE DEPARTURE RIGHT"

            lane_color = (0,0,255)

    else:
        lane_status = "LANE NOT DETECTED"
        lane_color = (255,255,255)
    
   # -------------------------
    # Lane Keeping Assist
    # -------------------------

    adas.lka_active = False
    adas.lka_steer = 0.0

    if lane_status in ["DRIFTING LEFT", "DRIFTING RIGHT"]:

        adas.lka_active = True

        adas.lka_steer = offset / 300.0

        # Limit maximum steering
        adas.lka_steer = max(min(adas.lka_steer, 0.65), -0.65)
    if adas.lka_active:
        log("LKA", f"Steering = {adas.lka_steer:.2f}")

    cv2.imshow("Lane Mask", lane_mask)

    # Create green overlay

    # Blend with original image
        # AI Lane Detection
    # -------------------------
    try:
        #output_img = img.copy()

        #left_lane_x = None
        #right_lane_x = None

        # Use only the bottom 10 points (closest to the vehicle)
        #if len(lane_points[1]) >= 10:
            #left_lane_x = np.mean([p[0] for p in lane_points[1][:10]])

        #if len(lane_points[2]) >= 10:
            #right_lane_x = np.mean([p[0] for p in lane_points[2][:10]])
        
        #print("=" * 40)
        #print("LEFT LANE :", left_lane_x)
        #print("RIGHT LANE:", right_lane_x)
        #print("=" * 40)

        # -------------------------
        # YOLO Object Detection
        # -------------------------
        results = yolo_model(
            output_img,
            conf=0.5,
            classes=[0, 1, 2, 3, 5, 7, 9, 11]
        )
        # # -------------------------
        # # Traffic Sign Detection
        # # -------------------------
        # traffic_results = traffic_model(
        #     img,
        #     conf=0.40
        # )
        # # -------------------------
        # # Read Traffic Sign Detections
        # # -------------------------
        # traffic_boxes = traffic_results[0].boxes

        # for box in traffic_boxes:

        #     x1, y1, x2, y2 = map(int, box.xyxy[0])

        #     cls = int(box.cls[0])
        #     conf = float(box.conf[0])

        #     label = traffic_model.names[cls]

        #     # Draw bounding box
        #     cv2.rectangle(
        #         img,
        #         (x1, y1),
        #         (x2, y2),
        #         (255, 0, 255),
        #         2
        #     )

        #     # Display class name
        #     cv2.putText(
        #         img,
        #         f"{label} {conf:.2f}",
        #         (x1, y1 - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (255, 0, 255),
        #         2
        #     )

        #     # Print in terminal
        #     print("Traffic Sign:", label)
        boxes = results[0].boxes

        log("YOLO", f"Detections = {len(boxes)}")

        closest_vehicle = None
        largest_height = 0


        traffic_state = "UNKNOWN"
        traffic_color = (255, 255, 255)
        closest_light = None
        largest_area = 0
        stop_sign_detected = False
        stop_sign_box = None
        largest_stop_area = 0

        # -------------------------
        # Pedestrian Detection
        # -------------------------
        largest_pedestrian_height = 0
        closest_pedestrian = None

        pedestrian_detected = False
        pedestrian_warning = "SAFE"

        # Pass 1: locate only the nearest (largest) traffic light
        for box in boxes:
            if int(box.cls[0]) == 9:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area > largest_area:
                    largest_area = area
                    closest_light = (x1,y1,x2,y2)

        # Run HSV detection only once on the nearest traffic light
        if closest_light is not None:
            log("TRAFFIC", "Closest traffic light detected")
            x1,y1,x2,y2 = closest_light
            traffic_roi = img[y1:y2, x1:x2]
            if traffic_roi.size != 0 and traffic_roi.shape[0] >= 10 and traffic_roi.shape[1] >= 10:
                hsv = cv2.cvtColor(traffic_roi, cv2.COLOR_BGR2HSV)

                red = cv2.countNonZero(cv2.inRange(hsv,np.array([0,120,70]),np.array([10,255,255]))) +                       cv2.countNonZero(cv2.inRange(hsv,np.array([170,120,70]),np.array([180,255,255])))
                yellow = cv2.countNonZero(cv2.inRange(hsv,np.array([20,100,100]),np.array([35,255,255])))
                green = cv2.countNonZero(cv2.inRange(hsv,np.array([40,40,40]),np.array([90,255,255])))

                if red > max(yellow, green) and red > 80:
                    traffic_state, traffic_color = "RED", (0,0,255)
                elif yellow > max(red, green) and yellow > 80:
                    traffic_state, traffic_color = "YELLOW", (0,255,255)
                elif green > max(red, yellow) and green > 80:
                    traffic_state, traffic_color = "GREEN", (0,255,0)

                log("HSV", f"R={red} Y={yellow} G={green}")
                log("TRAFFIC", f"Signal = {traffic_state}")

                cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)

        # Pass 2: process vehicles only
        for box in boxes:
            cls = int(box.cls[0])

            # -------------------------
            # Pedestrian Detection
            # -------------------------

            if cls == 0:      # Person

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                height = y2 - y1

                if height > largest_pedestrian_height:

                    largest_pedestrian_height = height
                    closest_pedestrian = (x1, y1, x2, y2)
            # -------------------------
            # Stop Sign Detection
            # -------------------------

            if cls == 11:   # Stop Sign

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)

                if area > largest_stop_area:

                    largest_stop_area = area
                    stop_sign_detected = True
                    stop_sign_box = (x1, y1, x2, y2)
            if cls in [2, 3, 5, 7]:

                x1, y1, x2, y2 = box.xyxy[0]

                height = float(y2 - y1)

                x_center = (x1 + x2) / 2

                # Vehicle near the center of the road
                #if 450 < x_center < 830:

                if height > largest_height:
                    largest_height = height
                    closest_vehicle = (x1, y1, x2, y2)
                
                y_center = (y1 + y2) / 2

                log(
                    "VEHICLE",
                    f"Center=({int(x_center)},{int(y_center)}) "
                    f"Box=({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
                )
                print("------------------")

        warning = "SAFE"
        color = (0, 255, 0)   # Green

        if stop_sign_detected:

            x1, y1, x2, y2 = stop_sign_box

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2
            )

            print("STOP SIGN DETECTED")
            stop_text = "STOP SIGN"
            stop_instruction = "STOP VEHICLE"

        if closest_vehicle is not None:

            if largest_height > 220:
                warning = "COLLISION WARNING!"
                color = (0, 0, 255)     # Red

            elif largest_height > 150:
                warning = "CAUTION"
                color = (0, 255, 255)   # Yellow

            log("FCW", warning)

         # -------------------------
        # Pedestrian Warning
        # -------------------------

        if closest_pedestrian is not None:

            if largest_pedestrian_height > 180:

                pedestrian_detected = True
                pedestrian_warning = "PEDESTRIAN WARNING"

            elif largest_pedestrian_height > 100:

                pedestrian_detected = True
                pedestrian_warning = "PEDESTRIAN AHEAD"

            else:

                pedestrian_detected = False
                pedestrian_warning = "SAFE"

        else:

            pedestrian_detected = False
            pedestrian_warning = "SAFE"

        log("PEDESTRIAN", pedestrian_warning)
        annotated_img = results[0].plot()

        img = cv2.addWeighted(
            output_img,
            1.0,
            annotated_img,
            1.0,
            0
        )
        # if output_img is not None:
#     img = output_img

    except Exception:
        print("Lane Detector Error")
        traceback.print_exc()
    # Keep showing the original camera image
        pass

    # -------------------------
    # Vehicle Speed
    # -------------------------
    velocity = vehicle.get_velocity()


    speed = 3.6 * np.sqrt(
        velocity.x**2 +
        velocity.y**2 +
        velocity.z**2
    )
    # -------------------------
    # Speed Limit Assist
    # -------------------------
    road_speed_limit = vehicle.get_speed_limit()
    overspeed = speed > road_speed_limit

    log("SLA", f"Road Limit = {road_speed_limit:.0f} km/h")

    if overspeed:
        log("SLA", "OVERSPEED")
    else:
        log("SLA", "Within Speed Limit")
    # -------------------------
    # Speed Limit Assist
    # -------------------------

    overspeed = speed > road_speed_limit
    log("VEHICLE", f"Velocity = {velocity}")
    log("SPEED", f"{speed:.2f} km/h")
    # -------------------------
    # Automatic Emergency Braking
    # -------------------------
    adas.aeb_active = False
    aeb_status = "OFF"

    # Activate only when collision is critical
    if (
        warning == "COLLISION WARNING!"
        and speed > 5
        and largest_height > 220
    ):
        adas.aeb_active = True
        aeb_status = "ACTIVE"
    else:
        adas.aeb_active = False
        aeb_status = "OFF"
    # -------------------------
    # Apply Automatic Emergency Braking
    # -------------------------

    if adas.aeb_active:
       log("AEB", f"ACTIVE | Speed = {speed:.2f} km/h")
    
    # -------------------------
    # Adaptive Cruise Control
    # -------------------------

    adas.acc_active = False
    adas.acc_throttle = 0.7

    # Enable ACC only if AEB is NOT active
    if not adas.aeb_active:

        if closest_vehicle is not None:

            # Vehicle is getting closer
            if largest_height > 120:

                adas.acc_active = True

                if largest_height > 200:
                    adas.acc_throttle = 0.20

                elif largest_height > 170:
                    adas.acc_throttle = 0.35

                elif largest_height > 140:
                    adas.acc_throttle = 0.50

                else:
                    adas.acc_throttle = 0.70

    if adas.acc_active:
        log("ACC", f"Throttle = {adas.acc_throttle:.2f}")

    cv2.putText(
        img,
        f"Speed: {speed:.1f} km/h",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        warning,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )
    cv2.putText(
        img,
        lane_status,
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        lane_color,
        3
    )

    # Black outline

    
    if offset is not None:
        text = f"Offset: {offset:+d} px"
    else:
        text = "Offset: N/A"

    # Black outline
    cv2.putText(
        img,
        text,
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        5
    )

    # Yellow text
    cv2.putText(
        img,
        text,
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Black outline
    if correction != "":

    # Black outline
        cv2.putText(
            img,
            correction,
            (20,240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Cyan text
        cv2.putText(
            img,
            correction,
            (20,240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

    cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
    if traffic_state != "UNKNOWN":

        traffic_text = f"Traffic: {traffic_state}"

        # Black outline
        cv2.putText(
            img,
            traffic_text,
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5
        )

        # Colored text
        cv2.putText(
            img,
            traffic_text,
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            traffic_color,
            2
        )
    if stop_sign_detected:

    # Black outline
        cv2.putText(
            img,
            stop_text,
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5
        )

        # Magenta text
        cv2.putText(
            img,
            stop_text,
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

    # Black outline
        cv2.putText(
            img,
            stop_instruction,
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            stop_instruction,
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )
    if left_blind_spot:

        cv2.putText(
            img,
            "LEFT BLIND SPOT",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5
        )

        cv2.putText(
            img,
            "LEFT BLIND SPOT",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    if right_blind_spot:

        cv2.putText(
            img,
            "RIGHT BLIND SPOT",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5
        )

        cv2.putText(
            img,
            "RIGHT BLIND SPOT",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    # -------------------------
    # Rear Collision Warning
    # -------------------------

    if rear_distance == "CAUTION":

        # Black outline
        cv2.putText(
            img,
            "REAR VEHICLE",
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Yellow text
        cv2.putText(
            img,
            "REAR VEHICLE",
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    elif rear_distance == "DANGER":

        # Black outline
        cv2.putText(
            img,
            "REAR COLLISION WARNING",
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "REAR COLLISION WARNING",
            (20,360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )
    # -------------------------
    # Pedestrian Warning
    # -------------------------

    if pedestrian_warning == "PEDESTRIAN AHEAD":

        # Black outline
        cv2.putText(
            img,
            "PEDESTRIAN AHEAD",
            (20,400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Yellow text
        cv2.putText(
            img,
            "PEDESTRIAN AHEAD",
            (20,400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    elif pedestrian_warning == "PEDESTRIAN WARNING":

        # Black outline
        cv2.putText(
            img,
            "PEDESTRIAN WARNING",
            (20,400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "PEDESTRIAN WARNING",
            (20,400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    # -------------------------
    # AI Risk Heatmap
    # -------------------------

    # Reset every frame
    front_risk = "GREEN"
    left_risk = "GREEN"
    right_risk = "GREEN"
    rear_risk = "GREEN"

    # ---------- Front ----------
    if warning == "CAUTION":
        front_risk = "YELLOW"

    if warning == "COLLISION WARNING!":
        front_risk = "RED"

    if pedestrian_warning == "PEDESTRIAN AHEAD":
        front_risk = "YELLOW"

    if pedestrian_warning == "PEDESTRIAN WARNING":
        front_risk = "RED"

    # ---------- Left ----------
    if left_blind_spot:
        left_risk = "RED"

    # ---------- Right ----------
    if right_blind_spot:
        right_risk = "RED"

    # ---------- Rear ----------
    if rear_distance == "CAUTION":
        rear_risk = "YELLOW"

    if rear_distance == "DANGER":
        rear_risk = "RED"


    # -------------------------
    # AI Risk Score
    # -------------------------

    risk_score = 0

    # Front Collision
    if warning == "CAUTION":
        risk_score += 20

    elif warning == "COLLISION WARNING!":
        risk_score += 40

    # Pedestrian
    if pedestrian_warning == "PEDESTRIAN AHEAD":
        risk_score += 20

    elif pedestrian_warning == "PEDESTRIAN WARNING":
        risk_score += 40

    # Blind Spot
    if left_blind_spot:
        risk_score += 15

    if right_blind_spot:
        risk_score += 15

    # Rear Collision
    if rear_distance == "CAUTION":
        risk_score += 15

    elif rear_distance == "DANGER":
        risk_score += 30

    # Lane Departure
    if lane_status in ["DRIFTING LEFT", "DRIFTING RIGHT"]:
        risk_score += 10

    elif lane_status in ["LANE DEPARTURE LEFT", "LANE DEPARTURE RIGHT"]:
        risk_score += 20

    # Stop Sign
    if stop_sign_detected:
        risk_score += 10

    # Limit to 100
    risk_score = min(risk_score, 100)

    # =====================================================
    # Driver Fatigue Contribution
    # =====================================================

    risk_score += int(adas.fatigue_score * 0.5)

    risk_score = min(risk_score, 100)
    # =====================================================
    # Driver Fatigue → Vehicle Control
    # =====================================================

    if adas.fatigue_level == "LOW":

        adas.fatigue_throttle = 0.70
        adas.fatigue_brake = 0.00

    elif adas.fatigue_level == "MEDIUM":

        adas.fatigue_throttle = 0.60
        adas.fatigue_brake = 0.00

    elif adas.fatigue_level == "HIGH":

        adas.fatigue_throttle = 0.40
        adas.fatigue_brake = 0.00

    else:   # CRITICAL

        adas.fatigue_throttle = 0.00
        adas.fatigue_brake = 0.25
    # Create overlay for transparent panel
    overlay = img.copy()
    # =====================================================
    # PROFESSIONAL AI RISK HEATMAP
    # =====================================================

    panel_w = 200
    panel_h = 240

    panel_x = img.shape[1] - 220
    panel_y = 15

    cv2.rectangle(
        overlay,
        (panel_x+5,panel_y+5),
        (panel_x+panel_w+5,panel_y+panel_h+5),
        (10,10,10),
        -1
    )

    # Dark background
    radius = 15

    # Center
    cv2.rectangle(
        overlay,
        (panel_x+radius, panel_y),
        (panel_x+panel_w-radius, panel_y+panel_h),
        (25,25,25),
        -1
    )

    cv2.rectangle(
        overlay,
        (panel_x, panel_y+radius),
        (panel_x+panel_w, panel_y+panel_h-radius),
        (25,25,25),
        -1
    )

    # Corners
    cv2.circle(overlay,(panel_x+radius,panel_y+radius),radius,(25,25,25),-1)
    cv2.circle(overlay,(panel_x+panel_w-radius,panel_y+radius),radius,(25,25,25),-1)
    cv2.circle(overlay,(panel_x+radius,panel_y+panel_h-radius),radius,(25,25,25),-1)
    cv2.circle(overlay,(panel_x+panel_w-radius,panel_y+panel_h-radius),radius,(25,25,25),-1)
        # Softer blue border
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (90,210,255),
        2
    )

    # Title
    cv2.rectangle(
        overlay,
        (panel_x,panel_y),
        (panel_x+panel_w,panel_y+38),
        (45,45,45),
        -1
    )
    cv2.putText(
        overlay,
        "AI RISK",
        (panel_x+56,panel_y+30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255,255,255),
        2
    )
    cv2.line(
        overlay,
        (panel_x+10,panel_y+42),
        (panel_x+panel_w-10,panel_y+42),
        (80,80,80),
        1
    )

    # -------------------------
    # AI Risk Score
    # -------------------------
    if risk_score < 30:
        score_color = (0,255,0)

    elif risk_score < 60:
        score_color = (0,255,255)

    elif risk_score < 80:
        score_color = (0,165,255)

    else:
        score_color = (0,0,255)
    cv2.putText(
        overlay,
        f"Risk Score : {risk_score}%",
        (panel_x + 20, panel_y + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        score_color,
        2
    )
    # -------------------------
    # Car Body
    # -------------------------

    car_x = panel_x + 82
    car_y = panel_y + 95

    # Car body
    cv2.rectangle(
        overlay,
        (car_x, car_y),
        (car_x + 36, car_y + 80),
        (170,170,170),
        -1
    )

    # Car outline
    cv2.rectangle(
        overlay,
        (car_x, car_y),
        (car_x + 36, car_y + 80),
        (255,255,255),
        2
    )

    # Roof
    cv2.rectangle(
        overlay,
        (car_x+6, car_y+18),
        (car_x+30, car_y+62),
        (120,120,120),
        -1
    )
    
    # FRONT
    cv2.circle(
        overlay,
        (car_x+18, car_y-14),
        10,
        risk_color(front_risk),
        -1
    )

    # LEFT
    cv2.circle(
        overlay,
        (car_x-20, car_y+40),
        10,
        risk_color(left_risk),
        -1
    )

    # RIGHT
    cv2.circle(
        overlay,
        (car_x+56, car_y+40),
        10,
        risk_color(right_risk),
        -1
    )

    # REAR
    cv2.circle(
        overlay,
        (car_x+18, car_y+100),
        10,
        risk_color(rear_risk),
        -1
    )

    cv2.putText(overlay,"FRONT",(car_x-30, car_y-8),
            cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    cv2.putText(overlay,"LEFT",(car_x-62,car_y+45),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    cv2.putText(overlay,"RIGHT",(car_x+67,car_y+45),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    cv2.putText(overlay,"REAR",(car_x-1, car_y+123),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)


    # =====================================================
    # DRIVER MONITORING PANEL
    # =====================================================

    dms_x = panel_x
    dms_y = panel_y + panel_h + 15

    dms_w = panel_w
    dms_h = 155

    # Shadow
    cv2.rectangle(
        overlay,
        (dms_x + 5, dms_y + 5),
        (dms_x + dms_w + 5, dms_y + dms_h + 5),
        (10, 10, 10),
        -1
    )

    # Background
    cv2.rectangle(
        overlay,
        (dms_x, dms_y),
        (dms_x + dms_w, dms_y + dms_h),
        (25, 25, 25),
        -1
    )

    # Border
    cv2.rectangle(
        overlay,
        (dms_x, dms_y),
        (dms_x + dms_w, dms_y + dms_h),
        (90, 210, 255),
        2
    )

    # Header
    cv2.rectangle(
        overlay,
        (dms_x, dms_y),
        (dms_x + dms_w, dms_y + 35),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        overlay,
        "DRIVER MONITOR",
        (dms_x + 18, dms_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255,255,255),
        2
    )            

    # =====================================================
    # Driver Monitoring Information
    # =====================================================

    # Driver Status Color
    if adas.driver_status == "ALERT":
        status_color = (0, 255, 0)

    elif adas.driver_status == "Eyes Closed":
        status_color = (0, 255, 255)

    else:
        status_color = (0, 0, 255)


    # Fatigue Risk Color
    if adas.fatigue_level == "LOW":
        fatigue_color = (0,255,0)

    elif adas.fatigue_level == "MEDIUM":
        fatigue_color = (0,255,255)

    elif adas.fatigue_level == "HIGH":
        fatigue_color = (0,165,255)

    else:
        fatigue_color = (0,0,255)


    # Driver Status
    cv2.putText(
        overlay,
        f"Status : {adas.driver_status}",
        (dms_x + 12, dms_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        status_color,
        2
    )

    # Yawning
    cv2.putText(
        overlay,
        f"Yawning : {'YES' if adas.yawning else 'NO'}",
        (dms_x + 12, dms_y + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255,255,255),
        2
    )

    # Fatigue Score
    cv2.putText(
        overlay,
        f"Fatigue : {adas.fatigue_score}%",
        (dms_x + 12, dms_y + 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        fatigue_color,
        2
    )

    # Risk Level
    cv2.putText(
        overlay,
        f"Risk : {adas.fatigue_level}",
        (dms_x + 12, dms_y + 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        fatigue_color,
        2
    )

    cv2.addWeighted(
        overlay,
        0.72,      # Panel opacity
        img,
        0.18,
        0,
        img
    )  
 
    # -------------------------
    # Automatic Emergency Braking HUD
    # -------------------------

    if adas.aeb_active:

        # Black outline
        cv2.putText(
            img,
            "AEB ACTIVE",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "AEB ACTIVE",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
    # -------------------------
    # Lane Keeping Assist HUD
    # -------------------------

    if adas.lka_active:

        # Black outline
        cv2.putText(
            img,
            "LKA ACTIVE",
            (20, 510),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5
        )

        # Green text
        cv2.putText(
            img,
            "LKA ACTIVE",
            (20, 510),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    
    # -------------------------
    # Adaptive Cruise Control HUD
    # -------------------------

    if adas.acc_active:

        # Black outline
        cv2.putText(
            img,
            "ACC ACTIVE",
            (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            5
        )

        # Blue text
        cv2.putText(
            img,
            "ACC ACTIVE",
            (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 150, 0),
            2
        )
    # ======================================
    # SPEED LIMIT ASSIST (Bottom of HUD)
    # ======================================

    y = 600

    # Road Speed Limit
    cv2.putText(
        img,
        f"Road Limit : {road_speed_limit:.0f} km/h",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    flash = int(time.time() * 2) % 2

    if overspeed:

        if flash:

            cv2.putText(
                img,
                "OVERSPEED WARNING",
                (20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,0,255),
                3
            )

    else:

        cv2.putText(
            img,
            "WITHIN SPEED LIMIT",
            (20, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )
    
    # End of Speed Limit Section


    # =====================================================
    # SAFE STOP ASSIST WARNING
    # =====================================================

    if adas.fatigue_level in ["HIGH", "CRITICAL"]:

        cv2.rectangle(
            img,
            (180,20),
            (1100,95),
            (0,0,255),
            -1
        )

        cv2.putText(
            img,
            "DRIVER DROWSY - SAFE STOP ASSIST ACTIVE",
            (205,68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255,255,255),
            3
        )


    cv2.imshow("Front Camera", img)

    cv2.waitKey(1)

def process_left(image):

    global left_blind_spot

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((720, 1280, 4))
    img = img[:, :, :3].copy()

    left_blind_spot = False

    # -------------------------
    # YOLO Detection
    # -------------------------
    results = yolo_model(
        img,
        conf=0.5,
        classes=[2, 3, 5, 7]
    )

    boxes = results[0].boxes
    
    # -------------------------
    # Check for Vehicles
    # -------------------------
    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        height = y2 - y1

        if height > 120:

            left_blind_spot = True

            log("BSM", "LEFT Blind Spot")

            break
    if left_blind_spot:

    # Black outline
        cv2.putText(
            img,
            "LEFT BLIND SPOT",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "LEFT BLIND SPOT",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )
    cv2.imshow("Left Camera", img)

    cv2.waitKey(1)

def process_right(image):

    global right_blind_spot

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((720, 1280, 4))
    img = img[:, :, :3].copy()

    right_blind_spot = False

    results = yolo_model(
        img,
        conf=0.5,
        classes=[2,3,5,7]
    )

    boxes = results[0].boxes

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        height = y2 - y1

        if height > 120:

            right_blind_spot = True

            log("BSM", "RIGHT Blind Spot")

            break
    if right_blind_spot:

    # Black outline
        cv2.putText(
            img,
            "RIGHT BLIND SPOT",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "RIGHT BLIND SPOT",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )
    cv2.imshow("Right Camera", img)

    cv2.waitKey(1)

def process_rear(image):

    global rear_collision
    global rear_distance

    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((720, 1280, 4))
    img = img[:, :, :3].copy()

    rear_collision = False
    rear_distance = "SAFE"

    # -------------------------
    # YOLO Rear Detection
    # -------------------------
    results = yolo_model(
        img,
        conf=0.5,
        classes=[2, 3, 5, 7]
    )

    boxes = results[0].boxes

    # -------------------------
    # Find Closest Rear Vehicle
    # -------------------------
    largest_height = 0

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        height = y2 - y1

        if height > largest_height:

            largest_height = height
    # -------------------------
    # Rear Collision Warning
    # -------------------------

    if largest_height > 220:

        rear_collision = True
        rear_distance = "DANGER"

    elif largest_height > 150:

        rear_collision = True
        rear_distance = "CAUTION"

    else:

        rear_collision = False
        rear_distance = "SAFE"

    log("RCW", rear_distance)

    # -------------------------
    # Display Rear Warning
    # -------------------------

    if rear_distance == "CAUTION":

        # Black outline
        cv2.putText(
            img,
            "REAR VEHICLE",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            5
        )

        # Yellow text
        cv2.putText(
            img,
            "REAR VEHICLE",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

    elif rear_distance == "DANGER":

        # Black outline
        cv2.putText(
            img,
            "REAR COLLISION WARNING",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            5
        )

        # Red text
        cv2.putText(
            img,
            "REAR COLLISION WARNING",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Rear Camera", img)

    cv2.waitKey(1)
# -------------------------
# Start Camera
# -------------------------
camera.listen(process_image)

left_camera.listen(process_left)

right_camera.listen(process_right)

rear_camera.listen(process_rear)
# -------------------------
# Keyboard Control
# -------------------------
threading.Thread(
    target=control_vehicle,
    args=(vehicle,),
    daemon=True
).start()

threading.Thread(
    target=start_driver_monitor,
    daemon=True
).start()

log("SYSTEM", "Camera Started")
# -------------------------
# Main Loop
# -------------------------
try:
    while True:
        time.sleep(0.01)

except KeyboardInterrupt:
    pass

# -------------------------
# Cleanup
# -------------------------
camera.stop()
left_camera.stop()
right_camera.stop()
rear_camera.stop()

camera.destroy()
left_camera.destroy()
right_camera.destroy()
rear_camera.destroy()

vehicle.destroy()