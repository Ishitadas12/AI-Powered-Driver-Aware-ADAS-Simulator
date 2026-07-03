import cv2
import mediapipe as mp
import math
import time
import winsound
import pygame
import carla_modules.adas_state as adas

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("sounds/driver_alert.mp3")
# =====================================================
# MediaPipe Face Mesh
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
MOUTH = [
    13,   # Upper lip
    14,   # Lower lip
    78,   # Left mouth corner
    308,  # Right mouth corner
    82,
    87,
    312,
    317
]
# =====================================================
# Eye Landmark Indices
# =====================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =====================================================
# Distance Function
# =====================================================

def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


# =====================================================
# Eye Aspect Ratio (EAR)
# =====================================================

def eye_aspect_ratio(points):

    A = distance(points[1], points[5])
    B = distance(points[2], points[4])
    C = distance(points[0], points[3])

    if C == 0:
        return 0

    ear = (A + B) / (2.0 * C)

    return ear

# =====================================================
# Driver Monitoring Parameters
# =====================================================

EAR_THRESHOLD = 0.88

eyes_closed_start = None

driver_status = "ALERT"

# =====================================================
# Mouth Aspect Ratio (MAR)
# =====================================================

def mouth_aspect_ratio(points):

    vertical = distance(points[0], points[1])

    horizontal = distance(points[2], points[3])

    if horizontal == 0:
        return 0

    mar = vertical / horizontal

    return mar

# =====================================================
# Yawning Detection
# =====================================================

MAR_THRESHOLD = 0.75

mouth_open_start = None

yawning = False
# =====================================================
# AI Driver Fatigue Score
# =====================================================

fatigue_score = 0

fatigue_level = "LOW"

# =====================================================
# Webcam
# =====================================================
def start_driver_monitor():
    global eyes_closed_start
    global mouth_open_start
    global driver_status
    global yawning
    global fatigue_score
    global fatigue_level

    alarm_played = False

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror Image
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face in results.multi_face_landmarks:

                h, w, _ = frame.shape

                mouth_points = []
                left_eye_points = []
                right_eye_points = []

                # ------------------------------------------
                # Collect Eye Landmarks
                # ------------------------------------------

                for idx, landmark in enumerate(face.landmark):

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    if idx in LEFT_EYE:
                        left_eye_points.append((x, y))
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    if idx in RIGHT_EYE:
                        right_eye_points.append((x, y))
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    if idx in MOUTH:

                        mouth_points.append((x, y))

                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (255,0,0),
                            -1
                        )
                # ------------------------------------------
                # Calculate EAR
                # ------------------------------------------

                if len(left_eye_points) == 6 and len(right_eye_points) == 6:

                    left_ear = eye_aspect_ratio(left_eye_points)
                    right_ear = eye_aspect_ratio(right_eye_points)

                    ear = (left_ear + right_ear) / 2

                    # -------------------------
                    # Driver Status
                    # -------------------------

                    if ear < EAR_THRESHOLD:

                        if eyes_closed_start is None:
                            eyes_closed_start = time.time()

                        closed_time = time.time() - eyes_closed_start

                        if closed_time > 2:

                            driver_status = "DRIVER DROWSY"

                            if not alarm_played:

                                alarm_sound.play()

                                alarm_played = True

                        else:

                            driver_status = "Eyes Closed"

                    else:

                        eyes_closed_start = None
                        driver_status = "ALERT"

                        alarm_sound.stop()

                        alarm_played = False
                    # =====================================================
                    # Driver Status Display
                    # =====================================================

                    # Black outline
                    cv2.putText(
                        frame,
                        f"Driver : {driver_status}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,0),
                        5
                    )

                    # Color

                    if driver_status == "ALERT":
                        color = (0,255,0)

                    elif driver_status == "Eyes Closed":
                        color = (0,255,255)

                    else:
                        color = (0,0,255)

                    cv2.putText(
                        frame,
                        f"Driver : {driver_status}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )

                    # EAR
                    cv2.putText(
                        frame,
                        f"EAR : {ear:.2f}",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,0,255),
                        2
                    )
                    # ------------------------------------------
                    # Calculate MAR
                    # ------------------------------------------

                    if len(mouth_points) >= 4:

                        mar = mouth_aspect_ratio(mouth_points)

                        # --------------------------------------
                        # Yawning Detection
                        # --------------------------------------

                        if mar > MAR_THRESHOLD:

                            if mouth_open_start is None:
                                mouth_open_start = time.time()

                            mouth_open_time = time.time() - mouth_open_start

                            if mouth_open_time > 1.5:

                                yawning = True

                            else:

                                yawning = False

                        else:

                            mouth_open_start = None
                            yawning = False
                    # =====================================================
                    # AI Driver Fatigue Score
                    # =====================================================

                    fatigue_score = 0

                    # Eyes Closed
                    if driver_status == "Eyes Closed":
                        fatigue_score += 40

                    # Driver Drowsy
                    elif driver_status == "DRIVER DROWSY":
                        fatigue_score += 80

                    # Yawning
                    if yawning:
                        fatigue_score += 20

                    # Maximum Score = 100
                    fatigue_score = min(fatigue_score, 100)

                    # Fatigue Level
                    if fatigue_score <= 20:
                        fatigue_level = "LOW"

                    elif fatigue_score <= 50:
                        fatigue_level = "MEDIUM"

                    elif fatigue_score <= 80:
                        fatigue_level = "HIGH"

                    else:
                        fatigue_level = "CRITICAL"
                    
                    # =====================================================
                    # Send Driver Monitoring Data to CARLA
                    # =====================================================

                    adas.driver_status = driver_status
                    adas.yawning = yawning
                    adas.fatigue_score = fatigue_score
                    adas.fatigue_level = fatigue_level

                    # --------------------------------------
                    # Display Yawning Status
                    # --------------------------------------

                    if yawning:

                        cv2.putText(
                            frame,
                            "YAWNING",
                            (20,160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3
                        )

                    else:

                        cv2.putText(
                            frame,
                            "NORMAL",
                            (20,160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2
                        )

                    cv2.putText(
                        frame,
                        f"MAR : {mar:.2f}",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255,0,0),
                        2
                    )
                    # =====================================================
                    # AI Driver Fatigue Display
                    # =====================================================

                    # Choose color based on fatigue level
                    if fatigue_level == "LOW":
                        fatigue_color = (0, 255, 0)

                    elif fatigue_level == "MEDIUM":
                        fatigue_color = (0, 255, 255)

                    elif fatigue_level == "HIGH":
                        fatigue_color = (0, 165, 255)

                    else:
                        fatigue_color = (0, 0, 255)

                    # Black outline
                    cv2.putText(
                        frame,
                        f"Fatigue : {fatigue_score}%",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        4
                    )

                    # Colored text
                    cv2.putText(
                        frame,
                        f"Fatigue : {fatigue_score}%",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        fatigue_color,
                        2
                    )

                    # Black outline
                    cv2.putText(
                        frame,
                        f"Risk : {fatigue_level}",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        4
                    )

                    # Colored text
                    cv2.putText(
                        frame,
                        f"Risk : {fatigue_level}",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        fatigue_color,
                        2
                    )
        # =====================================================
        # Show Window
        # =====================================================

        cv2.imshow("Driver Monitoring", frame)

        # ESC to Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_driver_monitor()