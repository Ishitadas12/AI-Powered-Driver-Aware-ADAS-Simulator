import pygame
import carla
import carla_modules.adas_state as adas
from datetime import datetime

def log(level, message):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] [{level}] {message}")

def control_vehicle(vehicle):
    """
    Reads keyboard input and applies it to the CARLA vehicle.
    W = Accelerate
    S = Brake
    A = Left
    D = Right
    ESC = Exit
    """

    pygame.init()

    # Small hidden window so pygame can receive keyboard events
    pygame.display.set_mode((300, 100))
    pygame.display.set_caption("ADAS Keyboard Control")

    clock = pygame.time.Clock()

    print("\n========== Controls ==========")
    print("W : Accelerate")
    print("S : Brake")
    print("A : Turn Left")
    print("D : Turn Right")
    print("ESC : Quit")
    print("==============================\n")

    running = True

    while running:

        control = carla.VehicleControl()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Exit
        if keys[pygame.K_ESCAPE]:
            running = False

        # -------------------------
        # Manual Controls
        # -------------------------

        # -------------------------
        # Throttle / ACC
        # -------------------------
        if keys[pygame.K_w]:

            # Base throttle
            if adas.acc_active:
                throttle = adas.acc_throttle
            else:
                throttle = 0.7

            # Driver fatigue limits ALL acceleration
            control.throttle = min(
                throttle,
                adas.fatigue_throttle
            )

            log(
                "DRIVER AI",
                f"Applied Throttle = {control.throttle:.2f}"
            )

        if keys[pygame.K_s]:
            control.brake = 1.0

        if keys[pygame.K_a]:
            control.steer = -0.5

        if keys[pygame.K_d]:
            control.steer = 0.5

        # -------------------------
        # Automatic Emergency Braking
        # -------------------------
        log("AEB", f"State = {adas.aeb_active}")
        if adas.aeb_active:

            control.throttle = 0.0
            control.brake = 1.0

            log("AEB", "BRAKING")

        # -------------------------
        # Driver Fatigue Safe Stop
        # -------------------------

        elif adas.fatigue_brake > 0:

            control.throttle = 0.0
            control.brake = adas.fatigue_brake

            log(
                "SAFE STOP",
                f"Driver Fatigue -> Brake={adas.fatigue_brake:.2f}"
            )
        # -------------------------
        # Lane Keeping Assist
        # -------------------------

        # -------------------------
        # Lane Keeping Assist
        # -------------------------

        if (
            adas.lka_active
            and not keys[pygame.K_a]
            and not keys[pygame.K_d]
        ):

            control.steer = adas.lka_steer

            log("LKA", f"Steering = {adas.lka_steer:.2f}")
            
        log("ACC", f"Throttle = {control.throttle:.2f}")
        vehicle.apply_control(control)

        clock.tick(30)

    pygame.quit()