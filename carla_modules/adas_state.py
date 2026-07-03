# -------------------------
# ADAS Shared State
# -------------------------

# Automatic Emergency Braking
aeb_active = False

# Lane Keeping Assist
lka_active = False
lka_steer = 0.0

# -------------------------
# Adaptive Cruise Control
# -------------------------
acc_active = False
acc_throttle = 0.7

# -------------------------
# Speed Limit Recognition
# -------------------------

speed_limit = None          # Current detected speed limit
overspeed_warning = False   # True if vehicle exceeds limit

# =====================================================
# Driver Monitoring Shared State
# =====================================================

driver_status = "ALERT"

yawning = False

fatigue_score = 0

fatigue_level = "LOW"

# -------------------------
# Driver Fatigue ACC Control
# -------------------------

fatigue_throttle = 0.7
fatigue_brake = 0.0