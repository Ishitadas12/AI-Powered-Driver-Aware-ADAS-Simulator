import carla
import random

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Choose a Tesla Model 3
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

# Choose a random spawn point
spawn_points = world.get_map().get_spawn_points()

spawn_point = random.choice(spawn_points)

# Spawn vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is not None:
    print("✅ Tesla Model 3 spawned successfully!")
    print("Vehicle ID:", vehicle.id)
else:
    print("❌ Failed to spawn vehicle.")