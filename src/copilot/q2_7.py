import numpy as np
import time
import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=7)

# Step 2: Get the chip ID for diecutter 7
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=7)

# Initialize a list to store the mean velocities
mean_velocities = []

# Step 3 & 4: Compute the mean velocity for each of the ten intervals
for interval in range(10):
    velocities = []
    start_time = time.time()
    while time.time() - start_time < 10:  # Collect data for 10 seconds
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        velocities.append(current_data["speed"])
        time.sleep(1)  # Assuming the refresh rate is 1 second
    mean_velocity = np.mean(velocities)
    mean_velocities.append({"interval": interval + 1, "mean_velocity": mean_velocity})

# Step 5 & 6: Write the list of dictionaries to a JSON file
with open('mean_velocities_session_{}.json'.format(session_id), 'w') as f:
    json.dump(mean_velocities, f)

print(f"Mean velocities for session {session_id} have been saved to 'mean_velocities_session_{session_id}.json'")