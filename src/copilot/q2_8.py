import json
import numpy as np
import time
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Assuming the chip ID and diecutter ID are the same for simplicity
diecutter_id = 7
chip_id = diecutter_id

# Get the current session ID for diecutter 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Initialize variables for calculating mean speeds
interval_durations = 10  # seconds
intervals = 10
mean_speeds = []

# Simulate data collection for each interval
for _ in range(intervals):
    speeds = []
    start_time = time.time()
    while time.time() - start_time < interval_durations:
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        speeds.append(current_data["speed"])
        time.sleep(1)  # Wait for a second before the next data point to simulate real-time data collection
    
    # Calculate mean speed for the current interval
    mean_speed = np.mean(speeds)
    mean_speeds.append({"session_id": session_id, "mean_speed": mean_speed})

# Save the mean speeds to a JSON file
with open('mean_speeds_session_{}.json'.format(session_id), 'w') as file:
    json.dump(mean_speeds, file)

print(f"Mean speeds for session {session_id} saved to 'mean_speeds_session_{session_id}.json'")