import time
import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=7)

# Step 2: Get the chip ID for diecutter 7
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=7)

# Initialize the list to store the results
results = []

# Step 3 & 4: Calculate the average speed for each 10-second interval
for interval in range(10):
    speeds = []
    start_time = time.time()
    while time.time() - start_time < 10:  # Collect data for 10 seconds
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        speeds.append(current_data["speed"])
        time.sleep(1)  # Assuming the refresh rate is 1 second
    average_speed = sum(speeds) / len(speeds)
    results.append({"interval": interval + 1, "average_speed": average_speed})

# Step 5: Write the results to a JSON file
with open('average_speeds.json', 'w') as f:
    json.dump(results, f)

print("Data written to average_speeds.json")