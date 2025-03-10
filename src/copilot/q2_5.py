import time
import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=7)

# Step 2: Get the chip ID associated with diecutter 7
# Assuming a function or method to get the chip ID exists. This is a placeholder.
# Replace `GetDiecutterChipIdFromDiecutterId.call` with the actual method if different.
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=7)

# Step 3: Collect data at 10-second intervals, 10 times
data_points = []
for _ in range(10):
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    data_points.append(current_data)
    time.sleep(10)  # Wait for 10 seconds before the next data collection

# Step 4: Calculate mean speeds and store in a list of dictionaries
mean_speeds = [{"interval": i+1, "mean_speed": data_point["speed"]} for i, data_point in enumerate(data_points)]

# Step 5: Write the list of dictionaries to a JSON file
with open('mean_speeds_session_{}.json'.format(session_id), 'w') as f:
    json.dump(mean_speeds, f)

print(f"Data written to mean_speeds_session_{session_id}.json")