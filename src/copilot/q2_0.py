import json
import time
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 7
diecutter_id = 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Initialize the list to store the average speed values
average_speeds = []

# Step 2: Collect data over 10 consecutive, non-overlapping time windows of 10 seconds
for window in range(10):
    speeds = []
    start_time = time.time()
    while time.time() - start_time < 10:  # Collect data for 10 seconds
        current_data = GetCurrentDataFromChip.call(chip_id=diecutter_id)
        speeds.append(current_data["speed"])
        time.sleep(1)  # Assuming the refresh rate is 1 second
    # Calculate the average speed for this window
    average_speed = sum(speeds) / len(speeds)
    average_speeds.append({"time_window": window + 1, "average_speed": average_speed})

# Step 3: Write the list of dictionaries to a JSON file
with open("average_speeds_session_{}.json".format(session_id), "w") as json_file:
    json.dump(average_speeds, json_file, indent=4)

print("JSON file with average speeds for session {} has been generated.".format(session_id))