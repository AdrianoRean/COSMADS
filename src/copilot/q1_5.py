import numpy as np
import time
import json
from diecutter import GetCurrentDataFromChip

# Initialize variables
diecutter_id = 25
interval_duration = 30  # seconds
start_time = time.time()
max_speed = 0
data_collection = []

# Simulate the process of collecting data over the specified interval
while time.time() - start_time < interval_duration:
    # Assuming the chip ID is the same as the diecutter ID for simplicity
    current_data = GetCurrentDataFromChip.call(chip_id=diecutter_id)
    current_speed = current_data["speed"]
    if current_speed > max_speed:
        max_speed = current_speed
    data_collection.append({
        "diecutter_id": diecutter_id,
        "speed": current_speed,
        "temperature": current_data["temperature"],
        "rotations": current_data["rotations"],
        "timestamp": time.time()
    })
    time.sleep(1)  # Wait for a second before collecting the next data point

# Filter the data to only include the maximum speed
max_speed_data = [data for data in data_collection if data["speed"] == max_speed]

# Save the filtered data to a JSON file
with open("max_speed_data.json", "w") as json_file:
    json.dump(max_speed_data, json_file, indent=4)

print("Data collection complete. The maximum speed data has been saved to max_speed_data.json.")