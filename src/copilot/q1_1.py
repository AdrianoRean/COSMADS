import time
import json
from diecutter import GetCurrentDataFromChip

# Initialize variables
diecutter_id = 25  # The ID of the diecutter we're interested in
interval_duration = 30  # Duration of the interval in seconds
highest_speed = 0  # To keep track of the highest speed

# Simulate data collection over the interval
for _ in range(interval_duration):
    current_data = GetCurrentDataFromChip.call(chip_id=diecutter_id)
    current_speed = current_data["speed"]
    if current_speed > highest_speed:
        highest_speed = current_speed
    time.sleep(1)  # Wait for 1 second before the next data retrieval

# Prepare the data for the JSON file
data_to_save = [{"diecutter_id": diecutter_id, "highest_speed": highest_speed}]

# Write the data to a JSON file
with open("diecutter_speed_data.json", "w") as json_file:
    json.dump(data_to_save, json_file, indent=4)

print("Data has been saved to diecutter_speed_data.json")