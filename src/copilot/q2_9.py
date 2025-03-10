import time
import json
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 7
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=7)

# Initialize variables for data collection
intervals = 10
interval_duration = 10  # seconds
speed_data = [[] for _ in range(intervals)]  # List of lists to hold speed values for each interval

# Step 2 & 3: Collect speed data over 100 seconds
for interval in range(intervals):
    for _ in range(interval_duration):
        current_data = GetCurrentDataFromChip.call(chip_id=7)  # Assuming chip_id and diecutter_id are the same
        speed_data[interval].append(current_data["speed"])
        time.sleep(1)  # Wait for 1 second to simulate real-time data collection

# Step 4: Calculate mean speed for each interval and prepare the table
table_data = []
for i, speeds in enumerate(speed_data):
    mean_speed = sum(speeds) / len(speeds)
    table_data.append({"interval": i+1, "mean_speed": mean_speed})

# Step 5: Write the table data to a JSON file
with open("mean_speed_data.json", "w") as json_file:
    json.dump(table_data, json_file)

print("Mean speed data for the ongoing session of diecutter 7 has been saved to 'mean_speed_data.json'")