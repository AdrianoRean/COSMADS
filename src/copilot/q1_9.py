import time
import json
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the chip ID for diecutter ID 25
diecutter_id = 25
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Step 2: Collect data over 30 seconds
end_time = time.time() + 30
speeds = []

while time.time() < end_time:
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    speeds.append(current_data["speed"])
    time.sleep(1)  # Assuming we collect data once every second

# Step 3: Find the maximum speed
max_speed = max(speeds)

# Step 4: Create a list of dictionaries for the table
table_data = [{"diecutter_id": diecutter_id, "maximum_speed": max_speed}]

# Step 5: Write the table data to a JSON file
with open("max_speed_data.json", "w") as json_file:
    json.dump(table_data, json_file)

print("Data written to max_speed_data.json")