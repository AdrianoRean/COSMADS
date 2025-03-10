import json
import time
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the chip id for diecutter id 25
diecutter_id = 25
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Step 2: Collect data for 30 seconds
data_collection_period = 30  # seconds
start_time = time.time()
highest_speed = 0
while time.time() - start_time < data_collection_period:
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    highest_speed = max(highest_speed, current_data["speed"])
    time.sleep(1)  # Wait for 1 second before the next data collection

# Step 3: Prepare the data for the JSON file
table_data = [{
    "diecutter_id": diecutter_id,
    "chip_id": chip_id,
    "highest_speed": highest_speed
}]

# Step 4: Write the data to a JSON file
with open("highest_speed_data.json", "w") as json_file:
    json.dump(table_data, json_file)

print("Data written to highest_speed_data.json")