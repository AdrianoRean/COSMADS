import time
import json
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the chip id for diecutter id 25
diecutter_id = 25
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Step 2: Collect data over a 30-second interval
start_time = time.time()
highest_speed = 0
while time.time() - start_time < 30:
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    highest_speed = max(highest_speed, current_data["speed"])
    time.sleep(1)  # Wait for a second before the next data collection

# Step 3: Prepare the data for the JSON file
data_to_save = [{
    "diecutter_id": diecutter_id,
    "highest_speed": highest_speed
}]

# Step 4: Save the data in a JSON file
with open('highest_speed_data.json', 'w') as json_file:
    json.dump(data_to_save, json_file)

print("Data saved to highest_speed_data.json")