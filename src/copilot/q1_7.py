import time
import json
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the chip ID for diecutter ID 25
diecutter_id = 25
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Step 2: Collect data over a period of 30 seconds
start_time = time.time()
data_collection = []
while time.time() - start_time < 30:
    data = GetCurrentDataFromChip.call(chip_id=chip_id)
    data_collection.append(data)
    time.sleep(1)  # Assuming a 1-second interval between data collections

# Step 3: Find the highest speed
highest_speed = max(data['speed'] for data in data_collection)

# Step 4: Save the result in a JSON file
result = [{"diecutter_id": diecutter_id, "highest_speed": highest_speed}]
with open('result.json', 'w') as f:
    json.dump(result, f)

print("Data saved to result.json")