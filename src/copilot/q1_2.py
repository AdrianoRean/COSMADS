import json
from diecutter import GetCurrentDataFromChip
from database import GetDiecutterChipIdFromDiecutterId

# Get the chip ID for the diecutter ID 25
diecutter_id = 25
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

# Initialize variables to track the highest speed
highest_speed = 0

# Simulate fetching data for 30 seconds
for _ in range(30):
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    current_speed = current_data["speed"]
    if current_speed > highest_speed:
        highest_speed = current_speed

# Prepare the data for the JSON file
data_to_save = [{
    "diecutter_id": diecutter_id,
    "highest_speed": highest_speed
}]

# Write the data to a JSON file
with open('highest_speed_data.json', 'w') as json_file:
    json.dump(data_to_save, json_file)

print("Data saved to highest_speed_data.json")