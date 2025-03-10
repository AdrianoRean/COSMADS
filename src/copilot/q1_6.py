import time
import json
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

def generate_max_speed_table(diecutter_id, duration):
    # Get the chip id for the given diecutter id
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    start_time = time.time()
    max_speed = 0
    
    # Loop for the specified duration
    while time.time() - start_time < duration:
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        max_speed = max(max_speed, current_data["speed"])
        time.sleep(1)  # Assuming we want to check every second
    
    # Prepare the data for the JSON file
    data = [{
        "diecutter_id": diecutter_id,
        "chip_id": chip_id,
        "max_speed": max_speed
    }]
    
    # Write the data to a JSON file
    with open("max_speed_data.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
generate_max_speed_table(diecutter_id=25, duration=30)