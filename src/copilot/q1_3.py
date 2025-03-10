import json
import time
from diecutter import GetCurrentDataFromChip
from database import GetDiecutterChipIdFromDiecutterId

# Function to get the maximum speed of a diecutter within a 30-second interval
def get_max_speed(diecutter_id):
    # Get the chip id from the diecutter id
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    max_speed = 0
    for _ in range(30):  # 30-second interval
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        max_speed = max(max_speed, current_data["speed"])
        time.sleep(1)  # Wait for 1 second before the next data fetch
    
    return max_speed

# Main script
if __name__ == "__main__":
    diecutter_id = 25
    max_speed = get_max_speed(diecutter_id=diecutter_id)
    
    # Prepare the data for the JSON file
    data = [{"diecutter_id": diecutter_id, "max_speed": max_speed}]
    
    # Write the data to a JSON file
    with open("max_speed_data.json", "w") as json_file:
        json.dump(data, json_file)

    print("Data has been written to max_speed_data.json")