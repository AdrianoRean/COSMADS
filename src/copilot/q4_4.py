import json
import numpy as np
import time
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

# Constants
DIECUTTER_ID = 14
NUM_STACKS = 10

# Get the current session ID for diecutter 14
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=DIECUTTER_ID)

# Get the chip ID for diecutter 14
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=DIECUTTER_ID)

# Initialize the list to store the data for each stack
stacks_data = []

for stack in range(NUM_STACKS):
    # Get the current data from the chip
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    
    # Detect errors using both cameras
    errors_camera1 = DetectErrorsInCardboardUsingCamera1()
    errors_camera2 = DetectErrorsInCardboardStackUsingCamera2()
    
    # Determine if there are errors and the error type
    has_errors = errors_camera1['has_errors'] or errors_camera2['has_errors']
    error_type = None
    if has_errors:
        error_type = errors_camera1.get('error_type', errors_camera2.get('error_type', 'Unknown'))
    
    # Append the data for the current stack to the list
    stacks_data.append({
        "stack_number": stack + 1,
        "presence_of_errors": has_errors,
        "error_type": error_type,
        "operating_speed": current_data["speed"],
        "current_temperature": current_data["temperature"]
    })

# Write the data to a JSON file
with open('diecutter_session_details.json', 'w') as f:
    json.dump(stacks_data, f, indent=4)

print("Data for the ongoing session of diecutter 14 has been saved to 'diecutter_session_details.json'")