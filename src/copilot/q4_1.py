import json
import numpy as np
import time
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

# Initialize the list to hold the table rows
table_data = []

# Get the chip ID for diecutter 14
diecutter_id = 14
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)

for _ in range(10):
    # Get the current data from the chip
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    
    # Check for errors using both cameras
    has_errors_camera1, error_type_camera1 = DetectErrorsInCardboardUsingCamera1()
    has_errors_camera2, error_type_camera2 = DetectErrorsInCardboardStackUsingCamera2()
    
    # Determine if there are any errors and the error type
    has_errors = has_errors_camera1 or has_errors_camera2
    error_type = error_type_camera1 if has_errors_camera1 else error_type_camera2 if has_errors_camera2 else None
    
    # Extract the running speed and current temperature from the current data
    running_speed = current_data["speed"]
    current_temperature = current_data["temperature"]
    
    # Append the row to the table data
    table_data.append({
        "has_errors": has_errors,
        "error_type": error_type,
        "running_speed": running_speed,
        "current_temperature": current_temperature
    })
    
    # Wait for a bit before processing the next stack
    time.sleep(1)

# Write the table data to a JSON file
with open("diecutter_operation_details.json", "w") as json_file:
    json.dump(table_data, json_file, indent=4)

print("Diecutter operation details have been saved to diecutter_operation_details.json")