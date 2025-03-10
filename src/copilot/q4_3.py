import json
import numpy as np
import time
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

def generate_table(diecutter_id, num_stacks):
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id)
    table_data = []

    for _ in range(num_stacks):
        current_data = GetCurrentDataFromChip.call(chip_id)
        error_camera1 = DetectErrorsInCardboardUsingCamera1()
        error_camera2 = DetectErrorsInCardboardStackUsingCamera2()

        row = {
            "presence_of_errors": error_camera1['has_errors'] or error_camera2['has_errors'],
            "type_of_error": error_camera1.get('error_type', '') + ' ' + error_camera2.get('error_type', ''),
            "operational_speed": current_data["speed"],
            "temperature": current_data["temperature"]
        }
        table_data.append(row)
        time.sleep(1)  # Simulate time between processing stacks

    return table_data

# Generate the table for diecutter 14 and 10 stacks
table = generate_table(14, 10)

# Write the table to a JSON file
with open('diecutter_session_data.json', 'w') as f:
    json.dump(table, f)

print("Table data has been written to diecutter_session_data.json")