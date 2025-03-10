import json
import numpy as np
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip
# Assuming DetectErrorsInCardboardUsingCamera1 and DetectErrorsInCardboardStackUsingCamera2 are functions that return a boolean and a list of strings respectively
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

def generate_table(diecutter_id):
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id)
    diecutter_data = GetCurrentDataFromChip.call(chip_id)
    speed = diecutter_data["speed"]
    temperature = diecutter_data["temperature"]
    
    table_data = []
    for stack_number in range(10):
        has_errors_camera1 = DetectErrorsInCardboardUsingCamera1()
        has_errors_camera2 = DetectErrorsInCardboardStackUsingCamera2()
        errors = []
        if has_errors_camera1 or has_errors_camera2:
            errors = ["Type1", "Type2"]  # Example error types, replace with actual error detection logic
        
        row = {
            "stack_number": stack_number + 1,
            "presence_of_errors": bool(errors),
            "types_of_errors": errors,
            "operational_speed": speed,
            "operating_temperature": temperature
        }
        table_data.append(row)
    
    with open("diecutter_14_data.json", "w") as json_file:
        json.dump(table_data, json_file, indent=4)

# Call the function with diecutter_id 14
generate_table(14)