import json
import numpy as np
from diecutter import GetCurrentDataFromChip
from database import GetDiecutterChipIdFromDiecutterId

def generate_table(diecutter_id: int, stacks: int):
    # Get the chip ID for the diecutter
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    table_data = []
    for _ in range(stacks):
        # Simulate getting current data from the chip
        current_data = GetCurrentDataFromChip.call(chip_id)
        
        # Simulate error presence and specific error types
        error_present = np.random.choice([True, False], p=[0.2, 0.8])
        specific_error_types = None
        if error_present:
            specific_error_types = np.random.choice(["Error Type 1", "Error Type 2", "Error Type 3"], size=np.random.randint(1, 4), replace=False).tolist()
        
        # Append the data for the current stack
        table_data.append({
            "error_presence": error_present,
            "specific_error_types": specific_error_types,
            "present_operating_speed": current_data["speed"],
            "current_operating_temperature": current_data["temperature"]
        })
    
    # Write the table data to a JSON file
    with open("diecutter_operation_details.json", "w") as json_file:
        json.dump(table_data, json_file, indent=4)

# Assuming diecutter ID 14 and we want details for the next 10 stacks
generate_table(diecutter_id=14, stacks=10)