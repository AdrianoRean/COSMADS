import json
import numpy as np
import random
from database import GetDiecutterChipIdFromDiecutterId, GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Step 1: Get the current session ID for diecutter 14
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=14)

# Step 2: Get the chip ID of diecutter 14
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=14)

# Initialize the list to store each row of the table
table_data = []

# Step 3: Simulate data for the next 10 stacks produced by diecutter 14
for _ in range(10):
    # Get current data from the chip
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    
    # Randomly determine if there are errors
    has_errors = random.choice([True, False])
    error_type = None
    if has_errors:
        # Randomly generate an error type
        error_type = random.choice(['Mechanical Failure', 'Material Defect', 'Operator Error'])
    
    # Collect the data for the current stack
    stack_data = {
        'has_errors': has_errors,
        'error_type': error_type,
        'operating_speed': current_data['speed'],
        'operating_temperature': current_data['temperature']
    }
    
    # Add the stack data to the table data list
    table_data.append(stack_data)

# Step 6: Write the table data to a JSON file
with open('diecutter_14_session_data.json', 'w') as json_file:
    json.dump(table_data, json_file, indent=4)

print(f"Data for session {session_id} of diecutter 14 has been saved to 'diecutter_14_session_data.json'")