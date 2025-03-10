import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip
import numpy as np

# Step 1: Get the current session ID of diecutter 14
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=14)

# Step 2: Get the chip ID of diecutter 14
chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=14)

# Initialize the list to hold each row of the table
table_data = []

# Step 3: Generate data for the next 10 stacks
for stack in range(10):
    # Simulate the presence of errors and their types
    error_present = np.random.choice([True, False])
    error_type = np.random.choice(['Mechanical', 'Electrical', 'None'])
    if not error_present:
        error_type = 'None'
    
    # Get the current operational speed and temperature of diecutter 14
    current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
    
    # Create a dictionary for the current stack
    stack_data = {
        'session_id': session_id,
        'stack_number': stack + 1,
        'error_present': error_present,
        'error_type': error_type,
        'operational_speed': current_data['speed'],
        'temperature': current_data['temperature']
    }
    
    # Add the dictionary to the list
    table_data.append(stack_data)

# Step 4: Write the list to a JSON file
with open('session_data.json', 'w') as json_file:
    json.dump(table_data, json_file, indent=4)

print("JSON file 'session_data.json' has been generated.")