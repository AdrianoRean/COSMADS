import json
import numpy as np
import random
import time
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Set the diecutter ID for which the session information is needed
diecutter_id = 14

# Get the current session ID for the diecutter
session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id)

# Assuming chip_id is the same as diecutter_id for simplicity
chip_id = diecutter_id

# Initialize the list to store the table rows
table_data = []

for _ in range(10):  # For each of the upcoming 10 stacks
    # Get the current data from the chip
    current_data = GetCurrentDataFromChip.call(chip_id)
    
    # Simulate whether the stack has any errors
    has_error = random.choice([True, False])
    error_type = None
    if has_error:
        error_type = random.choice(["Paper Jam", "Ink Low", "Mechanical Failure"])
    
    # Extract the operating speed and temperature of the diecutter
    operating_speed = current_data["speed"]
    present_temperature = current_data["temperature"]
    
    # Append the row to the table data
    table_data.append({
        "session_id": session_id,
        "has_error": has_error,
        "error_type": error_type,
        "operating_speed": operating_speed,
        "present_temperature": present_temperature
    })

# Write the table data to a JSON file
with open("session_info.json", "w") as json_file:
    json.dump(table_data, json_file, indent=4)

print("Session information for diecutter 14 has been saved to 'session_info.json'")