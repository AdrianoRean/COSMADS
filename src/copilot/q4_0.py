import json
import numpy as np
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

def generate_table(diecutter_id, stacks):
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id)
    chip_id = session_id  # Assuming chip_id can be derived from session_id for simulation purposes

    table = []
    for _ in range(stacks):
        current_data = GetCurrentDataFromChip.call(chip_id)
        speed = current_data["speed"]
        temperature = current_data["temperature"]

        # Simulating error presence and type
        has_error = np.random.choice([True, False], p=[0.2, 0.8])
        error_type = np.random.choice(["Type A", "Type B", "Type C"]) if has_error else None

        row = {
            "has_error": has_error,
            "error_type": error_type,
            "current_speed": speed,
            "current_temperature": temperature
        }
        table.append(row)

    return table

# Diecutter ID and number of stacks
diecutter_id = 14
stacks = 10

# Generate table
table_data = generate_table(diecutter_id, stacks)

# Save to JSON file
with open("table_data.json", "w") as json_file:
    json.dump(table_data, json_file, indent=4)

print("Table data saved to table_data.json")