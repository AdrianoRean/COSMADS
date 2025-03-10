import json
import numpy as np
from diecutter import GetCurrentDataFromChip

# Simulate fetching the chip ID for diecutter 14
# Assuming a function or a mapping exists that can provide this, here we simulate it
chip_id_for_diecutter_14 = 1400  # This is a simulated value

# Simulate fetching error details for the upcoming 10 stacks
# In a real scenario, this would involve querying a database or an API
def simulate_stack_errors(num_stacks):
    np.random.seed(42)  # For reproducibility
    errors = []
    for _ in range(num_stacks):
        has_error = np.random.choice([True, False], p=[0.2, 0.8])
        if has_error:
            error_type = np.random.choice(["Type A", "Type B", "Type C"])
        else:
            error_type = None
        errors.append({"has_error": has_error, "error_type": error_type})
    return errors

# Get current data from diecutter 14
current_data = GetCurrentDataFromChip.call(chip_id=chip_id_for_diecutter_14)

# Generate table data
table_data = []
stack_errors = simulate_stack_errors(10)
for stack_error in stack_errors:
    row = {
        "has_error": stack_error["has_error"],
        "error_type": stack_error["error_type"],
        "current_speed": current_data["speed"],
        "operating_temperature": current_data["temperature"]
    }
    table_data.append(row)

# Save the table data to a JSON file
with open("diecutter_14_stacks_data.json", "w") as file:
    json.dump(table_data, file, indent=4)

print("Data saved to diecutter_14_stacks_data.json")