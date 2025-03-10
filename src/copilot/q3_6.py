import numpy as np
import json

# Simulate the inspection process
def simulate_inspection(diecutter_id, units):
    np.random.seed(diecutter_id)  # Seed for reproducibility
    results = []
    for _ in range(units):
        # Simulate the defect status, 0 for defect-free, 1 for defects
        defect_status = np.random.choice([0, 1], p=[0.8, 0.2])  # 80% chance of being defect-free
        if defect_status == 0:
            results.append({"defect-free": 1, "with defects": 0})
        else:
            results.append({"defect-free": 0, "with defects": 1})
    return results

# Generate the table for 30 cardboard units processed by diecutter number 7
table_data = simulate_inspection(diecutter_id=7, units=30)

# Save the table data as a JSON file
with open('cardboard_inspection_results.json', 'w') as json_file:
    json.dump(table_data, json_file, indent=4)

print("JSON file 'cardboard_inspection_results.json' has been generated.")