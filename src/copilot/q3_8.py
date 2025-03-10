import json
import numpy as np
from database import GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

def simulate_cardboard_processing(diecutter_id: int, units: int) -> list:
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id)
    defect_free_count = 0
    defects_count = 0
    
    for _ in range(units):
        data = GetCurrentDataFromChip.call(chip_id)
        # Simulate defect detection logic
        if data["speed"] > 5 and data["temperature"] < 20:
            defect_free_count += 1
        else:
            defects_count += 1
    
    return [{"defect_free_count": defect_free_count, "defects_count": defects_count}]

# Simulate processing for diecutter with id 7 for 30 units
tabular_data = simulate_cardboard_processing(7, 30)

# Generate JSON file
with open("cardboard_processing_results.json", "w") as json_file:
    json.dump(tabular_data, json_file)

print("JSON file 'cardboard_processing_results.json' has been generated.")