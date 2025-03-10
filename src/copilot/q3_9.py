import json
import numpy as np
from diecutter import GetCurrentDataFromChip
from database import GetDiecutterChipIdFromDiecutterId

# Simulate determining if a cardboard piece is defect-free or contains defects
def is_defect_free(cardboard_data):
    # Example condition: if the temperature is below 15 and speed is above 5, it's defect-free
    return cardboard_data['temperature'] < 15 and cardboard_data['speed'] > 5

def generate_cardboard_inspection_report(diecutter_id, num_pieces):
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id)
    defect_free_count = 0
    defects_count = 0
    report = []

    for _ in range(num_pieces):
        cardboard_data = GetCurrentDataFromChip.call(chip_id)
        if is_defect_free(cardboard_data):
            defect_free_count += 1
        else:
            defects_count += 1
        report.append({
            "defect_free_count": defect_free_count,
            "defects_count": defects_count
        })

    return report

# Generate the report for the next 30 cardboard pieces processed by diecutter with ID 7
report = generate_cardboard_inspection_report(7, 30)

# Save the report to a JSON file
with open('cardboard_inspection_report.json', 'w') as f:
    json.dump(report, f)

print("Report generated and saved to cardboard_inspection_report.json.")