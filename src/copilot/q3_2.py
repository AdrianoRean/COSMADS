import json
import numpy as np

# Simulate processing of cardboards
np.random.seed(7)  # Using diecutter id as seed for reproducibility
total_cardboards = 30
defects = np.random.randint(0, 2, total_cardboards)  # 0 for defect-free, 1 for defects

# Count defect-free and defective cardboards
defect_free_count = np.count_nonzero(defects == 0)
defects_count = np.count_nonzero(defects == 1)

# Construct the table
table = [
    {"count_of_defect_free_cardboards": defect_free_count, "count_of_cardboards_with_defects": defects_count}
]

# Write the table to a JSON file
with open("cardboard_quality.json", "w") as json_file:
    json.dump(table, json_file, indent=4)

print("JSON file 'cardboard_quality.json' has been created.")