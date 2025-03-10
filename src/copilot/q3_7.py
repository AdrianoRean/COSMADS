import json
import numpy as np

# Simulate checking for defects in cardboard units
def check_for_defects():
    # This is a placeholder. In a real scenario, this function would
    # check if a cardboard unit has defects or not.
    return np.random.choice(['defect-free', 'defect'], p=[0.8, 0.2])

def main():
    diecutter_id = 7
    results = {"defect-free": 0, "defect": 0}
    
    for _ in range(30):
        result = check_for_defects()
        results[result] += 1
    
    # Prepare data for JSON file
    data_to_save = [
        {"count_of_defect_free_cardboards": results["defect-free"], "count_of_cardboards_with_defects": results["defect"]}
    ]
    
    # Save data to a JSON file
    with open("cardboard_quality_report.json", "w") as file:
        json.dump(data_to_save, file, indent=4)
    
    print("Report generated successfully.")

if __name__ == "__main__":
    main()