import numpy as np
import json
from database import GetCamera1IdFromDiecutterId

# Simulate the DetectErrorsInCardboardUsingCamera1 class
def simulate_detect_errors(camera_id, num_units):
    np.random.seed(camera_id)
    defects = np.random.binomial(n=1, p=0.2, size=num_units)  # Assuming 20% chance of defects
    defect_free_count = len(defects) - np.sum(defects)
    defective_count = np.sum(defects)
    return defect_free_count, defective_count

def generate_report(diecutter_id, num_units):
    # Get the camera1 id associated with the diecutter
    camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    # Simulate the examination of cardboard units
    defect_free_count, defective_count = simulate_detect_errors(camera1_id, num_units)
    
    # Prepare the data for the JSON file
    data = {
        "diecutter_id": diecutter_id,
        "total_units_examined": num_units,
        "defect_free_count": defect_free_count,
        "defective_count": defective_count
    }
    
    # Save the data to a JSON file
    with open("report.json", "w") as json_file:
        json.dump([data], json_file, indent=4)

# Generate the report for diecutter id 7 examining 30 units
generate_report(diecutter_id=7, num_units=30)