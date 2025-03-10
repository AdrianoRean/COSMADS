import numpy as np
from camera1 import DetectErrorsInCardboardUsingCamera1
from database import GetCamera1IdFromDiecutterId
import json

# Simulate the frame captured by the camera
def simulate_frame():
    return np.random.rand(10, 10)  # Simplified simulation

def assess_cardboards(diecutter_id, num_cardboards):
    camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
    defect_free_count = 0
    defects_count = 0
    
    for _ in range(num_cardboards):
        frame = simulate_frame()
        contains_errors, _ = DetectErrorsInCardboardUsingCamera1.call(camera1_id=camera1_id, frame=frame)
        if contains_errors:
            defects_count += 1
        else:
            defect_free_count += 1
    
    return {
        "defect_free_count": defect_free_count,
        "defects_count": defects_count
    }

# Assess the upcoming 30 cardboards processed by diecutter id 7
results = assess_cardboards(diecutter_id=7, num_cardboards=30)

# Save the results in a JSON file
with open('assessment_results.json', 'w') as f:
    json.dump([results], f)

print("Assessment results saved to 'assessment_results.json'")