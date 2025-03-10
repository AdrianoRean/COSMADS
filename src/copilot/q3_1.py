import json
from database import GetCamera1IdFromDiecutterId, GetCamera2IdFromDiecutterId
# Assuming the existence of these functions based on the workspace information
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

def simulate_cardboard_processing(diecutter_id, num_cardboards):
    camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id)
    camera2_id = GetCamera2IdFromDiecutterId.call(diecutter_id)
    
    defect_free_count = 0
    defect_count = 0
    
    for _ in range(num_cardboards):
        # Simulate defect detection using Camera1 and Camera2
        has_defect_camera1 = DetectErrorsInCardboardUsingCamera1(camera1_id)
        has_defect_camera2 = DetectErrorsInCardboardStackUsingCamera2(camera2_id)
        
        if not has_defect_camera1 and not has_defect_camera2:
            defect_free_count += 1
        else:
            defect_count += 1
    
    return {
        "defect_free_count": defect_free_count,
        "defect_count": defect_count
    }

# Simulate processing of 30 cardboards for diecutter with id 7
result = simulate_cardboard_processing(7, 30)

# Generate JSON file
with open('cardboard_processing_results.json', 'w') as json_file:
    json.dump([result], json_file)

print("JSON file 'cardboard_processing_results.json' has been generated.")