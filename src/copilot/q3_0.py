import json
from database import GetCamera1IdFromDiecutterId, GetCamera2IdFromDiecutterId
from camera1 import DetectErrorsInCardboardUsingCamera1
from camera2 import DetectErrorsInCardboardStackUsingCamera2

# Initialize the count of cardboards with and without defects
cardboards_with_defects = 0
cardboards_without_defects = 0

# Get the camera IDs for the given diecutter ID
diecutter_id = 7
camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
camera2_id = GetCamera2IdFromDiecutterId.call(diecutter_id=diecutter_id)

# Consider the next 30 cardboards
for _ in range(30):
    # Simulate checking each cardboard with both cameras
    has_errors_camera1 = DetectErrorsInCardboardUsingCamera1(camera1_id)
    has_errors_camera2 = DetectErrorsInCardboardStackUsingCamera2(camera2_id)
    
    # If either camera detects errors, increment the count of cardboards with defects
    if has_errors_camera1 or has_errors_camera2:
        cardboards_with_defects += 1
    else:
        cardboards_without_defects += 1

# Prepare the data for the JSON file
data = [
    {"cardboards_with_no_defects": cardboards_without_defects, "cardboards_with_defects": cardboards_with_defects}
]

# Write the data to a JSON file
with open('cardboards_data.json', 'w') as json_file:
    json.dump(data, json_file)

print("Data has been written to cardboards_data.json")