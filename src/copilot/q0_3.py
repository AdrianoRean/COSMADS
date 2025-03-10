import json
from database import GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId, GetDiecuttersIdOfFactory

# Assuming GetDiecuttersIdOfFactory returns a list of diecutter IDs within the plant
diecutter_ids = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to hold the data
data = []

# Loop through each diecutter ID to get the serial numbers for both cameras
for diecutter_id in diecutter_ids:
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    # Append a dictionary for each row in the table
    data.append({
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    })

# Write the data to a JSON file
with open('serial_numbers.json', 'w') as f:
    json.dump(data, f)

print("JSON file 'serial_numbers.json' has been created with the serial numbers for both cameras from every diecutter within the plant.")