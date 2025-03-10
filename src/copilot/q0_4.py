import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

# Get the list of diecutters installed in the factory
diecutters_id_list = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to store the data
data = []

# Iterate through each diecutter ID to get the serial numbers for camera 1 and camera 2
for diecutter_id in diecutters_id_list:
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)
    # Append the data as a dictionary to the list
    data.append({
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    })

# Write the data to a JSON file
with open('serial_numbers.json', 'w') as f:
    json.dump(data, f, indent=4)

print("JSON file 'serial_numbers.json' has been created with the serial numbers for camera 1 and camera 2 associated with each diecutter throughout the factory.")