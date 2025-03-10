import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

# Retrieve the list of diecutters installed in the factory
diecutters_id = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to store the tabular data
tabular_data = []

# Iterate over each diecutter ID to retrieve the serial numbers for camera 1 and camera 2
for diecutter_id in diecutters_id:
    # Retrieve the serial number for camera 1 installed in the current diecutter
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
    # Retrieve the serial number for camera 2 installed in the current diecutter
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)
    
    # Append the retrieved data as a dictionary to the tabular data list
    tabular_data.append({
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    })

# Write the tabular data to a JSON file
with open('serial_numbers.json', 'w') as json_file:
    json.dump(tabular_data, json_file, indent=4)

print("JSON file 'serial_numbers.json' has been created with the serial numbers for camera 1 and camera 2 from every diecutter within the plant.")