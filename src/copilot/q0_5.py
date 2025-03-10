import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

# Fetch the list of diecutters installed in the factory
diecutters_id_list = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to hold the data for each diecutter
data_list = []

# Iterate over each diecutter ID to fetch the serial numbers for camera 1 and camera 2
for diecutter_id in diecutters_id_list:
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)
    
    # Create a dictionary for the current diecutter and append it to the data list
    data_list.append({
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    })

# Write the data list to a JSON file
with open('diecutters_camera_serials.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

print("JSON file 'diecutters_camera_serials.json' has been created with the serial numbers for camera 1 and camera 2 associated with every diecutter within the factory.")