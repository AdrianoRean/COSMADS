import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

# Fetch the list of diecutter IDs in the factory
diecutters_id = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to hold the tabular data
tabular_data = []

# Iterate over each diecutter ID to fetch the serial numbers for cameras 1 and 2
for diecutter_id in diecutters_id:
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)
    
    # Create a dictionary for the current row in the table
    row = {
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    }
    
    # Append the row to the list of tabular data
    tabular_data.append(row)

# Write the tabular data to a JSON file
with open('serial_numbers.json', 'w') as json_file:
    json.dump(tabular_data, json_file, indent=4)

print("JSON file 'serial_numbers.json' has been created with the serial numbers for cameras 1 and 2 from each diecutter.")