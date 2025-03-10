import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

# Fetch the list of diecutter IDs in the factory
diecutters_id = GetDiecuttersIdOfFactory.call()

# Initialize an empty list to hold the data for each row
table_data = []

# Iterate over each diecutter ID to fetch the serial numbers for camera1 and camera2
for diecutter_id in diecutters_id:
    serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
    serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)
    
    # Create a dictionary for the current row and append it to the table_data list
    row = {
        "diecutter_id": diecutter_id,
        "serial_camera1": serial_camera1,
        "serial_camera2": serial_camera2
    }
    table_data.append(row)

# Convert the table_data list to JSON format
json_data = json.dumps(table_data, indent=4)

# Write the JSON data to a file
with open("serial_numbers.json", "w") as json_file:
    json_file.write(json_data)

print("JSON file 'serial_numbers.json' has been created with the serial numbers for cameras one and two from every diecutter in the plant.")