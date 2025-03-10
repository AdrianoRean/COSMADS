import json
from database import GetDiecuttersIdOfFactory, GetSerialCamera1IdFromDiecutterId, GetSerialCamera2IdFromDiecutterId

def generate_serial_numbers_json():
    # Get all diecutter IDs in the factory
    diecutters_id = GetDiecuttersIdOfFactory.call()

    # Initialize an empty list to hold the data
    data = []

    # Iterate through each diecutter ID
    for diecutter_id in diecutters_id:
        # Get the serial number for camera 1 and camera 2 for the current diecutter
        serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id)
        serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id)

        # Append the data as a dictionary to the list
        data.append({
            "diecutter_id": diecutter_id,
            "serial_camera1": serial_camera1,
            "serial_camera2": serial_camera2
        })

    # Write the data to a JSON file
    with open('serial_numbers.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Call the function to generate the JSON file
generate_serial_numbers_json()