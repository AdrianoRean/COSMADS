import json
import numpy as np
import time
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Constants
DIECUTTER_ID = 7
TIME_INTERVAL = 10  # seconds
NUM_INTERVALS = 10

def get_mean_speeds(diecutter_id):
    # Get the current session ID for the diecutter
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    # Initialize variables to store speed data
    interval_speeds = []
    mean_speeds = []
    
    # Simulate data collection for each time interval
    for _ in range(NUM_INTERVALS):
        interval_speeds.clear()
        start_time = time.time()
        
        # Collect speed data for the current interval
        while time.time() - start_time < TIME_INTERVAL:
            current_data = GetCurrentDataFromChip.call(chip_id=session_id)
            interval_speeds.append(current_data["speed"])
            time.sleep(1)  # Simulate waiting for new data
        
        # Calculate the mean speed for the current interval
        mean_speed = np.mean(interval_speeds)
        mean_speeds.append({"interval_mean_speed": mean_speed})
    
    return mean_speeds

# Generate the table data
table_data = get_mean_speeds(DIECUTTER_ID)

# Write the table data to a JSON file
with open("mean_speeds_session.json", "w") as json_file:
    json.dump(table_data, json_file)

print("JSON file with mean speeds for the present session of diecutter 7 has been generated.")