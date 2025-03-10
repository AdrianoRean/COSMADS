import json
import numpy as np
import time
from database import GetCurrentSessionIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

# Constants
DIECUTTER_ID = 7
TIME_INTERVAL = 10  # seconds
NUM_INTERVALS = 10

def get_mean_speeds(diecutter_id: int) -> list:
    # Get the current session ID for the diecutter
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)
    print(f"Session ID for Diecutter {diecutter_id}: {session_id}")

    mean_speeds = []
    for _ in range(NUM_INTERVALS):
        speeds = []
        start_time = time.time()
        while time.time() - start_time < TIME_INTERVAL:
            # Assuming the chip ID is the same as the diecutter ID for simplicity
            current_data = GetCurrentDataFromChip.call(chip_id=diecutter_id)
            speeds.append(current_data["speed"])
            time.sleep(1)  # Wait for a second before the next data fetch

        # Calculate the mean speed for the current interval
        mean_speed = np.mean(speeds) if speeds else 0
        mean_speeds.append({"interval": len(mean_speeds) + 1, "mean_speed": mean_speed})

    return mean_speeds

# Generate the table
mean_speeds_table = get_mean_speeds(DIECUTTER_ID)

# Save the table to a JSON file
with open("mean_speeds_table.json", "w") as file:
    json.dump(mean_speeds_table, file, indent=4)

print("Mean speeds table saved to mean_speeds_table.json.")