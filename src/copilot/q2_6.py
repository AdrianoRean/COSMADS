import time
import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

def generate_mean_speeds_json(diecutter_id, intervals, interval_length):
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id)
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id)
    mean_speeds = []

    for _ in range(intervals):
        speeds = []
        start_time = time.time()
        while time.time() - start_time < interval_length:
            current_data = GetCurrentDataFromChip.call(chip_id)
            speeds.append(current_data["speed"])
            time.sleep(1)  # Ensure we don't call the function more than once per second

        mean_speed = sum(speeds) / len(speeds)
        mean_speeds.append({"session_id": session_id, "interval_start": start_time, "mean_speed": mean_speed})

    with open("mean_speeds.json", "w") as json_file:
        json.dump(mean_speeds, json_file, indent=4)

# Assuming diecutter_id is 7, we want 10 intervals of 10 seconds each
generate_mean_speeds_json(7, 10, 10)