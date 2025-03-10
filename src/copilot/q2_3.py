import time
import json
from database import GetCurrentSessionIdFromDiecutterId, GetDiecutterChipIdFromDiecutterId
from diecutter import GetCurrentDataFromChip

def get_mean_speeds(diecutter_id, intervals, interval_length):
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    mean_speeds = []

    for _ in range(intervals):
        speeds = []
        start_time = time.time()
        while time.time() - start_time < interval_length:
            current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
            speeds.append(current_data["speed"])
            time.sleep(1)  # Assuming we collect data every second within the interval

        mean_speed = sum(speeds) / len(speeds) if speeds else 0
        mean_speeds.append({"session_id": session_id, "interval_start": start_time, "mean_speed": mean_speed})

    return mean_speeds

def generate_json_file(data, filename="mean_speeds_session.json"):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    diecutter_id = 7
    intervals = 10
    interval_length = 10  # 10 seconds
    mean_speeds = get_mean_speeds(diecutter_id, intervals, interval_length)
    generate_json_file(mean_speeds)