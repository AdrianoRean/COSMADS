import sys

sys.path.append("../../../")

from data_services.diecutter import GetCurrentDataFromChip
from data_services.database import GetDiecutterChipIdFromDiecutterId
import time

def pipeline_function():
    diecutter_id = 25
    
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
        
    speeds = []
    start_time = time.time()
    end_time = start_time + 30
    while time.time() < end_time:
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        speeds.append(current_data["speed"])
        time.sleep(1)
    max_speed = max(speeds)
    results = {
        'max_speed': max_speed
    }
    return [results]

if __name__ == "__main__":
    result = pipeline_function()
    import json
    import pandas as pd
    with open("result.json", "w") as f:
        json.dump(result, f, indent=4)
    result = pd.DataFrame(result)
    from tabulate import tabulate
    result = tabulate(result, headers='keys', tablefmt='psql')
    print(result)
    