from data_services.diecutter import GetCurrentDataFromChip
from data_services.database import GetDiecutterChipIdFromDiecutterId
import time

def pipeline_function():
    diecutter_id = 7
    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    results = []
    for window in range(10):
        speeds = []
        start_time = time.time()
        while time.time() - start_time < 10:
            current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
            speeds.append(current_data['speed'])
            time.sleep(1)
        
        average_speed = sum(speeds) / len(speeds)
        results.append({
            'interval': window + 1,
            'average_speed': average_speed,
        })
    
    return results

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
    