from data_services.database import GetDiecutterChipIdFromDiecutterId
from data_services.database import GetCamera2IdFromDiecutterId
from data_services.database import GetCurrentSessionIdFromDiecutterId
from data_services.diecutter import GetCurrentDataFromChip
from data_services.camera2 import GetFrameFromCamera2
from data_services.camera2 import DetectErrorsInCardboardStackUsingCamera2
import threading
import time

def pipeline_function():
    diecutter_id = 14
    n_stacks = 10

    chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=diecutter_id)
    session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=diecutter_id)
    camera2_id = GetCamera2IdFromDiecutterId.call(diecutter_id=diecutter_id)

    def check_error(camera2_id, frame, card_errors_res):
        has_error, error_type = DetectErrorsInCardboardStackUsingCamera2.call(camera2_id=camera2_id, frame=frame)
        card_errors_res['has_error'] = has_error
        card_errors_res['error_type'] = error_type
    
    def retrieve_diecutter_data(chip_id, diecutter_info_res):
        current_data = GetCurrentDataFromChip.call(chip_id=chip_id)
        diecutter_info_res.update(current_data)

    results = []
    for _ in range(n_stacks):
        frame = GetFrameFromCamera2.call(camera2_id=camera2_id)
        timestamp = time.time()
        card_errors_res = {}
        card_errors = threading.Thread(target=check_error, args=(camera2_id, frame, card_errors_res))
        diecutter_info_res = {}
        diecutter_info = threading.Thread(target=retrieve_diecutter_data, args=(chip_id, diecutter_info_res))
        card_errors.start()
        diecutter_info.start()
        card_errors.join()
        diecutter_info.join()
        result_row = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(timestamp)),
            'session_id': session_id,
            'has_error': card_errors_res['has_error'],
            'error_type': card_errors_res['error_type'],
            'speed': diecutter_info_res['speed'],
            'temperature': diecutter_info_res['temperature']
        }
        results.append(result_row)

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
    