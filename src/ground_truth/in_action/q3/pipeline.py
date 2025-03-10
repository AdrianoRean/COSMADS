from data_services.camera1 import GetFrameFromCamera1
from data_services.camera1 import DetectErrorsInCardboardUsingCamera1
from data_services.database import GetCamera1IdFromDiecutterId


def pipeline_function():
    diecutter_id = 7
    num_cardboards = 30
    no_defects_count = 0
    with_errors_count = 0
    
    camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
    
    for _ in range(num_cardboards):
        frame = GetFrameFromCamera1.call(camera1_id=camera1_id)
        has_error, _ = DetectErrorsInCardboardUsingCamera1.call(camera1_id=camera1_id, frame=frame)
        if has_error:
            with_errors_count += 1
        else:
            no_defects_count += 1
    
    results = [{
        'no_errors_count': no_defects_count,
        'with_errors_count': with_errors_count
    }]
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
    