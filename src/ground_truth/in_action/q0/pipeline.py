from data_services.database import GetSerialCamera1IdFromDiecutterId
from data_services.database import GetSerialCamera2IdFromDiecutterId
from data_services.database import GetDiecuttersIdOfFactory

def pipeline_function():
    results = []

    diecutters_ids = GetDiecuttersIdOfFactory.call()

    for diecutter_id in diecutters_ids:
        serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id=diecutter_id)
        serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id=diecutter_id)
        results.append({
            'diecutter_id': diecutter_id,
            'serial_camera1': serial_camera1,
            'serial_camera2': serial_camera2
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
    