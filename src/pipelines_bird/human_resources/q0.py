from data_services_bird.human_resources.employee import GetDataFromEmployee
from data_services_bird.human_resources.position import GetDataFromPosition

def pipeline_function():
    ssn = 222-52-5555

    results = []

    employee_df = GetDataFromEmployee.call(ssn=ssn)
    positionID = employee_df["positionID"].iloc[0]
    position_df = GetDataFromPosition.call(positionID=positionID)
    position_info = position_df.iloc[0]
    results.append({
        'positionID': position_info['positionID'],
        'positiontitle': position_info['positiontitle'],
        'educationrequired': position_info['educationrequired'],
        'minsalary': position_info['minsalary'],
        'maxsalary': position_info['maxsalary']
    })
    
    return results