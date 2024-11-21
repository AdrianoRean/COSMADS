from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.position import GetDataFromPosition

def pipeline_function():
    ssn = "222-52-5555"

    results = []

    employees = GetDataFromEmployee()
    positions = GetDataFromPosition()
    
    employees.open_connection()
    positions.open_connection()
    
    employee_df = employees.call(ssn=ssn)
    positionID = employee_df["positionID"].iloc[0]
    position_df = positions.call(positionID=positionID)
    position_info = position_df.iloc[0]
    
    position_info = position_info.astype(str)  
    results.append({
        'positionID': position_info['positionID'],
        'positiontitle': position_info['positiontitle'],
        'educationrequired': position_info['educationrequired'],
        'minsalary': position_info['minsalary'],
        'maxsalary': position_info['maxsalary']
    })
    
    return results
