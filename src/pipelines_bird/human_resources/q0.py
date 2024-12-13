from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.position import GetDataFromPosition

def pipeline_function():
    
    #STANDARD
    ssn = ("222-52-5555", "EQUAL")
    results = []
    
    employees = GetDataFromEmployee()
    employees.open_connection()

    #RETRIEVE
    employee_df = employees.call(ssn=ssn)
    
    #STANDARD
    positionID = employee_df["positionID"].iloc[0]
    
    positions = GetDataFromPosition()
    positions.open_connection()
    
    #RETRIEVE
    position_df = positions.call(positionID=(positionID, "EQUAL"))
    
    #STANDARD
    position_info = position_df.iloc[0]
    position_info = position_info.astype(str)  
    
    results.append({
        'positionID': position_info['positionID'],
        'positiontitle': position_info['positiontitle'],
        'educationrequired': position_info['educationrequired'],
        'minsalary': position_info['minsalary'],
        'maxsalary': position_info['maxsalary']
    })
    
    #STOP
    return results
