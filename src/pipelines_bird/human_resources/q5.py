from data_service_bird.human_resources.employee import GetDataFromEmployee

def pipeline_function():
    
    #STANDARD
    positionID = (3, "MINOR OR EQUAL")
    results = []

    employees = GetDataFromEmployee()
    employees.open_connection()
    
    #RETRIEVE    
    employee_df = employees.call(positionID=positionID)
    
    #STANDARD
    employees_count = employee_df.shape[0]
    
    results.append({
            'employees count': employees_count
        })
    
    #STOP
    return results
