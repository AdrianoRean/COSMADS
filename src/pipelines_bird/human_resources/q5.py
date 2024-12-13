from data_service_bird.human_resources.employee import GetDataFromEmployee

def pipeline_function():
    positionID = (3, "MINOR OR EQUAL")

    results = []

    employees = GetDataFromEmployee()
    
    employee_df = employees.call(positionID=positionID)
    
    employees_count = employee_df.shape[0]
    
    results.append({
            'employees count': employees_count
        })
    
    return results
