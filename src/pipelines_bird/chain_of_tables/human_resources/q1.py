from data_service_bird.human_resources.employee import GetDataFromEmployee

def pipeline_function():
    
    #STANDARD
    performance = ("Good", "EQUAL")
    results = []
    
    employees = GetDataFromEmployee()
    employees.open_connection()

    #RETRIEVE
    employee_df = employees.call(performance=performance)
    
    #STANDARD
    salaries = employee_df['salary']
    salaries = salaries.str.replace("US$", "")
    salaries = salaries.str.replace(",", "")
    salaries = salaries.astype(float)
    
    average_salary = salaries.mean()
    average_salary = str(average_salary)
    
    results.append({
        'performance': performance,
        'average salary': average_salary
    })
    
    #STOP
    return results