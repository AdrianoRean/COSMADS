from data_service_bird.human_resources.employee import GetDataFromEmployee

def pipeline_function():
    performance = ("Good", "EQUAL")

    results = []

    employees = GetDataFromEmployee()
    
    employees.open_connection()
    
    employee_df = employees.call(performance=performance)
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
    
    return results