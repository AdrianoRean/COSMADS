from data_service_bird.human_resources.position import GetDataFromPosition

def pipeline_function():
    positiontitle = ("Account Representative", "EQUAL")

    results = []

    positions = GetDataFromPosition()
    
    positions.open_connection()
    
    position_df = positions.call(positiontitle=positiontitle)
    
    minsalary = position_df['minsalary']
    minsalary = minsalary.str.replace("US$", "")
    minsalary = minsalary.str.replace(",", "")
    minsalary = float(minsalary)
    
    maxsalary = position_df['maxsalary']
    maxsalary = maxsalary.str.replace("US$", "")
    maxsalary = maxsalary.str.replace(",", "")
    maxsalary = float(maxsalary)
    
    mean_salary = (minsalary + maxsalary)/2
    
    mean_salary = str(mean_salary)
    results.append({
        'positiontitle': positiontitle,
        'mean_salary': mean_salary
    })
    
    return results
