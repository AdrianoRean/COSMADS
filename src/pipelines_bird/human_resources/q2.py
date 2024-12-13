from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.location import GetDataFromLocation

def pipeline_function():
    #STANDARD
    address = ("312 Mount View Dr", "EQUAL")
    results = []
    
    locations = GetDataFromLocation()
    locations.open_connection()

    #RETRIEVE
    locations_df = locations.call(address=address)
    
    #STANDARD
    locationID = locations_df['locationID'].iloc[0]
    
    employees = GetDataFromEmployee()
    employees.open_connection()
    
    #RETRIEVE
    employees_df = employees.call(locationID=(locationID, "EQUAL"))
    
    #STANDARD
    employees_info = employees_df[["firstname", "lastname"]]
    employees_info = employees_info.astype(str)  
    
    for index, employee in employees_info.iterrows():
        results.append({
            'firstname': employee["firstname"],
            'lastname': employee["lastname"]
        })
        
    #STOP    
    return results