from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.location import GetDataFromLocation

def pipeline_function():
    address = "312 Mount View Dr"

    results = []

    locations = GetDataFromLocation()
    employees = GetDataFromEmployee()
    
    locations.open_connection()
    employees.open_connection()
    
    locations_df = locations.call(address=address)
    locationID = locations_df['locationID'].iloc[0]
    
    employees_df = employees.call(locationID=locationID)
    employees_info = employees_df[["firstname", "lastname"]]
    
    employees_info = employees_info.astype(str)  
    for index, employee in employees_info.iterrows():
        results.append({
            'firstname': employee["firstname"],
            'lastname': employee["lastname"]
        })
    
    return results