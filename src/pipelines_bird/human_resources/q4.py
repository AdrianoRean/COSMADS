from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.position import GetDataFromPosition
from data_service_bird.human_resources.location import GetDataFromLocation

def pipeline_function():
    positiontitle = ("Manager", "EQUAL")

    results = []

    employees = GetDataFromEmployee()
    positions = GetDataFromPosition()
    locations = GetDataFromLocation()
    
    employees.open_connection()
    positions.open_connection()
    locations.open_connection()
    
    position_df = positions.call(positiontitle=positiontitle)
    positionID = position_df['positionID'].iloc[0]
    
    employee_df = employees.call(positionID=(positionID, "EQUAL"))
    
    locationecities = []
    for index, employee in employee_df.iterrows():
        locationID = employee["locationID"]
        locations_df = locations.call(locationID=(locationID, "EQUAL"))
        locationcity = locations_df['locationcity'].iloc[0]
        locationecities.append(str(locationcity))

    locationecities = list(set(locationecities))
    
    for city in locationecities: 
        results.append({
            'locationcity': city
        })
    
    return results
