from data_service_bird.human_resources.employee import GetDataFromEmployee
from data_service_bird.human_resources.position import GetDataFromPosition
from data_service_bird.human_resources.location import GetDataFromLocation

def pipeline_function():
    
    #STANDARD
    positiontitle = ("Manager", "EQUAL")
    results = []
    
    positions = GetDataFromPosition()
    positions.open_connection()
    
    #RETRIEVE
    position_df = positions.call(positiontitle=positiontitle)
    
    #STANDARD
    positionID = position_df['positionID'].iloc[0]
    positionID = (positionID, "EQUAL")
    
    employees = GetDataFromEmployee()
    employees.open_connection()

    #RETRIEVE
    employee_df = employees.call(positionID=positionID)
    
    #STANDARD
    locations = GetDataFromLocation()
    locations.open_connection()
    
    locationecities = []
    for index, employee in employee_df.iterrows():
        locationID = employee["locationID"]
        
        #RETRIEVE
        locations_df = locations.call(locationID=(locationID, "EQUAL"))
        
        #STANDARD
        locationcity = locations_df['locationcity'].iloc[0]
        locationecities.append(str(locationcity))

    locationecities = list(set(locationecities))
    
    for city in locationecities: 
        results.append({
            'locationcity': city
        })
    
    #STOP    
    return results
