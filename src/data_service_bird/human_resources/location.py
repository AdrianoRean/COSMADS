import pandas as pd
import sqlite3
import inspect
from data_service_bird.utilities import selectOperator

class GetDataFromLocation:
    database_location = "data_service_bird/human_resources/human_resources.sqlite"
    connection = None
    call_parameters_list = ["locationID", "locationcity", "address", "state", "zipcode", "officephone"]
    description = {
        "brief_description": "Data service that provides data in a dataframe format about offices and their location.",
        
        "detailed_description": 
        """Data service that provides data in a dataframe format about offices and their location.
        Each data entry has the following attributes: locationID, locationcity, address, state, zipcode, officephone.
        The attribute "locationID" is unique for each office.
        The attribute "locationcity" represent the city the office is in.
        The attribute "address" represent the actual address of the office.
        The attribute "state" is the state the office is in.
        The attribute "zipcode" is the postal code of the office.
        The attribute "officephone" is the telephone number of the office.""",

        "usage_example":"""
        # If I want to obtain all the information from the office with locationID 123 I can write:
        locationID = (123, "EQUAL")
        locations = GetDataFromLocation()
        locations.open_connection()
        location_df = location.call(locationID=locationID)
        # assuming the result is a pandas dataframe
        print(location_df.shape)
        """,
        
        "input_parameters": ["locationID:int", "locationcity:str", "address:str", "state:str", "zipcode:int", "officephone:str"],
        
        "output_values": ["location_df:pandas.DataFrame"],
        
        "module": "location"
    }
    
    def open_connection(self):
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, locationID = None, locationcity = None, address = None, state = None, zipcode = None, officephone = None) -> pd.DataFrame:
        # Ottieni la firma della funzione
        signature = inspect.signature(self.call)
        parameters = list(signature.parameters.keys())
        
        # Crea un dizionario con i parametri passati alla funzione
        passed_args = {k: v for k, v in locals().items() if v is not None and k in parameters}
        
        if passed_args == {}:
            query = "SELECT * FROM location"
        else:    
            query = "SELECT * FROM location WHERE"    
            # Cicla sugli argomenti effettivamente passati
            for param, (value, operator) in passed_args.items():
                operator = selectOperator(operator)
                if type(value) == str:
                    query = query + f" {param} {operator} '{value}' and"
                else:
                    query = query + f" {param} {operator} {value} and"
            query = query[:-4]
                
        df = pd.read_sql_query(query, self.connection)
        return df
    
'''
#if run as main, add 'src/' to db path

gd = GetDataFromLocation()

gd.open_connection()
    
print(gd.call())
'''