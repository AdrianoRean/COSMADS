import pandas as pd
import sqlite3
import inspect
import os

class GetDataFromLocation:
    database_location = "data_service_bird/human_resources/human_resources.sqlite"
    connection = None
    description = {
        "brief_description": "Data service that provides data in a dataframe format about offices and their location.",
        "detailed_description": 
        """Data service that provides data in a dataframe format about offices and their location.
        Each data entry has the following attributes: locationID, locationcity, address, state, zipcode, officephone.
        The attribute "locationID" is unique for each office.
        You may select data trough any combination of this attributes. They are all optional.
        If all attributes are left undeclared, it returns all the available data.

        Example usage:
        - If I want to obtain all the information from the office with locationID 123 I can write:
        locationID = 123
        location_df = GetDataFromLocation.call(locationID=123)
        # assuming the result is a pandas dataframe
        print(location_df.shape)

        Things to keep in mind:
        - The frame is a pandas dataframe, so you may order, project and group the result if needed.""",
        "input_parameters": ["locationID:int", "locationcity:str", "address:str", "state:str", "zipcode:int", "officephone:str"],
        "output_values": ["location_df:pandas.DataFrame"],
        "module": "human_resources"
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
            for param, value in passed_args.items():
                query = query + f" {param} = {value}"
                
        df = pd.read_sql_query(query, self.connection)
        return df
    
'''
#if run as main, add 'src/' to db path

gd = GetDataFromLocation()

print(os.getcwd())

gd.open_connection()
    
print(gd.call())
'''