import pandas as pd
import sqlite3
import inspect

class GetDataFromLocation:
    database_location = "data_service_bird/human_resources/human_resources.sqlite"
    connection = None
    description = {
        "brief_description": "Data service that provides data in a dataframe format about offices and their location.",
        "detailed_description": 
        """Data service that provides data in a dataframe format about offices and their location.
        Each data entry has the following attributes: location_ID, location_city, address, state, zipcode, office_phone.
        The attribute "locationID" is unique for each office.
        You may select data trough any combination of this attributes. They are all optional.
        If all attributes are left undeclared, it returns all the available data.

        Example usage:
        - If I want to obtain all the information from the office with locationID 123 I can write:
        location_id = 123
        location_df = GetDataFromLocation.call(location_ID=123)
        # assuming the result is a pandas dataframe
        print(location_df.shape)

        Things to keep in mind:
        - The frame is a pandas dataframe, so you may order, project and group the result if needed.""",
        "input_parameters": ["location_ID:int", "location_city:str", "address:str", "state:str", "zipcode:int", "office_phone:str"],
        "output_values": ["location_df:pandas.DataFrame"],
        "module": "human_resources"
    }
    
    def open_connection(self):
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, location_ID = None, location_city = None, address = None, state = None, zipcode = None, office_phone = None) -> pd.DataFrame:
        # Ottieni la firma della funzione
        signature = inspect.signature(self.call)
        parameters = list(signature.parameters.keys())
        
        # Crea un dizionario con i parametri passati alla funzione
        passed_args = {k: v for k, v in locals().items() if v is not None and k in parameters}
        
        if passed_args == {}:
            query = "SELECT * FROM table"
        else:    
            query = "SELECT * FROM table WHERE"    
            # Cicla sugli argomenti effettivamente passati
            for param, value in passed_args.items():
                query = query + " {param} = {value}"
                
        df = pd.read_sql_query(query, self.connection)
        return df