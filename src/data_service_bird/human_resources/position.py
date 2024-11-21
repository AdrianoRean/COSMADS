import pandas as pd
import sqlite3
import inspect
import os

class GetDataFromPosition:
    database_location = "data_service_bird/human_resources/human_resources.sqlite"
    connection = None
    description = {
        "brief_description": "Data service that provides data in a dataframe format about job positions, their requirements and salaries.",
        "detailed_description": 
        """Data service that provides data in a dataframe format about job positions, their requirements and salaries.
        Each data entry has the following attributes: positionID, positiontitle, educationrequired, minsalary, maxsalary.
        The attribute "positionID" is unique for each job position.
        You may select data trough any combination of this attributes. They are all optional.
        If all attributes are left undeclared, it returns all the available data.

        Example usage:
        - If I want to obtain all the information from the job position with positionID 123 I can write:
        positionID = 123
        positions = GetDataFromPosition()
        positions.open_connection()
        position_df = GetDataFromPosition.call(positionID=123)
        # assuming the result is a pandas dataframe
        print(position_df.shape)

        Things to keep in mind:
        - The frame is a pandas dataframe, so you may order, project and group the result if needed.""",
        "input_parameters": ["positionID:int", "positiontitle:str", "educationrequired:str", "minsalary:str", "maxsalary:int"],
        "output_values": ["position_df:pandas.DataFrame"],
        "module": " position"
    }
    
    def open_connection(self):
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, positionID = None, positiontitle = None, educationrequired = None, minsalary = None, maxsalary = None) -> pd.DataFrame:
        # Ottieni la firma della funzione
        signature = inspect.signature(self.call)
        parameters = list(signature.parameters.keys())
        
        # Crea un dizionario con i parametri passati alla funzione
        passed_args = {k: v for k, v in locals().items() if v is not None and k in parameters}
        
        if passed_args == {}:
            query = "SELECT * FROM position"
        else:    
            query = "SELECT * FROM position WHERE"    
            # Cicla sugli argomenti effettivamente passati
            for param, value in passed_args.items():
                if type(value) == str:
                    query = query + f" {param} = '{value}'"
                else:
                    query = query + f" {param} = {value}"
                
        df = pd.read_sql_query(query, self.connection)
        return df
    
    
'''
#if run as main, add 'src/' to db path

gd = GetDataFromPosition()

print(os.getcwd())

gd.open_connection()
    
print(gd.call(positionID=3))
'''