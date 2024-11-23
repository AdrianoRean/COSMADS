import pandas as pd
import sqlite3
import inspect
from data_service_bird.utilities import selectOperator

class GetDataFromEmployee:
    database_location = "data_service_bird/human_resources/human_resources.sqlite"
    connection = None
    description = {
        "brief_description": "Data service that provides data in a dataframe format about employees, their personal data and jobs.",
        "detailed_description": 
        """Data service that provides data in a dataframe format about employees, their personal data and jobs.
        Each data entry has the following attributes: ssn, lastname, firstname, hiredate, salary, gender, performance, positionID, locationID.
        The attribute "ssn" (which stands for social security number) is unique for each employee.
        The attribute "hiredate" has format "dd-mm-yy".
        The attriute "salary" is saved as strings and start with the prefix "US$" and contains "," to separate thousand.
        The attriute "gender" is saved as either "M" or "F".
        The attributes "positionID" and "locationID" are foreign keys to the position and location collections respectively.
        You may select data trough any combination of this attributes. They are all optional.
        For each attribute, you must specify which kind of operator you want to apply. You may specify: "EQUAL", "GREATER", "GREATER OR EQUAL", "MINOR", "MINOR OR EQUAL".
        If all attributes are left undeclared, it returns all the available data.

        Example usage:
        - If I want to obtain all the information from the employee with ssn 123 I can write:
        employeessn = ('123', "EQUAL")
        employees = GetDataFromEmployee()
        employees.open_connection()
        employee_df = employees.call(employeessn=employeessn)
        # assuming the result is a pandas dataframe
        print(position_df.shape)

        Things to keep in mind:
        - The frame is a pandas dataframe, so you may order, project and group the result if needed.""",
        "input_parameters": ["ssn:str", "lastname:str", "firstname:str", "hiredate:str", "salary:str", "gender:str", "performance:str", "positionID:int", "locationID:int"],
        "output_values": ["employee_df:pandas.DataFrame"],
        "module": "employee"
    }
    
    def open_connection(self):
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, ssn = None, lastname = None, firstname = None, hiredate = None, salary = None, gender = None, performance = None, positionID = None, locationID = None) -> pd.DataFrame:
        # Ottieni la firma della funzione
        signature = inspect.signature(self.call)
        parameters = list(signature.parameters.keys())
        
        # Crea un dizionario con i parametri passati alla funzione
        passed_args = {k: v for k, v in locals().items() if v is not None and k in parameters}
        
        if passed_args == {}:
            query = "SELECT * FROM employee"
        else:    
            query = "SELECT * FROM employee WHERE"    
            # Cicla sugli argomenti effettivamente passati
            for param, (value, operator) in passed_args.items():
                operator = selectOperator(operator)
                if type(value) == str:
                    query = query + f" {param} {operator} '{value}'"
                else:
                    query = query + f" {param} {operator} {value}"
                
        df = pd.read_sql_query(query, self.connection)
        return df
    
    
'''
#if run as main, add 'src/' to db path

gd = GetDataFromEmployee()

gd.open_connection()
    
print(gd.call(positionID=(3, "GREATER OR EQUAL")))
'''