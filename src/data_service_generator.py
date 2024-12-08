import ast
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
import dotenv
import os
import json
import pandas as pd
import sqlite3

DATA_SERVICE_EXAMPLE = """
        "brief_description": "Data service that provides data in a dataframe format about employees, their personal data and jobs.",
        
        "detailed_description": 
        \"\"\"Data service that provides data in a dataframe format about employees, their personal data and jobs.
        Each data entry has the following attributes: ssn, lastname, firstname, hiredate, salary, gender, performance, positionID, locationID.
        The attribute "ssn" (which stands for social security number) is unique for each employee.
        The attribute "hiredate" has format "dd-mm-yy".
        The attriute "salary" is saved as strings and start with the prefix "US$" and contains "," to separate thousand.
        The attriute "gender" is saved as either "M" or "F".
        The attributes "positionID" and "locationID" are foreign keys to the position and location collections respectively.\"\"\",

        "useful_info": \"\"\"
        - You may select data trough any combination of this attributes. They are all optional.
        - For each attribute, you must specify which kind of operator you want to apply. You may specify: "EQUAL", "GREATER", "GREATER OR EQUAL", "MINOR", "MINOR OR EQUAL".
        - If all attributes are left undeclared, it returns all the available data.
        - The result of a call is a pandas dataframe, so you may order, project and group the result if needed.\"\"\",
        
        "usage_example": \"\"\"
        - If I want to obtain all the information from the employee with ssn 123 I can write:
        employeessn = ('123', "EQUAL")
        employees = GetDataFromEmployee()
        employees.open_connection()
        employee_df = employees.call(employeessn=employeessn)
        # assuming the result is a pandas dataframe
        print(position_df.shape)
        \"\"\",
        
        "input_parameters": ["ssn:str", "lastname:str", "firstname:str", "hiredate:str", "salary:str", "gender:str", "performance:str", "positionID:int", "locationID:int"],
        
        "output_values": ["employee_df:pandas.DataFrame"],
        
        "module": "employee"
"""

PROMPT_LLM_DESCRIPTION_GLOBAL = """
You are a proficient SQL and Python programmer. 
You are given in exam the SQL database "{database_name}" with some samples for each of its table.
Your goal is, for each table, to create a brief descrition, a detailed description and an example of a Python class which wraps a SQL selector operator on that specific table.

Here an example of such wrapper:
======
{data_service_example}
======
Each wrapper is represented by a JSON string having the following structure:
{{
    "name": <name>,
    "brief_description": <brief_description>,
    "detailed_description": <description>,
    "useful_info": <useful_info>,
    "usage_example": <usage_example>,
    "input_parameters": <input_parameters>,
    "output_values": <output_values>,
    "module": <module>
}}
where:
    - <name> is the name of the callable python class
    - <brief_description> is a string representing a brief description of the callable python class
    - <detailed_description> is a string representing a detailed description of the callable python class
    - <input_parameters> is the list of input parameters of the data service, separated by a comma. Each input parameter has the following structure <name>:<type> where <name> is the name of the input parameter and <type> is the type of the input parameter. 
    - <output_values> is the list of output values of the data service, separated by a comma. Each output value has the following structure <name>:<type> where <name> is the name of the output value and <type> is the type of the output value.
    - <module> is the module where the callable python class is defined. It is useful to get a sense of which physical or software component the callable python class is related to.
    
Here there is the list of tables of the database with their details:
======
{database_table_list}
======
Each element of the list has the following structure:
{{
    "table_name": <table_name>,
    "table_columns": <table_columns>,
    "table_primary_key": <table_primary_keys>,
    "table_foreign_keys": <table_foreign_keys>,
    "table_data_samples": <table_data_samples>
}}
The "table_foreign_keys" is a list where each element is a triplet with the following structure: [<table_column>, <referenced_column>, <referenced_table>]
The "table_data_samples" is pandas dataframe converted to string.

The output should be a JSON list where each of the element represent a table and has the following structure:
{{
    "table_name": <table_name>,
    "brief_description" : <brief_description>,
    "detailed_description" : <detailed_description>,
    "usage_example" : <usage_example>
}}

Guidelines:
- Do not provide any further text, only the JSON list you have been required.
- Generate the response within the ``` and ``` delimiters after the "Answer:" line.
"""

DATA_SERVICE_BIRD_TEMPLATE = """
import pandas as pd
import sqlite3
import inspect
from data_service_bird_automatic.utilities import selectOperator

class GetDataFromEmployee:
    database_location = "data_service_bird_automatic/train_databases/{database}/{database}.sqlite"
    connection = None
    description = {{
        "brief_description": \"\"\"{brief_description}\"\"\",
        
        "detailed_description": \"\"\"{detailed_description}\"\"\",
        
        "useful_info": \"\"\"
        - You may select data trough any combination of this attributes. They are all optional.
        - For each attribute, you must specify which kind of operator you want to apply. You may specify: "EQUAL", "GREATER", "GREATER OR EQUAL", "MINOR", "MINOR OR EQUAL".
        - If all attributes are left undeclared, it returns all the available data.
        - The result of a call is a pandas dataframe, so you may order, project and group the result if needed.\"\"\",

        "usage_example": \"\"\"{example_usage}\"\"\",
        
        "input_parameters": {input_parameters},
        
        "output_values": ["{table_name}_df:pandas.DataFrame"],
        
        "module": "{table_name}"
    }}
    
    def open_connection(self):
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, {call_parameters}) -> pd.DataFrame:
        # Ottieni la firma della funzione
        signature = inspect.signature(self.call)
        parameters = list(signature.parameters.keys())
        
        # Crea un dizionario con i parametri passati alla funzione
        passed_args = {{k: v for k, v in locals().items() if v is not None and k in parameters}}
        
        if passed_args == {{}}:
            query = "SELECT * FROM {table_name}"
        else:    
            query = "SELECT * FROM {table_name} WHERE"
            # Cicla sugli argomenti effettivamente passati
            for param, (value, operator) in passed_args.items():
                operator = selectOperator(operator)
                if type(value) == str:
                    query = query + f" {{param}} {{operator}} '{{value}}' and"
                else:
                    query = query + f" {{param}} {{operator}} {{value}} and"
            query = query[:-4]
                
        df = pd.read_sql_query(query, self.connection)
        return df
"""

class CustomOutputParser(BaseOutputParser):
    """The output parser for the LLM."""
    def parse(self, text: str) -> str:
        text = text.strip("\n")
        text = text.strip()
        # count how many ``` are in the text
        back_count = text.count("```")
        if back_count != 2:
            print(text)
            raise ValueError("The string should contain exactly two triple backticks")
        code = text.split("```")[1]
        #print(code)
        return code

class ChainGeneratorAgent:
    def __init__(self, openai_key, mode):
        """Initialize the agent."""
        if mode == "description":
            prompt_template = PROMPT_LLM_DESCRIPTION_GLOBAL
        elif mode == "data_service":
            prompt_template = DATA_SERVICE_BIRD_TEMPLATE
        else: 
            raise ValueError(f"Mode {mode} is not recognized.")
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # define the LLM
        self.llm = ChatOpenAI(model="gpt-4o",
                              api_key=openai_key,
                              temperature=0.0)
        # define the output parser
        self.output_parser = CustomOutputParser()

    def get_chain(self):
        # generate the python function
        agent_chain = self.prompt | self.llm | self.output_parser
        return agent_chain
    
class DataServiceGenerator:
    def __init__(self):
        dotenv.load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    def get_sample_data(self, database_location, table):
        connection = sqlite3.connect(database_location)
        query = f"SELECT * FROM {table} LIMIT 10"
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df

    def create_data_services(self, database, database_location):
        database_name = database["db_id"]
        database_table_names = database["table_names"]
        database_columns = database["column_names"]
        database_types = database["column_types"]
        database_primary_keys = database["primary_keys"]
        database_foreign_keys = database["foreign_keys"]
        
        #print(f"********************** \n Database {database_name}")
        
        database_table_list = []
        
        tables_pair = []
        id = 0
        table_start = 1
        for index, column in enumerate(database_columns):
            if column[0] > id :
                tables_pair.append([id, database_table_names[id], table_start, index - 1])
                id += 1
                table_start = index
        tables_pair.append([id, database_table_names[id], table_start, index])
        
        #print(f"[DEBUG] Table pairs {tables_pair}")
        
        for table in tables_pair:
            #print(f"------------------ \n Table {table}")
            #Creating inputs list types
            table_inputs = "["
            table_parameters = ""
            for index in range(table[2], len(database_columns)):
                if table[0] == database_columns[index][0] and index != len(database_columns) - 1:
                    table_inputs += f"\"{database_columns[index][1]}:{database_types[index-1]}\", "
                    table_parameters += f"{database_columns[index][1]} = None, "
                else:
                    table_inputs = table_inputs[:-2] + "]"
                    table_parameters = table_parameters[:-2]
                    break            
            #Signaling primary keys
            primary_key_text = ""
            primary_keys = [i for i in database_primary_keys if i >= table[2] and i < index]
            if primary_keys != []:
                primary_key = database_columns[primary_keys[0]][1]
                primary_key_text = f"The attribute {primary_key} is unique.\n"
            else:
                primary_key_text = "No attribute is unique.\n"
            
            #Signaling foreign keys
            foreign_key_text = ""
            foreign_key_pairs = []
            foreign_keys_list = [i for i in database_foreign_keys if i[0] >= table[2] and i[0] < index]
            if foreign_keys_list != []:
                for foreign_key in foreign_keys_list:
                    refenced_table = [j[1] for j in tables_pair if j[2] <= foreign_key[1] <= j[3]]
                    foreign_key_text += f"The attribute {database_columns[foreign_key[0]][1]} is a foreign key to attribute {database_columns[foreign_key[1]][1]} of collection {refenced_table[0]}\n"
                    foreign_key_pairs.append([database_columns[foreign_key[0]][1], database_columns[foreign_key[1]][1], refenced_table[0]])
            else:
                foreign_key_text = "No attribute reference other collections.\n"
        
            #print(f"Table inputs {table_inputs}")
            #print(f"Table parameters {table_parameters}")
            #print(f"Primary Keys {primary_key_text}")
            #print(f"Foreign Keys {foreign_key_text}")
            
            data_samples = self.get_sample_data(database_location, table[1]).to_string()
            
            database_table_list.append({
                "table_name" : table[1],
                "table_columns" : table_inputs,
                "table_primary_key" : primary_key,
                "table_foreign_keys" : foreign_key_pairs,
                "table_parameters" : table_parameters,
                "table_data_samples" : data_samples
            })
            
        #print(database_table_list)
        #print("\n\n\n ----------------- \n\n\n")
            
        #Get LLM description
        generator_chain_output = {
            "output": ChainGeneratorAgent(self.OPENAI_API_KEY,mode="description").get_chain(),
            "inputs": RunnablePassthrough()
            }
        
        chain = (
            RunnableLambda( 
                    lambda x: {
                        "database_name": x[0],
                        "database_table_list": x[1],
                        "data_service_example": x[2]
                    }
                )
                | generator_chain_output
        )
        
        result = chain.invoke((database_name, database_table_list, DATA_SERVICE_EXAMPLE))["output"]
        result = ast.literal_eval(result)
        
        dir_location = f"data_service_bird_automatic/train_databases/{database_name}/data_services"
        
        os.makedirs(f"data_service_bird_automatic/train_databases/{database_name}/data_services", exist_ok=True) 
        
        for index, table in enumerate(database_table_list):
            file_location = dir_location + f"/{table['table_name']}.py"
            output = result[index]
            
            filled_template = DATA_SERVICE_BIRD_TEMPLATE.format(
                database = database_name,
                brief_description = output["brief_description"],
                detailed_description = output["detailed_description"],
                example_usage = output["usage_example"],
                input_parameters = table["table_columns"],
                table_name = table["table_name"],
                call_parameters = table["table_parameters"]
            )
            with open(file_location, "w") as f:
                f.write(filled_template)
    
if __name__ == "__main__":
    databases_description_location = "data_service_bird_automatic/train_databases/train_tables.json"
    databases = None
    with open(databases_description_location) as f:
        databases = json.load(f)
    
    database = databases[0]
    database_location = f"data_service_bird_automatic/train_databases/{database['db_id']}/{database['db_id']}.sqlite"
    print(database['db_id'])
    
    generator = DataServiceGenerator()
    generator.create_data_services(database, database_location)
    
    