import pandas as pd
import sqlite3
import inspect

class GetDataFromDatabase:
    database_location = None
    connection = None
    
    def open_connection(self, database_location):
        self.database_location = database_location
        self.connection = sqlite3.connect(self.database_location)
        
    def close_connection(self):
        self.connection.close()

    def call(self, query) -> pd.DataFrame:
        df = pd.read_sql_query(query, self.connection)
        return df