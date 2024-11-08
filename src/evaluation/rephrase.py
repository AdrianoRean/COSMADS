from langchain_ollama import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
import dotenv

PROMPT_TEMPLATE = """
You are very proficient in rephrasing a given natural language query while preserving the original meaning. 
The query is in the context of industrial manufacturing and asks for data in tabular format. Do your best to preserve the underlying schema of the data the user is asking for, as well as the input parameters (if any), while rephrasing the query.
This means that the names of the input parameters and the schema can be also rephrased, but the underlying meaning should be preserved.
The output is a string containing the rephrased query.

Suggestions:
- Avoid attaching additional symbols before or after numerical/string identifiers. For example, "spindle number 1" should not be rephrased to "spindle #1" or "spindle n1" or "spindle 1st".
- Avoid repeating the "Response" label in the rephrased query.

I will now provide you with the query after the "Query:" label. Generate a rephrased query and provide it after the "Response:" label.
Query: {query}
Response: 
"""

class Rephraser:

    def __init__(self, model) -> None:
        self.model = model
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.model = ChatOllama(model=self.model,
                                temperature=1)
        self.output_parser = StrOutputParser()

    def get_chain(self):
        return self.prompt_template | self.model | self.output_parser
    
    def rephrase(self, query: str) -> str:
        return self.get_chain().invoke({"query": query})
    
if __name__ == "__main__":
    dotenv.load_dotenv()
    model = "llama3.2"
    #model = "mistral"
    rephraser = Rephraser(model)
    queries_json = json.load(open("../queries_pipelines.json"))
    n_repr = 20
    result = {}
    for query_id in queries_json.keys():
        query = queries_json[query_id]["query"]
        rephrased_queries = []
        for i in range(n_repr):
            rephrased_query = rephraser.rephrase(query)
            print(f"Original query: {query}")
            print(f"Rephrased query: {rephrased_query}")
            rephrased_queries.append(rephrased_query)
        result[query_id] = rephrased_queries
    with open("rephrased_queries.json", "w") as f:
        json.dump(result, f, indent=4)