from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

TEMPLATE_WITH_DOCUMENT_PRE_REASONING = """
You are a proficient python developer that generates a python function that solves a natural language query. The python function always returns a list of dictionaries (in some cases the list may contain a single dictionary).

You are given the query under study.
Query:
======
{query}
======

Your goal is to determine which of the following tools should be used to generate a valid Python function that correctly generates the data specified in the queries:
======
{data_services}
======
Each tool is represented by a JSON string having the following structure:
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
    
You have been provided also some evidence to help you in your task.
======
{evidence}
======
notes:
- Evidence may be missing
- The evidence may be referring to other programming languages, like SQL or Java. You have only to suggest Python advices.
- The evidence is always useful, but be careful in using it as it is.

Guidelines:
- Make sure to generate a correct and concise response.
- Write down just a json dictionary of the tools you would use, explaining shortly the motivation they are needed and suggestions on how they should be used.
- The dictionary should contain also a special field called "general suggestions", which includes your general idea on how to solve the problem.
- The dictionary should have a format like whis: {{ "general_suggestions" : <general_suggestions>, "tools" : [ {{ "tool_name" : <tool_name>, "motivation" : <motivation> , "suggestions" : <suggestions> }} ] }} .
- The suggestions should be highly technical and documentation-like, so that another programmer may generate code efficiently and without errors.
- Before and after the dictionary, always put a newline character and a triple backtick (```).
- Do not add any other information between the dictionary and the triple backtick (```).

Answer:
"""

TEMPLATE_WITH_DOCUMENT_POST_REASONING = """
You are a proficient python developer that generates a python function that solves a natural language query. The python function always returns a list of dictionaries (in some cases the list may contain a single dictionary).

You are given the query under study.
Query:
======
{query}
======

Your goal is to generate a valid Python function that correctly generates the data specified in the queries, using the following tools:
======
{data_services}
======
Each tool is represented by a JSON string having the following structure:
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
    
You have been provided with useful and precise advices. Use them to produce a functional and correct code.
======
{advices}
======
The advices is represented by a JSON dictionary having the following structure:
{{
    "general_suggestion": <general_suggestion>,
    "tools": 
        [
            {{ 
                "tool_name" : <tool_name>,
                "motivation" : <motivation>, 
                "suggestions" : <suggestions>
            }}
        ]
}}
where:
    - <general_suggestions> are useful information to solve the query
    - <tools> is the list of tools you should use to produce the code. The number of tools contained in that list is variable.
    - each tool in <tools> have the same format.
    - <tool_name> is the name of a tool
    - <motivation> is the reason why that tool is needed
    - <suggestions> are a guide on how to use that tool. That it in serious consideration when generating the code.
    
You have been provided also some evidence to help you in your task.
======
{evidence}
======
notes:
- Evidence may be missing
- The evidence may be referring to other programming languages, like SQL or Java. You have to use only Python.
- The evidence is always useful, but be careful in using it as it is.

Guidelines:
- Make sure to generate a correct and concise python function.
- Generate the function within the ``` and ``` delimiters after the "Answer:" line.
- Always end the script with a newline character and a triple backtick (```). It is important that after the return statement there is a newline character, followed by a triple backtick (```). 
- Do not add any other information between the return statement and the triple backtick (```).
- The python function should return a list of dictionaries (in some cases the list may contain a single dictionary) as specified in the output schema of the problem statement.
- The python function should use the available tools to answer the query. To invoke a tool, just call the class name of the tool and pass the input parameters to it, e.g. ToolName(input_parameter1=input_value1, input_parameter2=input_value2, ...).
- The tools are already imported. Do not import them.
- You can define helper functions if necessary.
- The function should be generated with a fixed name, which is "pipeline_function".

Here an example of a pipeline that may help you in generating a new pipeline:
======
Query: {example_query}
Pipeline: {example_pipeline}
======

Answer:
"""

TEMPLATE_WITH_DOCUMENT = """
You are a proficient python developer that generates a python function that solves a natural language query. The python function always returns a list of dictionaries (in some cases the list may contain a single dictionary).

You are given the query under study.
Query:
======
{query}
======

Your goal is to generate a valid Python function that correctly generates the data specified in the queries, using the following tools:
======
{data_services}
======
Each tool is represented by a JSON string having the following structure:
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
    
You have been provided also some evidence to help you in your task.
======
{evidence}
======
notes:
- Evidence may be missing
- The evidence may be referring to other programming languages, like SQL or Java. You have only to suggest Python advices.
- The evidence is always useful, but be careful in using it as it is.

Guidelines:
- Make sure to generate a correct and concise python function.
- Generate the function within the ```python and ``` delimiters after the "Answer:" line.
- Always end the script with a newline character and a triple backtick (```). It is important that after the return statement there is a newline character, followed by a triple backtick (```). 
- Do not add any other information between the return statement and the triple backtick (```).
- The python function should return a list of dictionaries (in some cases the list may contain a single dictionary) as specified in the output schema of the problem statement.
- The python function should use the available tools to answer the query. To invoke a tool, just call the class name of the tool and pass the input parameters to it, e.g. ToolName(input_parameter1=input_value1, input_parameter2=input_value2, ...).
- The tools are already imported. Do not import them.
- You can define helper functions if necessary.
- The function should be generated with a fixed name, which is "pipeline_function".

Here an example of a pipeline that may help you in generating a new pipeline:
======
Query: {example_query}
Pipeline: {example_pipeline}
======

Answer:
"""


TEMPLATE_WITHOUT_PIPELINE = """
You are a proficient python developer that generates a python function that solves a natural language query. The python function always returns a list of dictionaries (in some cases the list may contain a single dictionary).

You are given the query under study.
Query:
======
{query}
======

Your goal is to generate a valid Python function that correctly generates the data specified in the queries, using the following tools:
======
{data_services}
======
Each tool is represented by a JSON string having the following structure:
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

Guidelines:
- Make sure to generate a correct and concise python function.
- Generate the function within the ```python and ``` delimiters after the "Answer:" line.
- Always end the script with a newline character and a triple backtick (```). It is important that after the return statement there is a newline character, followed by a triple backtick (```). 
- Do not add any other information between the return statement and the triple backtick (```).
- The python function should return a list of dictionaries (in some cases the list may contain a single dictionary) as specified in the output schema of the problem statement.
- The python function should use the available tools to answer the query. To invoke a tool, just call the class name of the tool and pass the input parameters to it, e.g. ToolName(input_parameter1=input_value1, input_parameter2=input_value2, ...).
- Import the correct tools from the correct modules to use them in the python function. The modules are in the database folder. To import a tool, use the following syntax: from data_services.<module> import <name>.
- You can define helper functions if necessary.
- The function should be generated with a fixed name, which is "pipeline_function".
- Use the threading module whenever required to parallelize the data collection process based on the problem statement and the tools available to you.

Answer:
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


class PipelineGeneratorAgent:
    """The agent that designs the pipeline."""

    def __init__(self, openai_key, mode = "standard"):
        """Initialize the agent."""
        # define the prompt
        if mode == "standard":
            prompt_template = TEMPLATE_WITH_DOCUMENT
        elif mode == "chain_of_thoughs_post":
            prompt_template = TEMPLATE_WITH_DOCUMENT_POST_REASONING
        elif mode == "chain_of_thoughs":
            prompt_template = TEMPLATE_WITH_DOCUMENT_PRE_REASONING
        elif mode == "wo_pipeline":
            prompt_template = TEMPLATE_WITHOUT_PIPELINE
        elif mode == "wrong":
            prompt_template = TEMPLATE_WITH_DOCUMENT
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


if __name__ == "__main__":
    generator = PipelineGeneratorAgent()
    chain = generator.get_chain()
    
    query ={
        "query": "Generate a table containing the max speed of the diemachine with id 25 over a time span of 30 seconds.",
        "data_services": []
    }

    result = chain.invoke(query)
    print(result)