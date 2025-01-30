DATA_SERVICE_SECTION = """
Each tool is represented by a JSON string having the following structure:
{{
    "name": <name>,
    "brief_description": <brief_description>,
    "detailed_description": <description>,
    "usage_example": <usage_example>,
    "input_parameters": <input_parameters>,
    "output_values": <output_values>,
    "module": <module>
}}
where:
    - <name> is the name of the callable python class
    - <brief_description> is a string representing a brief description of the callable python class
    - <detailed_description> is a string representing a detailed description of the callable python class
    - <usage_example> is a multiline string which contains an example on how to use the tool
    - <input_parameters> is the list of input parameters of the data service, separated by a comma. Each input parameter has the following structure <name>:<type> where <name> is the name of the input parameter and <type> is the type of the input parameter. 
    - <output_values> is the list of output values of the data service, separated by a comma. Each output value has the following structure <name>:<type> where <name> is the name of the output value and <type> is the type of the output value.
    - <module> is the module where the callable python class is defined. It is useful to get a sense of which physical or software component the callable python class is related to.

All tools share the following useful informations:
    - You may select data trough any combination of this attributes. They are all optional.
    - For each attribute, you must specify which kind of operator you want to apply. You may specify: "EQUAL", "GREATER", "GREATER OR EQUAL", "MINOR", "MINOR OR EQUAL".
    - If all attributes are left undeclared, it returns all the available data.
    - You cannot pass a list as value for the attributes.
    - Sometimes data may have missing values.
    - The result of a call is a pandas dataframe, so you may order, project and group the result if needed.
"""

TEMPLATE_WITH_DOCUMENT_TOOLS = """
You are a proficient python developer is able to code a python function that solves a natural language query.

You are given the query under study.
Query:
======
{query}
======

Your goal is to determine which of the following tools should be used to generate a valid Python function that correctly generates the data specified in the queries:
======
{data_services}
======
{DATA_SERVICE_SECTION}
    
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
- Write down just a list of the tools you would use.
- The list should have a format like this: ["tool1_module", "tool2_module", ...].
- Before and after the dictionary, always put a newline character and a triple backtick (```).
- Do not add any other information between the dictionary and the triple backtick (```).

Answer:
"""

TEMPLATE_WITHOUT_PIPELINE_BUT_VIEW = """
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
{DATA_SERVICE_SECTION}

You have been provided also some evidence to help you in your task.
======
{evidence}
======
notes:
- Evidence may be missing
- The evidence may be referring to other programming languages, like SQL or Java. You have only to suggest Python advices.
- The evidence is always useful, but be careful in using it as it is.

You have been provided some samples from the data_services to help you in your task.
======
{data_samples}
======
The data samples have the following structure:
[{{ "table_name" : <table_name>, "table_data_samples" : <table_data_samples> }}, ... , {{ "table_name" : <table_name>, "table_data_samples" : <table_data_samples> }}]
notes:
- the table_data_samples are pandas dataframes converted to string.

Guidelines:
- Make sure to generate a correct and concise python function.
- Generate the function within the ```python and ``` delimiters after the "Answer:" line.
- Always end the script with a newline character and a triple backtick (```). It is important that after the return statement there is a newline character, followed by a triple backtick (```). 
- Do not add any other information between the return statement and the triple backtick (```).
- The python function should return a list of dictionaries (in some cases the list may contain a single dictionary) as specified in the output schema of the problem statement.
- The python function should use the available tools to answer the query. To invoke a tool, just call the class name of the tool and pass the input parameters to it, e.g. ToolName(input_parameter1=input_value1, input_parameter2=input_value2, ...).
- Import the correct tools from the correct modules to use them in the python function. The modules are in the database folder. To import a tool, use the following syntax: from data_services.<module> import <name>.
- You can define helper functions if necessary.
- If you use libraries, be sure to call the correct functions.
- If you need to parse strings or dates, be sure to parse them correctly (use information both from the tool description and the data samples to help you in these cases)
- Ensure that the final result is json serializable. 
- The function should be generated with a fixed name, which is "pipeline_function".

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
{DATA_SERVICE_SECTION}

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
- Import the correct tools from the correct modules to use them in the python function. The modules are in the database folder. To import a tool, use the following syntax: from data_services.<module> import <name>.
- You can define helper functions if necessary.
- Ensure that the final result is json serializable.
- The function should be generated with a fixed name, which is "pipeline_function".

Answer:
"""