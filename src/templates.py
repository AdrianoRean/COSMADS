TEMPLATE_WITH_DOCUMENT_RETRY = """
You are a proficient python developer have to patch python functions that solve natural language query. 
The python functions always return a list of dictionaries (in some cases the list may contain a single dictionary).

You are given the query, function and previous result under study.
Query:
======
{query}
======
Function:
======
{pipeline}
======
Function result/error:
======
{output}
======

Your goal is to determine the causes of the code exception.
If they are not correct, you should provide a patched version of the function that correctly respond to the query and write "WRONG" and the reasons before the ``` and ``` delimiters.

The python function that correctly generates the data specified in the queries should use at least one of the following tools:
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
- The result you have been provided will be a runtime exception.
- Generate the correct the function within the ```python and ``` delimiters after the "Answer:" line.
- Always end the script with a newline character and a triple backtick (```). It is important that after the return statement there is a newline character, followed by a triple backtick (```). 
- Before the ``` and ``` delimiters, state "WRONG"  and the causes of exception.
- Do not add any other information between the return statement and the triple backtick (```).
- The python function should return a list of dictionaries (in some cases the list may contain a single dictionary) as specified in the output schema of the problem statement.
- The python function should use the available tools to answer the query.
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

TEMPLATE_WITH_DOCUMENT_TABLES = """
You are a proficient python developer that generates a python function that solves a natural language query. The python function always returns a list of dictionaries (in some cases the list may contain a single dictionary).

You are given a query and a pipeline as inputs.
Query:
======
{query}
======
Pipeline:
======
{pipeline}
======

Your goal is concur into generating a piece valid Python function that correctly generates the data specified in the queries, using the following tools:
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
======
    
You have to choose between three actions:
- STOP: The function already solve the problem, so there only need to return the data and close the function.
- RETRIEVE: The function need to retrieve additional data though the tools.
- STANDARD: The function need to perfom some variable declaration and/or transformation onto the data.

On the basis of such action, you will add to the pipeline code:
- STOP: Only the return statement of the function.
- RETRIEVE: Only the function calls needed to retrieve the new data.
- STANDARD: Only the lines of code needed to declare variables and/or transform the data.

For each such action, add an adequate comment to signal the start the new lines of codes.
You can only take one action.
======
    
You have been provided also some evidence to help you in your task.
======
{evidence}
======
notes:
- Evidence may be missing
- The evidence may be referring to other programming languages, like SQL or Java. You have to use only Python.
- The evidence is always useful, but be careful in using it as it is.

Guidelines:
- Choose a single action
- Only generate the code for the chosen action append it to the pipeline given in input.
- Generate the code within the ``` and ``` delimiters after the "Answer:" line.
- Be careful, there should be only a ``` and ``` delimiters pair.
- Before the ``` and ``` delimiters, state the <ACTION> chosen.
- The python pipeline should use the available tools to answer the query.
- The tools are already imported. Do not import them.
- You can define helper functions if necessary.
- The function should be generated with a fixed name, which is "pipeline_function".

Here an example of an input pipeline and a possible expected output.
======
Pipeline input example: "
        python
        def pipeline_function():

            #STANDARD
            locationID = (4, "MINOR")
            results = []

            employees = GetDataFromEmployee()
            employees.open_connection()
    "
Output example: "
        RETRIEVE
        ```
        python
        def pipeline_function():

            #STANDARD
            locationID = (4, "MINOR")
            results = []

            employees = GetDataFromEmployee()
            employees.open_connection()
            
            #RETRIEVE
            employee_df = employees.call(locationID=locationID)
        ```
    "
======

Here an example of a finalized pipeline that may help you in generating a new pipeline:
======
Query example: {example_query}
Pipeline example: {example_pipeline}
======

Answer:
"""

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
- The function should be generated with a fixed name, which is "pipeline_function".
- Use the threading module whenever required to parallelize the data collection process based on the problem statement and the tools available to you.

Answer:
"""