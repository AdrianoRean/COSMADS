import sys
import json
from pathlib import Path
import dotenv
import os
import ast
import glob
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

# append the path to the parent directory to the system path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline_manager_db import PipelineManagerDB
from pipeline_chain import PipelineGeneratorAgent
from runner_chain import PipelineRunner
from document_manager_db import DocumentManagerDB

INTERMEDIATE_RESULTS_FILEPATH = Path(__file__).parent / "temp_pipeline.py"

class LLMAgent:
    def __init__(self, model, mode = "standard"):
        dotenv.load_dotenv()
        print(mode)

        self.model = model

        self.pipeline_manager = PipelineManagerDB(self.model)
        self.document_manager = DocumentManagerDB()

        self.generator = PipelineGeneratorAgent(self.model, mode=mode)
        self.runner = PipelineRunner()

        self.ds_directory = "data_services"
        self.doc_directory = "documents"
        self.current_production = "cardboard_production"
        self.sep: str = " - "

    def get_example(self, res_search):
        simil_query = res_search.page_content
        pipeline_id = res_search.metadata["pipeline"]
        pipeline_text = open(pipeline_id).read()
        return [simil_query, pipeline_text]
    
    def get_example_wrong(self, res_search):
        simil_query = res_search.page_content
        pipeline_id = res_search.metadata["pipeline"]
        if pipeline_id == "pipelines/q0.py":
            pipeline_id = "pipelines/q4.py"
        elif pipeline_id == "pipelines/q1.py":
            pipeline_id = "pipelines/q4.py"
        elif pipeline_id == "pipelines/q2.py":
            pipeline_id = "pipelines/q0.py"
        elif pipeline_id == "pipelines/q3.py":
            pipeline_id = "pipelines/q0.py"
        elif pipeline_id == "pipelines/q4.py":
            pipeline_id = "pipelines/q0.py"
        pipeline_text = open(pipeline_id).read()
        return [simil_query, pipeline_text]
    
    def convert_data_service_to_document(self, data_service_doc: dict) -> str:
        document = data_service_doc
        document_str = self.sep.join([f"{key}: {value}" for key, value in document.items()])
        return document_str
    
    def get_data_services(self):
        """ pipeline_text = self.get_example(res_search)[1]
        data_services = ""
        data_services_list = []
        for line in pipeline_text.split("\n"):
            if f"from {self.ds_directory}." in line:
                module_ds = line.split(f"from {self.ds_directory}.")[1].split(" import ")[0]
                name_ds = line.split(f"from {self.ds_directory}.")[1].split(" import ")[1]
                with open(f"{self.ds_directory}/{module_ds}.py", mode="r") as f:
                    content = f.read()
                    tree = ast.parse(content)
                    class_obj = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name_ds][0]
                    body = class_obj.body
                    description = [node for node in body if isinstance(node, ast.Assign) and node.targets[0].id == "description"]
                    description_value = description[0].value
                    description_dict = ast.literal_eval(description_value)
                    description_dict["class_name"] = name_ds
                    data_services += self.convert_data_service_to_document(description_dict)   # data services for prompt
                    data_services_list.append(description_dict)    # data services for saving pipeline """
        data_services_all = glob.glob(f"{self.ds_directory}/*.py")
        data_services = ""
        data_services_list = []
        for data_service in data_services_all:
            with open(f"{data_service}", mode="r") as f:
                content = f.read()
                tree = ast.parse(content)
                class_objs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
                for class_obj in class_objs:
                    name_ds = class_obj.name
                    body = class_obj.body
                    description = [node for node in body if isinstance(node, ast.Assign) and node.targets[0].id == "description"]
                    description_value = description[0].value
                    description_dict = ast.literal_eval(description_value)
                    description_dict["class_name"] = name_ds
                    data_services += self.convert_data_service_to_document(description_dict)   # data services for prompt
                    data_services_list.append(description_dict)    # data services for saving pipeline
        return data_services, data_services_list
    
    def get_relevant_document(self, query):
        document = self.document_manager.extract_document(query)
        return document
    
    def convert_data_service_to_document(self, data_service_doc: dict) -> str:
        document = data_service_doc
        document_str = self.sep.join([f"{key}: {value}" for key, value in document.items()])
        return document_str

    def save_intermediate_result_to_json(self, pipeline, data_services) -> str:
        file_to_save = ""

        for data_service in data_services:
            module = data_service['module']
            class_name = data_service['class_name']
            file_to_save += f"from {self.ds_directory}.{module} import {class_name}\n"
        
        file_to_save += f"{pipeline}\n"

        main_function = f"""
if __name__ == "__main__":
    result = pipeline_function()
    import json
    import pandas as pd
    with open("result.json", "w") as f:
        json.dump(result, f, indent=4)
    result = pd.DataFrame(result)
    from tabulate import tabulate
    result = tabulate(result, headers='keys', tablefmt='psql')
    print(result)
    """
        file_to_save += main_function

        with open(INTERMEDIATE_RESULTS_FILEPATH, "w") as f:
            f.write(file_to_save)

    def get_chain(self) -> Runnable:

        generator_chain = self.generator.get_chain()
        runner_chain = self.runner.get_chain()

        generator_chain_output = {
            "pipeline": generator_chain,
            "inputs": RunnablePassthrough()
        }

        runner_chain_output = {
            "output": runner_chain,
            "inputs": RunnablePassthrough()
        }

        chain = (
            RunnableLambda(lambda x: {
                "query": x,
                "pipeline_search": self.pipeline_manager.pipeline_store.search(x),
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "example": self.get_example(x["pipeline_search"]["output"]),
                    "data_services": self.get_data_services()
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "data_services": x["data_services"][0],
                    "data_services_list": x["data_services"][1],
                    "example_query": x["example"][0],
                    "example_pipeline": x["example"][1],
                }
            )
            | generator_chain_output
            | RunnableParallel(
                gen = RunnableLambda(lambda x: {
                    "query": x["inputs"]["query"],
                    "data_services": x["inputs"]["data_services"],
                    "example_query": x["inputs"]["example_query"],
                    "example_pipeline": x["inputs"]["example_pipeline"],
                    "pipeline": x["pipeline"]
                }),
                exe = RunnableLambda(lambda x:
                    self.save_intermediate_result_to_json(x["pipeline"], x["inputs"]["data_services_list"])
                )
            )
            | RunnableLambda(lambda x: {
                "inputs": x,
                "pipeline_filepath": str(INTERMEDIATE_RESULTS_FILEPATH)
            })
            | RunnableParallel(
                inputs = RunnableLambda(lambda x: {
                    "query": x["inputs"]["gen"]["query"],
                    "data_services": x["inputs"]["gen"]["data_services"],
                    "example_query": x["inputs"]["gen"]["example_query"],
                    "example_pipeline": x["inputs"]["gen"]["example_pipeline"],
                    "pipeline": x["inputs"]["gen"]["pipeline"],
                }),
                output = runner_chain_output
            )
            | RunnableLambda(lambda x: {
                "query": x["inputs"]["query"],
                "data_services": x["inputs"]["data_services"],
                "example_query": x["inputs"]["example_query"],
                "example_pipeline": x["inputs"]["example_pipeline"],
                "pipeline": x["inputs"]["pipeline"],
                "output": x["output"]["output"],
            })
        )

        # return the chain
        return chain
    
    def get_chain_wrong(self) -> Runnable:
        generator_chain = self.generator.get_chain()
        runner_chain = self.runner.get_chain()

        generator_chain_output = {
            "pipeline": generator_chain,
            "inputs": RunnablePassthrough()
        }

        runner_chain_output = {
            "output": runner_chain,
            "inputs": RunnablePassthrough()
        }

        chain = (
            RunnableLambda(lambda x: {
                "query": x,
                "pipeline_search": self.pipeline_manager.pipeline_store.search(x),
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "example": self.get_example_wrong(x["pipeline_search"]["output"]),
                    "data_services": self.get_data_services()
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "data_services": x["data_services"][0],
                    "data_services_list": x["data_services"][1],
                    "example_query": x["example"][0],
                    "example_pipeline": x["example"][1],
                }
            )
            | generator_chain_output
            | RunnableParallel(
                gen = RunnableLambda(lambda x: {
                    "query": x["inputs"]["query"],
                    "data_services": x["inputs"]["data_services"],
                    "example_query": x["inputs"]["example_query"],
                    "example_pipeline": x["inputs"]["example_pipeline"],
                    "pipeline": x["pipeline"]
                }),
                exe = RunnableLambda(lambda x:
                    self.save_intermediate_result_to_json(x["pipeline"], x["inputs"]["data_services_list"])
                )
            )
            | RunnableLambda(lambda x: {
                "inputs": x,
                "pipeline_filepath": str(INTERMEDIATE_RESULTS_FILEPATH)
            })
            | RunnableParallel(
                inputs = RunnableLambda(lambda x: {
                    "query": x["inputs"]["gen"]["query"],
                    "data_services": x["inputs"]["gen"]["data_services"],
                    "example_query": x["inputs"]["gen"]["example_query"],
                    "example_pipeline": x["inputs"]["gen"]["example_pipeline"],
                    "pipeline": x["inputs"]["gen"]["pipeline"],
                }),
                output = runner_chain_output
            )
            | RunnableLambda(lambda x: {
                "query": x["inputs"]["query"],
                "data_services": x["inputs"]["data_services"],
                "example_query": x["inputs"]["example_query"],
                "example_pipeline": x["inputs"]["example_pipeline"],
                "pipeline": x["inputs"]["pipeline"],
                "output": x["output"]["output"],
            })
        )

        # return the chain
        return chain


if __name__ == "__main__":
    q = "q3"
    model = "llama3.2"
    #model = "mistral"
    llm = LLMAgent(model)
    with open("queries_pipelines.json", "r") as f:
        queries = json.load(f)
    query = queries[q]["query"]
    print(query)
    result = llm.get_chain().invoke(query)
    print(result["output"])