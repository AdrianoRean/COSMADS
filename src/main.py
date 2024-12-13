import sys
import json
from pathlib import Path
import dotenv
import os
import ast
import glob
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch

# append the path to the parent directory to the system path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline_manager_db import PipelineManagerDB
from pipeline_chain import PipelineGeneratorAgent
from runner_chain import PipelineRunner
from document_manager_db import DocumentManagerDB

INTERMEDIATE_RESULTS_FILEPATH = Path(__file__).parent / "temp_pipeline.py"

class LLMAgent:
    def __init__(self, mode = "standard"):
        dotenv.load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        print(mode)
        self.mode = mode

        self.pipeline_manager = PipelineManagerDB(OPENAI_API_KEY)
        self.document_manager = DocumentManagerDB()
        
        if mode == "chain_of_thoughs":
            self.suggestor = PipelineGeneratorAgent(OPENAI_API_KEY, mode="chain_of_thoughs")
            self.generator = PipelineGeneratorAgent(OPENAI_API_KEY, mode="chain_of_thoughs_post")
        else:
            self.generator = PipelineGeneratorAgent(OPENAI_API_KEY, mode=mode)
        self.runner = PipelineRunner()

        #self.ds_directory = "data_services"
        self.ds_directory = "data_service_bird/human_resources"
        self.ds_directory_import = "data_service_bird.human_resources"
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
        
        modules = []
        classes = []
        for data_service in data_services:
            modules.append(data_service['module'])
            classes.append(data_service['class_name'])
        
        new_pipeline = ""
        for line in pipeline.split('\n'):
            if "from" in line and (any(module in line for module in modules) or any(m_class in line for m_class in classes)):
                continue
            else:
                new_pipeline += f"\n{line}"

        for data_service in data_services:
            module = data_service['module']
            class_name = data_service['class_name']
            file_to_save += f"from {self.ds_directory_import}.{module} import {class_name}\n"
        
        file_to_save += f"{new_pipeline}\n"

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
    
    def save_advice(self, advice):
        with open("advice.json", "w") as f:
            json.dump(json.loads(advice), f, indent=4)
            
    def chain_of_thoughs(self, x, generator_chain_output):
        suggestor_chain = self.suggestor.get_chain()
        suggestor_chain_output = {
            "pipeline": suggestor_chain,
            "inputs": RunnablePassthrough()
        }
        
        chain = ( 
                 suggestor_chain_output
                 |
                 RunnableParallel(
                    gen = RunnableLambda(lambda x: {
                        "query": x["inputs"]["query"],
                        "evidence": x["inputs"]["evidence"],
                        "data_services": x["inputs"]["data_services"],
                        "data_services_list": x["inputs"]["data_services_list"],
                        "example_query": x["inputs"]["example_query"],
                        "example_pipeline": x["inputs"]["example_pipeline"],
                        "advices" : x["pipeline"][1]
                    }),
                    exe = RunnableLambda(lambda x:
                        self.save_advice(x["pipeline"][1])
                    )
                )
                | RunnableLambda(lambda x: {
                        "query": x["gen"]["query"],
                        "evidence": x["gen"]["evidence"],
                        "data_services": x["gen"]["data_services"],
                        "data_services_list": x["gen"]["data_services_list"],
                        "example_query": x["gen"]["example_query"],
                        "example_pipeline": x["gen"]["example_pipeline"],
                        "advices" : x["gen"]["advices"]
                    })
                | generator_chain_output
        )
        
        return chain.invoke(x)
    
    def chain_of_tables(self, x, generator_chain_output):
        
        x["action"] = "NONE"
        x["pipeline"] = ""
        
        chain = ( 
                 generator_chain_output
                 | RunnableBranch(
                    (lambda x: x["pipeline"][0] != "STOP", lambda x: {
                        "query": x["inputs"]["query"],
                        "evidence": x["inputs"]["evidence"],
                        "data_services": x["inputs"]["data_services"],
                        "data_services_list": x["inputs"]["data_services_list"],
                        "example_query": x["inputs"]["example_query"],
                        "example_pipeline": x["inputs"]["example_pipeline"],
                        "pipeline" : x["pipeline"][1],
                        "action" : x["pipeline"][0]
                        }),
                    lambda x: {
                        "inputs": x["inputs"],
                        "action" : x["pipeline"][0]
                        }
                    )
        )
        
        while x["action"] != "STOP":   
            print("**********************")
            print(f"Last action is: {x['action'].strip()}")            
            print(f"Actual pipeline is: {x['pipeline']}")
            x = chain.invoke(x)
        
        return x
    
    def run_chain(self, x, generator_output_chain):
        return (
            RunnableLambda(
                lambda x: x)
            |
            generator_output_chain
            ).invoke(x)

    def get_chain(self) -> Runnable:
        
        print(self.mode)
        
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
                "query": x["query"],
                "evidence": x["evidence"],
                "pipeline_search": self.pipeline_manager.pipeline_store.search(x["query"]),
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "example": self.get_example(x["pipeline_search"]["output"]),
                    "data_services": self.get_data_services()
                }
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "data_services": x["data_services"][0],
                    "data_services_list": x["data_services"][1],
                    "example_query": x["example"][0],
                    "example_pipeline": x["example"][1],
                }
            )
            | RunnableBranch(
                (lambda x: self.mode == "chain_of_thoughs", lambda x: self.chain_of_thoughs(x, generator_chain_output)),
                (lambda x: self.mode == "chain_of_tables", lambda x: self.chain_of_tables(x, generator_chain_output)),
                lambda x: self.run_chain(x, generator_chain_output)
            )
            | RunnableParallel(
                gen = RunnableLambda(lambda x: {
                    "query": x["inputs"]["query"],
                    "evidence": x["inputs"]["evidence"],
                    "data_services": x["inputs"]["data_services"],
                    "example_query": x["inputs"]["example_query"],
                    "example_pipeline": x["inputs"]["example_pipeline"],
                    "pipeline": x["pipeline"][1].strip()[len("python"):].strip()
                }),
                exe = RunnableLambda(lambda x:
                    self.save_intermediate_result_to_json(x["pipeline"][1].strip()[len("python"):].strip(), x["inputs"]["data_services_list"])
                )
            )
            | RunnableLambda(lambda x: {
                "inputs": x,
                "pipeline_filepath": str(INTERMEDIATE_RESULTS_FILEPATH)
            })
            | RunnableParallel(
                inputs = RunnableLambda(lambda x: {
                    "query": x["inputs"]["gen"]["query"],
                    "evidence": x["inputs"]["gen"]["evidence"],
                    "data_services": x["inputs"]["gen"]["data_services"],
                    "example_query": x["inputs"]["gen"]["example_query"],
                    "example_pipeline": x["inputs"]["gen"]["example_pipeline"],
                    "pipeline": x["inputs"]["gen"]["pipeline"],
                }),
                output = runner_chain_output
            )
            | RunnableLambda(lambda x: {
                "query": x["inputs"]["query"],
                    "evidence": x["inputs"]["evidence"],
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
    q = "q5"
    llm = LLMAgent(mode="standard")
    with open("queries/queries_pipelines_human_resources.json", "r") as f:
        queries = json.load(f)
        
    input_file = {
        "query" : queries[q]["query"],
        "evidence" : queries[q]["evidence"]
    }
    print(queries[q]["query"])
    result = llm.get_chain().invoke(input_file)
    print(result["output"])