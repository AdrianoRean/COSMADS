from copy import deepcopy
import sys
import json
from pathlib import Path
import dotenv
import os
import ast
import glob
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableBranch
from data_service_generator import get_sample_data
import re
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from templates import DATA_SERVICE_SECTION

# append the path to the parent directory to the system path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline_chain import PipelineGeneratorAgent
from runner_chain import PipelineRunner

INTERMEDIATE_RESULTS_FILEPATH = Path(__file__).parent / "temp_pipeline.py"

def get_queries(database):
    all_queries = json.load(open(f"queries/train.json"))
    return [query for query in all_queries if query["db_id"] == database]

def extract_tables(sql_query):
    # Regular expressions to capture tables in FROM and JOIN clauses
    from_pattern = re.compile(r'FROM\s+([^\s,]+(?:\s*,\s*[^\s,]+)*)', re.IGNORECASE)
    join_pattern = re.compile(r'JOIN\s+([^\s]+)', re.IGNORECASE)
    
    # Extract tables from FROM clause, including comma-separated lists
    from_matches = from_pattern.findall(sql_query)
    if from_matches:
        # Split by commas if multiple tables are listed
        from_tables = [table.strip() for match in from_matches for table in match.split(',')]
    else:
        from_tables = []
    
    # Extract tables from JOIN clauses
    join_matches = join_pattern.findall(sql_query)
    
    # Combine and return unique table names
    all_tables = list(set(from_tables + join_matches))
    #clean = re.compile(r'[^a-zA-Z0-9, _]')
    #all_tables = [tb for tb in all_tables if clean.sub("", tb) != ""]
    return all_tables

class LLMAgent:
    def __init__(self, enterprise, model, pipeline_mode = "standard", evidence_mode = "standard_evidence", dataservice_mode = None, similarity_treshold = 0.8, automatic=False, database="human_resources", verbose = False):
        print(f"Mode is: {pipeline_mode}")
        self.pipeline_mode = pipeline_mode
        self.evidence_mode = evidence_mode
        self.dataservice_mode = dataservice_mode
        
        self.enterprise = enterprise
        self.model = model
        
        self.database = database
        self.automatic = automatic
        
        self.similarity_treshold = similarity_treshold
        self.verbose = verbose
        
        self.generator = PipelineGeneratorAgent(enterprise, model, mode=pipeline_mode, additional_mode = dataservice_mode)
        self.runner = PipelineRunner()

        #self.ds_directory = "data_services"
        safe_model = str(model.replace("-", "_"))
        if automatic:
            self.ds_directory = f"data_service_bird_automatic/train_databases/{database}/data_services/{enterprise}/{safe_model}"
            self.ds_directory_import = f"data_service_bird_automatic.train_databases.{database}.data_services.{enterprise}.{safe_model}"
        else:
            self.ds_directory = f"data_service_bird/{database}"
            self.ds_directory_import = f"data_service_bird.{database}"
            
        self.db_info_file_location = "data_service_bird_automatic/train_databases/train_tables.json"
        self.db_info_file = json.load(open(self.db_info_file_location))
        self.db_all_tables = [db["table_names"] for db in self.db_info_file if db["db_id"] == database][0]
        
        if self.verbose:
            print(f"Tables from json file are: {self.db_all_tables}")
            
        if self.verbose:
            print(f"Content of directory: {os.listdir(self.ds_directory)}")
            
        self.doc_directory = "documents"
        self.current_production = "cardboard_production"
        self.sep: str = " - "
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
    def bert_cosine_similarity(self, sentence1, sentence2):
        
        # Tokenize the sentences
        tokens1 = self.bert_tokenizer.tokenize(sentence1)
        tokens2 = self.bert_tokenizer.tokenize(sentence2)
        
        # Convert tokens to input IDs
        input_ids1 = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
        input_ids2 = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

        # Obtain the BERT embeddings
        with torch.no_grad():
            outputs1 = self.bert_model(input_ids1)
            outputs2 = self.bert_model(input_ids2)
            embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

        # Calculate similarity
        similarity_score = cosine_similarity(embeddings1, embeddings2)
        return similarity_score[0][0]
    
    def check_word_simliarity(self, desired_word, words_to_check, similarity_treshold = 0.8):
        best_matches = []
        for word in words_to_check:
            similarity = self.bert_cosine_similarity(desired_word, word)
            if similarity >= similarity_treshold:
                best_matches.append((word, similarity))
        return best_matches

    def correct_obvious_word_mistake(self, function_to_change, call_keys = [], hard_coded_words = [], similarity_treshold = 0.8):
        function_changed = deepcopy(function_to_change)
        
        #find all call keys in the function
        call_regex_string = re.compile(r"""
                                            call\( #anything inside a call()
                                            (.*)
                                        
                                        \)
                                    """, re.VERBOSE)
        
        call_matches = list(set(call_regex_string.findall(function_changed)))  
        
        call_match_dict = {} 
        inside_call_matches = []
        
        inside_call_regex_string = re.compile(r"""
                                            \b
                                            (\w+)
                                                \s*
                                                =
                                            """, re.VERBOSE)
        for index, match in enumerate(call_matches):
            temp_inside_matches = inside_call_regex_string.findall(match)
            inside_call_matches = inside_call_matches + temp_inside_matches
            for inside_match in temp_inside_matches:
                if inside_match not in call_match_dict:
                    call_match_dict[inside_match] = []
                call_match_dict[inside_match].append(index)
                
        if self.verbose:
            print(f"Words inside the call: {inside_call_matches}")
        
        #check for each possible value if there are matches
        for word in call_keys:
            best_matches = self.check_word_simliarity(word, inside_call_matches, similarity_treshold)
            if self.verbose:
                print(f"Key: {word} - Matches: {best_matches}")
            for match in best_matches:
                if match[1] == 1: #perfect match
                    if word == match[0]: #also cases are equal, no need to change
                        continue
                #imperfect match or case difference, probable typo mistake
                calls_indexes = call_match_dict[match[0]] #find calls to be replaced
                for call_index in calls_indexes:
                    call = call_matches[call_index] #retrieve single call
                    if self.verbose:
                        print(f"Key: {word} - calls : {call}")
                    new_call = call.replace(match[0], word) #fix it
                    call_matches[call_index] = new_call #save bettered version for future use
                    function_changed = function_changed.replace(call, new_call) #replace in actual function
                        
        #find all hard coded words in function
        hard_coded_regex_string = re.compile(r"\"([\w+\s]+)\"")
        hard_coded_matches = list(set(hard_coded_regex_string.findall(function_changed)))  
        if self.verbose:
            print(f"Hard coded words: {hard_coded_matches}")  
        #check for each possible value if there are matches
        for word in hard_coded_words:
            best_matches = self.check_word_simliarity(word, hard_coded_words, similarity_treshold)
            if self.verbose:
                print(f"Word: {word} - Matches: {best_matches}")
            for match in best_matches:
                if match[1] == 1: #perfect match
                    if word == match[0]: #also cases are equal, no need to change
                        continue
                #imperfect match or case difference, probable typo mistake
                specific_hard_coded_regex = re.compile(f"\"{match[0]}\"")
                function_changed = specific_hard_coded_regex.sub(f'"{word}"', function_changed)
                
        return function_changed
    
    def convert_data_service_to_document(self, data_service_doc: dict) -> str:
        document = data_service_doc
        document_str = self.sep.join([f"{key}: {value}" for key, value in document.items()])
        return document_str
    
    def get_data_services(self, sql = None):
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
        data_services_all = os.listdir(self.ds_directory)
        data_services_all = [ds[:-3] for ds in data_services_all if ds[0:2] != "__"]
        if self.verbose:
            print(f"Avaible dataservices are: {data_services_all}")
        data_services = ""
        data_services_list = []
        tables = []
        query_services_list = []
        call_parameters_list = []
        if sql != None:
            tables = extract_tables(sql)
            query_services_list = [tb.replace("_", "") for tb in tables]
            if self.verbose:
                print(f"Tables find from sql are: {tables}, As services_list: {query_services_list}")
            data_service_to_process = [tb for tb in query_services_list if tb in data_services_all]
            if self.verbose:
                print(f"Services to process: {data_service_to_process}")
        else:
            data_service_to_process = data_services_all
            tables = self.db_all_tables
        for data_service in data_service_to_process:
            with open(f"{self.ds_directory}/{data_service}.py", mode="r") as f:
                content = f.read()
                tree = ast.parse(content)
                class_objs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
                for class_obj in class_objs:
                    name_ds = class_obj.name
                    body = class_obj.body
                    call_parameters = [node for node in body if isinstance(node, ast.Assign) and node.targets[0].id == "call_parameters_list"]
                    call_parameters = call_parameters[0].value
                    call_parameters = ast.literal_eval(call_parameters)
                    description = [node for node in body if isinstance(node, ast.Assign) and node.targets[0].id == "description"]
                    description_value = description[0].value
                    description_dict = ast.literal_eval(description_value)
                    description_dict["class_name"] = name_ds
                    data_services += self.convert_data_service_to_document(description_dict)   # data services for prompt
                    call_parameters_list.extend(call_parameters)
                    data_services_list.append(description_dict)    # data services for saving pipeline
                
        return tables, data_services, data_services_list, data_service_to_process, call_parameters_list
    
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
        
        if self.verbose:
            print(f"Dataservices to add as import: {modules}")
        
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
    
    def chain_view(self, database, x, generator_output_chain):
        database_location = f"data_service_bird_automatic/train_databases/{database['db_id']}/{database['db_id']}.sqlite"
        data_samples = []
        for table in database['tables']:
            try:
                data_samples.append(
                    {
                        "table_name" : table,
                        "table_data_samples" : get_sample_data(database_location, table).to_string()
                    }
                )
            except:
                print(f"Exception during view extraction. Probably unexisting table: {table}")
            
        x["data_samples"] = data_samples
        
        return self.run_chain(x, generator_output_chain)
    
    def run_chain(self, x, generator_output_chain):
        return (
            RunnableLambda(
                lambda x: x)
            |
            generator_output_chain
            ).invoke(x)
        
    def add_evidence(self, second_mode, database_name, query_evidence):
        if second_mode != "added_evidence":
            return query_evidence
        else:
            further_evidence = "Salaries may be strings needing to be parsed."
            new_evidence =  query_evidence + "\n" + further_evidence
            if self.verbose:
                print(f"Added evidence: {new_evidence}")
            return new_evidence
        
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
                "query": x["query"],
                "evidence": x["evidence"],
                "ground_truth": x["ground_truth"],
                }
            )
            | RunnableBranch( 
                (lambda x: self.dataservice_mode == "ground_truth", lambda x : {
                    "query": x["query"],
                    "evidence": self.add_evidence(self.evidence_mode, self.database, x["evidence"]),
                    "data_services": self.get_data_services(sql = x["ground_truth"])
                }), lambda x : {
                    "query": x["query"],
                    "evidence": self.add_evidence(self.evidence_mode, self.database, x["evidence"]),
                    "data_services": self.get_data_services()
                }   
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "tables": x["data_services"][0],
                    "data_services": x["data_services"][1],
                    "data_services_list": x["data_services"][2],
                    "data_services_list_names": x["data_services"][3],
                    "call_parameters": x["data_services"][4],
                    "DATA_SERVICE_SECTION" : DATA_SERVICE_SECTION
                }
            )
            | RunnableBranch(
                (lambda x: self.pipeline_mode == "wo_pipeline_view", lambda x: self.chain_view({"db_id" : self.database, "tables" : x["tables"]}, x, generator_chain_output)),
                lambda x: self.run_chain(x, generator_chain_output)
            )
            | RunnableLambda (
                lambda x: {
                    "query": x["inputs"]["query"],
                    "evidence": x["inputs"]["evidence"],
                    "data_services": x["inputs"]["data_services"],
                    "data_services_list": x["inputs"]["data_services_list"],
                    "pipeline": self.correct_obvious_word_mistake(x["pipeline"][1].strip()[len("python"):].strip(), x["inputs"]["call_parameters"], similarity_treshold=self.similarity_treshold)
                }
            )
            | RunnableParallel(
                gen = RunnableLambda(lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "data_services": x["data_services"],
                    "pipeline": x["pipeline"]
                }),
                exe = RunnableLambda(lambda x:
                    self.save_intermediate_result_to_json(x["pipeline"], x["data_services_list"])
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
                    "pipeline": x["inputs"]["gen"]["pipeline"],
                }),
                output = runner_chain_output
            )
            | RunnableLambda(lambda x: {
                "query": x["inputs"]["query"],
                "evidence": x["inputs"]["evidence"],
                "data_services": x["inputs"]["data_services"],
                "pipeline": x["inputs"]["pipeline"],
                "output": x["output"]["output"],
            })
            
        )

        # return the chain
        return chain
    
    def get_chain_truth(self) -> Runnable:
        
        generator_chain = self.generator.get_chain()
        
        generator_chain_output = {
            "tools": generator_chain,
            "inputs": RunnablePassthrough()
        }

        chain = (
            RunnableLambda(lambda x: {
                "query": x["query"],
                "evidence": x["evidence"],
                "sql": x["sql"]
                }
            )
            | RunnableLambda(
                lambda x : {
                    "query": x["query"],
                    "sql": x["sql"],
                    "evidence": self.add_evidence(self.evidence_mode, self.database, x["evidence"]),
                    "data_services": self.get_data_services()
                } 
            )
            | RunnableLambda( 
                lambda x : {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "tables": x["data_services"][0],
                    "data_services": x["data_services"][1],
                    "ground_truth": self.get_data_services(sql = x["sql"])[0],
                    "DATA_SERVICE_SECTION" : DATA_SERVICE_SECTION
                }
            )
            | RunnableBranch(
                (lambda x: self.dataservice_mode == "with_view", lambda x: self.chain_view({"db_id" : self.database, "tables" : x["tables"]}, x, generator_chain_output)),
                lambda x: self.run_chain(x, generator_chain_output)
            )
            | RunnableLambda(
                lambda x: {
                    "output" : x['tools'][1].strip(),
                    "ground_truth": x['inputs']['ground_truth']
                }
            )
        )
        
        return chain


if __name__ == "__main__":
    database="chicago_crime"
    enterprise = "Mistral"
    model = "mistral-large-latest"
    mode = "wo_pipeline_view"
    dataservice_mode = "ground_truth"
    
    test_mode = "bird" # test or bird
    if test_mode == "test":
        q = "q5"
        with open("queries/queries_pipelines_human_resources.json", "r") as f:
            queries = json.load(f)
            query = queries[q]["query"]
    else:
        q = 2
        queries = get_queries(database)
        query = queries[q]["question"]
        
    sql = queries[q]["SQL"]
    
    
    llm = LLMAgent(enterprise=enterprise, model= model, pipeline_mode=mode, dataservice_mode=dataservice_mode, similarity_treshold=0.9, automatic=True, database=database, verbose=True)
    
    input_file = {
        "query" : query,
        "evidence" : queries[q]["evidence"],
        "ground_truth" : sql
    }
    
    print(f"Natural language query is: {query}")
    if sql != "":
        print(f"SQL query is: {sql}")
        
    if mode == "check_ground_truth":
        result = llm.get_chain_truth().invoke(input_file)
    else:
        result = llm.get_chain().invoke(input_file)
    
    print(result["output"])