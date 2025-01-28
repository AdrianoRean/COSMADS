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

# append the path to the parent directory to the system path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline_manager_db import PipelineManagerDB
from pipeline_chain import PipelineGeneratorAgent
from runner_chain import PipelineRunner
from document_manager_db import DocumentManagerDB

INTERMEDIATE_RESULTS_FILEPATH = Path(__file__).parent / "temp_pipeline.py"

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
    all_tables = set(from_tables + join_matches)
    return list(all_tables)

class LLMAgent:
    def __init__(self, model="GPT", mode = "standard", second_mode = "standard_evidence", services_mode = None, combinatory = False, automatic=False, database="human_resources"):
        dotenv.load_dotenv()
        if model == "GPT":
            self.key = os.getenv("OPENAI_API_KEY")
        elif model == "Mistral":
            self.key = os.getenv("MISTRAL_API_KEY")
        print(mode)
        self.mode = mode
        self.second_mode = second_mode
        self.services_mode = services_mode
        
        self.database = database
        self.automatic = automatic

        self.pipeline_manager = PipelineManagerDB(model, self.key)
        self.document_manager = DocumentManagerDB()
        
        if mode == "chain_of_thoughs":
            self.suggestor = PipelineGeneratorAgent(model, self.key, mode="chain_of_thoughs")
            self.generator = PipelineGeneratorAgent(model, self.key, mode="chain_of_thoughs_post")
        else:
            self.generator = PipelineGeneratorAgent(model, self.key, mode=mode, combinatory=combinatory)
        self.runner = PipelineRunner()

        #self.ds_directory = "data_services"
        if automatic:
            self.ds_directory = f"data_service_bird_automatic/train_databases/{database}/data_services"
            self.ds_directory_import = f"data_service_bird_automatic.train_databases.{database}.data_services"
        else:
            self.ds_directory = f"data_service_bird/{database}"
            self.ds_directory_import = f"data_service_bird.{database}"
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
                
        print(inside_call_matches)
        
        #check for each possible value if there are matches
        for word in call_keys:
            best_matches = self.check_word_simliarity(word, inside_call_matches, similarity_treshold)
            print(f"Key: {word} - Matches: {best_matches}")
            for match in best_matches:
                if match[1] == 1: #perfect match
                    if word == match[0]: #also cases are equal, no need to change
                        continue
                #imperfect match or case difference, probable typo mistake
                calls_indexes = call_match_dict[match[0]] #find calls to be replaced
                for call_index in calls_indexes:
                    call = call_matches[call_index] #retrieve single call
                    print(f"Key: {word} - calls : {call}")
                    new_call = call.replace(match[0], word) #fix it
                    call_matches[call_index] = new_call #save bettered version for future use
                    function_changed = function_changed.replace(call, new_call) #replace in actual function
                        
        #find all hard coded words in function
        hard_coded_regex_string = re.compile(r"\"([\w+\s]+)\"")
        hard_coded_matches = list(set(hard_coded_regex_string.findall(function_changed)))  
        print(hard_coded_matches)  
        #check for each possible value if there are matches
        for word in hard_coded_words:
            best_matches = self.check_word_simliarity(word, hard_coded_words, similarity_treshold)
            print(f"Word: {word} - Matches: {best_matches}")
            for match in best_matches:
                if match[1] == 1: #perfect match
                    if word == match[0]: #also cases are equal, no need to change
                        continue
                #imperfect match or case difference, probable typo mistake
                specific_hard_coded_regex = re.compile(f"\"{match[0]}\"")
                function_changed = specific_hard_coded_regex.sub(f'"{word}"', function_changed)
                
        return function_changed

    def get_example(self, res_search, pipeline_index, pipeline_index_2):
        if pipeline_index != None and pipeline_index_2 == None:
            return self.get_example_wrong_all(pipeline_index)
        elif pipeline_index_2 != None:
            #print("searching combinatory")
            return self.get_example_wrong_combinatory(pipeline_index, pipeline_index_2)
        
        simil_query = res_search.page_content
        pipeline_id = res_search.metadata["pipeline"]
        if self.mode == "chain_of_tables":
            pipeline_id = "pipelines_bird/chain_of_tables" + pipeline_id[len("pipelines_bird"):]
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
    
    def get_example_wrong_all(self, pipeline_index):
        f = json.load(open("queries/queries_pipelines_human_resources.json"))
        
        query = f[f"q{pipeline_index}"]["query"]
        pipeline = f[f"q{pipeline_index}"]["pipeline"]
            
        if self.mode == "chain_of_tables":
            pipeline = "pipelines_bird/chain_of_tables" + pipeline[len("pipelines_bird"):]
        pipeline_text = open(pipeline).read()
        
        return [query, pipeline_text]
    
    def get_example_wrong_combinatory(self, pipeline_index, pipeline_index_2):
        f = json.load(open("queries/queries_pipelines_human_resources.json"))
        
        query = f[f"q{pipeline_index}"]["query"]
        pipeline = f[f"q{pipeline_index}"]["pipeline"]
        query_2 = f[f"q{pipeline_index_2}"]["query"]
        pipeline_2 = f[f"q{pipeline_index_2}"]["pipeline"]
            
        if self.mode == "chain_of_tables":
            pipeline = "pipelines_bird/chain_of_tables" + pipeline[len("pipelines_bird"):]
        pipeline_text = open(pipeline).read()
        pipeline_text_2 = open(pipeline_2).read()
        
        return [(query, query_2), (pipeline_text, pipeline_text_2)]
    
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
        data_services_all = glob.glob(f"{self.ds_directory}/*.py")
        data_services = ""
        data_services_list = []
        query_services_list = []
        call_parameters_list = []
        if sql != None:
            query_services_list = extract_tables(sql)
            print(f"Tables find from sql are: {query_services_list}")
        for data_service in data_services_all:
            if self.automatic:
                data_service_name = data_service[len(f"data_service_bird_automatic/train_databases/{self.database}/data_services/"):-3]
            else:
                data_service_name = data_service[len(f"data_service_bird/{self.database}/"):-3]
            if sql == None or data_service_name in query_services_list: 
                with open(f"{data_service}", mode="r") as f:
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
                
        return data_services, data_services_list, query_services_list, call_parameters_list
    
    def get_relevant_document(self, query):
        document = self.document_manager.extract_document(query)
        return document
    
    def convert_data_service_to_document(self, data_service_doc: dict) -> str:
        document = data_service_doc
        document_str = self.sep.join([f"{key}: {value}" for key, value in document.items()])
        return document_str

    def save_intermediate_result_to_json(self, pipeline, data_services, pipeline_index = None, pipeline_index_2 = None) -> str:
        file_to_save = ""
        print(f"dataservices : {data_services}")
        
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

        if pipeline_index == None:
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
        elif pipeline_index_2 == None:
            main_function = f"""
if __name__ == "__main__":
    result = pipeline_function()
    import json
    import pandas as pd
    with open("result_{pipeline_index}.json", "w") as f:
        json.dump(result, f, indent=4)
    result = pd.DataFrame(result)
    from tabulate import tabulate
    result = tabulate(result, headers='keys', tablefmt='psql')
    print(result)
    """
        else:
            main_function = f"""
if __name__ == "__main__":
    result = pipeline_function()
    import json
    import pandas as pd
    with open("results/result_{pipeline_index}_{pipeline_index_2}.json", "w") as f:
        json.dump(result, f, indent=4)
    result = pd.DataFrame(result)
    from tabulate import tabulate
    result = tabulate(result, headers='keys', tablefmt='psql')
    print(result)
    """
        file_to_save += main_function

        with open(INTERMEDIATE_RESULTS_FILEPATH, "w") as f:
            f.write(file_to_save)
    
    def save_advice(self, advice, pipeline_index):
        if pipeline_index != None:
            with open("advice.json", "w") as f:
                json.dump(json.loads(advice), f, indent=4)
        else:
            with open(f"advice_{pipeline_index}.json", "w") as f:
                json.dump(json.loads(advice), f, indent=4)
            
    def chain_of_thoughs(self, x, generator_chain_output, pipeline_index = None):
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
                        self.save_advice(x["pipeline"][1], pipeline_index)
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
    
    def chain_of_error(self, x, generator_chain_output, runner_chain_output):
        
        first = True
        
        self.first_trier = PipelineGeneratorAgent(self.key, mode="standard")
        
        first_trier_chain_output = {
            "pipeline": self.first_trier.get_chain(),
            "inputs": RunnablePassthrough()
        }
        
        chain = ( 
                 RunnableBranch(
                    (lambda x: first, lambda x: self.run_chain(x, first_trier_chain_output)),
                    lambda x: self.run_chain(x, generator_chain_output)
                 )
                 | RunnableParallel(
                    gen = RunnableLambda(lambda x: {
                        "query": x["inputs"]["query"],
                        "evidence": x["inputs"]["evidence"],
                        "data_services": x["inputs"]["data_services"],
                        "data_services_list": x["inputs"]["data_services_list"],
                        "example_query": x["inputs"]["example_query"],
                        "example_pipeline": x["inputs"]["example_pipeline"],
                        "pipeline": x["pipeline"][1].strip()[len("python"):].strip(),
                        "analysis": x["pipeline"][0]
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
                        "data_services_list": x["inputs"]["gen"]["data_services_list"],
                        "example_query": x["inputs"]["gen"]["example_query"],
                        "example_pipeline": x["inputs"]["gen"]["example_pipeline"],
                        "pipeline": x["inputs"]["gen"]["pipeline"],
                        "analysis": x["inputs"]["gen"]["analysis"],
                    }),
                    output = runner_chain_output
                )
                | RunnableLambda(lambda x: {
                    "query": x["inputs"]["query"],
                    "evidence": x["inputs"]["evidence"],
                    "data_services": x["inputs"]["data_services"],
                    "data_services_list": x["inputs"]["data_services_list"],
                    "example_query": x["inputs"]["example_query"],
                    "example_pipeline": x["inputs"]["example_pipeline"],
                    "pipeline": x["inputs"]["pipeline"],
                    "analysis": x["inputs"]["analysis"],
                    "output": x["output"]["output"],
                })
            )
        
        for i in range (0,3):   
            x = chain.invoke(x)
            analysis = x['analysis'].strip()
            
            if x['output'][0] == "+":
                x["inputs"] = {
                    "query" : x["query"],
                    "evidence" : x["evidence"],
                    "data_services" : x["data_services"],
                    "data_services_list": x["data_services_list"],
                    "example_query" : x["example_query"],
                    "example_pipeline" : x["example_pipeline"],
                }
                x["pipeline"] = [None, x["pipeline"]]
                x["pipeline"][1] = "python" + x["pipeline"][1]
                break
            else:
                print("**********************")
                print(f"Last analysis is: {analysis}")            
                print(f"Last pipeline is: {x['pipeline']}")        
                print(f"Output 0: {x['output'][0]}")  
                print("------------------------ OUTPUT -----------------")   
                print(x["output"])   
                first = False
        
        x["inputs"] = {
            "query" : x["query"],
            "evidence" : x["evidence"],
            "data_services" : x["data_services"],
            "data_services_list": x["data_services_list"],
            "example_query" : x["example_query"],
            "example_pipeline" : x["example_pipeline"],
        }
        x["pipeline"] = [None, x["pipeline"]]
        x["pipeline"][1] = "python" + x["pipeline"][1]
                
        return x
    
    def chain_view(self, database, x, generator_output_chain, pipeline_index_2 = None):
        database_location = f"data_service_bird_automatic/train_databases/{database['db_id']}/{database['db_id']}.sqlite"
        data_samples = []
        for table in database['tables']:
            data_samples.append(
                {
                    "table_name" : table,
                    "table_data_samples" : get_sample_data(database_location, table).to_string()
                }
            )
            
        x["data_samples"] = data_samples
        
        if pipeline_index_2 != None:
            #print("separating examples")
            x["example_query_1"] = x["example_query"][0]
            x["example_query_2"] = x["example_query"][1]
            x["example_pipeline_1"] = x["example_pipeline"][0]
            x["example_pipeline_2"] = x["example_pipeline"][1]
        
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
            print(new_evidence)
            return new_evidence
        
    def get_chain(self, pipeline_index = None, pipeline_index_2 = None) -> Runnable:
        
        print(self.mode)
        print(self.second_mode)
        
        generator_chain = self.generator.get_chain()
        runner_chain = self.runner.get_chain(pipeline_index, pipeline_index_2)
        
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
                "pipeline_search": self.pipeline_manager.pipeline_store.search(x["query"]),
                }
            )
            | RunnableBranch( 
                (lambda x: self.services_mode == "ground_truth", lambda x : {
                    "query": x["query"],
                    "evidence": self.add_evidence(self.second_mode, self.database, x["evidence"]),
                    "example": ["", ""],
                    "data_services": self.get_data_services(sql = x["ground_truth"])
                }), lambda x : {
                    "query": x["query"],
                    "evidence": self.add_evidence(self.second_mode, self.database, x["evidence"]),
                    "example": self.get_example(x["pipeline_search"]["output"], pipeline_index, pipeline_index_2),
                    "data_services": self.get_data_services()
                }
                
            )
            | RunnableLambda( 
                lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "data_services": x["data_services"][0],
                    "data_services_list": x["data_services"][1],
                    "data_services_list_names": x["data_services"][2],
                    "call_parameters": x["data_services"][3],
                    "example_query": x["example"][0],
                    "example_pipeline": x["example"][1],
                }
            )
            | RunnableBranch(
                (lambda x: self.mode == "chain_of_thoughs", lambda x: self.chain_of_thoughs(x, generator_chain_output)),
                (lambda x: self.mode == "chain_of_tables", lambda x: self.chain_of_tables(x, generator_chain_output)),
                (lambda x: self.mode == "chain_of_error", lambda x: self.chain_of_error(x, generator_chain_output, runner_chain_output)),
                (lambda x: self.mode in ["wo_pipeline_view", "standard_view"], lambda x: self.chain_view({"db_id" : self.database, "tables" : x["data_services_list_names"]}, x, generator_chain_output, pipeline_index_2)),
                lambda x: self.run_chain(x, generator_chain_output)
            )
            | RunnableLambda (
                lambda x: {
                    "query": x["inputs"]["query"],
                    "evidence": x["inputs"]["evidence"],
                    "data_services": x["inputs"]["data_services"],
                    "data_services_list": x["inputs"]["data_services_list"],
                    "example_query": x["inputs"]["example_query"],
                    "example_pipeline": x["inputs"]["example_pipeline"],
                    "pipeline": self.correct_obvious_word_mistake(x["pipeline"][1].strip()[len("python"):].strip(), x["inputs"]["call_parameters"])
                }
            )
            | RunnableParallel(
                gen = RunnableLambda(lambda x: {
                    "query": x["query"],
                    "evidence": x["evidence"],
                    "data_services": x["data_services"],
                    "example_query": x["example_query"],
                    "example_pipeline": x["example_pipeline"],
                    "pipeline": x["pipeline"]
                }),
                exe = RunnableLambda(lambda x:
                    self.save_intermediate_result_to_json(x["pipeline"], x["data_services_list"], pipeline_index, pipeline_index_2)
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
    
    def get_chain_all(self, num_of_pipelines):
        chains = []
        for pipe in range(num_of_pipelines):
            chains.append(self.get_chain(pipe))
        return chains
    
    def get_chain_combinatory(self, num_of_pipelines):
        chains = []
        for pipe in range(num_of_pipelines):
            chains_2 = []
            for pipe_2 in range(num_of_pipelines):
                if pipe == pipe_2:
                    continue
                chains_2.append(self.get_chain(pipe, pipe_2))
            chains.append(chains_2)
        return chains
    
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
                "ground_truth": x["ground_truth"]
                }
            )
            | RunnableLambda( 
                lambda x : {
                    "query": x["query"],
                    "evidence": self.add_evidence(self.second_mode, self.database, x["evidence"]),
                    "data_services": self.get_data_services()[0],
                    "data_services_list": self.get_data_services(sql = x["ground_truth"])[2]
                }
            )
            | generator_chain_output
            | RunnableLambda(
                lambda x: {
                    "output" : x['tools'][1].strip(),
                    "ground_truth": x['inputs']['data_services_list']
                }
            )
        )
        
        return chain


if __name__ == "__main__":
    database="european_football_1"
    model = "GPT"
    mode = "wo_pipeline_view"
    llm = LLMAgent(model= model, mode=mode, services_mode="ground_truth", automatic=True, database="european_football_1")
    
    test_mode = "bird" # test or bird
    
    if test_mode == "test":
        q = "q5"
        with open("queries/queries_pipelines_human_resources.json", "r") as f:
            queries = json.load(f)
            query = queries[q]["query"]
    else:
        q = 45
        with open(f"queries/test/{database}.json", "r") as f:
            queries = json.load(f)
            query = queries[q]["question"]
        
    sql = queries[q]["SQL"]
    
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