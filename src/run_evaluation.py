import ast
import json
import os
import re
import time
import pandas as pd

from data_service_bird.database import GetDataFromDatabase
from result_averager import average_results

from data_service_generator import DataServiceGenerator, databases_description_location
from main import LLMAgent, get_queries
from evaluation.match_similarity import match_similarity

def remove_nbsp(data):
    """
    Recursively traverse the JSON data and remove non-breaking spaces from strings.
    """
    if isinstance(data, dict):
        # Process each key-value pair in dictionaries.
        return {key: remove_nbsp(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Process each element in lists.
        return [remove_nbsp(element) for element in data]
    elif isinstance(data, str):
        # Replace the non-breaking space with an empty string.
        return data.replace('\u00a0', '')
    else:
        # For numbers, booleans, None, etc., return as is.
        return data

def run_evaluation_ground_truth(database, enterprise, model, queries, automatic, verbose = False):
    llm = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="check_ground_truth", automatic=automatic, database=database, verbose=verbose)
    llm_chain = llm.get_chain_truth()
    
    num_queries = len(queries)

    res_eval = []
    for index, query in enumerate(queries):
        sql = query["SQL"]
    
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"],
            "ground_truth" : sql
        }
        question = query["question"]
        print(f"Index {index} of {num_queries}, Question: {question}")
        res = llm_chain.invoke(input_file)
        res_eval.append([index, res["output"], res["ground_truth"]])
        
        if enterprise == "Mistral":
            time.sleep(0.2)
    res_df = pd.DataFrame(res_eval, columns=["index", "tools_prediction", "ground_truth"])
    safe_model = str(model.replace("-", "_"))
    res_df.to_csv(f"evaluation/evaluation_results_check_ground_truth_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
    
def evaluate_ground_truth(database, enterprise, model):
    metrics_res = []
    safe_model = str(model.replace("-", "_"))
    eval_results = pd.read_csv(f"evaluation/evaluation_results_check_ground_truth_{database}_{enterprise}_{safe_model}.csv")

    for index, line in eval_results.iterrows():
        ground_truth = ast.literal_eval(line["ground_truth"])
        ground_num = len(ground_truth)
        correct_choices = 0
        wrong_choises = 0
        
        prediction = ast.literal_eval(line["tools_prediction"])
        for tool in prediction:
            if tool in ground_truth:
                correct_choices += 1
            else:
                wrong_choises += 1
        
        accuracy = correct_choices/(ground_num + wrong_choises)
        recall = correct_choices/ground_num
    
        metrics_res.append([index, accuracy, recall])
    
    metrics_res = pd.DataFrame(metrics_res, columns=["index", "accuracy", "recall"])
    averages = average_results(metrics_res, "ground_truth_check")
    metrics_res.to_csv(f"evaluation/detailed_results_check_ground_truth_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
    averages.to_csv(f"evaluation/summarized_results_check_ground_truth_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
    print("Detailed metrics are:")
    print(metrics_res)
    print("Summarized metrics are:")
    print(averages)
        

def run_evaluation(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = None, automatic=False, similarity_treshold = 0.9, verbose=False):
    llm = LLMAgent(enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = dataservice_mode, similarity_treshold=similarity_treshold, automatic=automatic, database=database, verbose=verbose)
    llm_chain = llm.get_chain()
    
    num_queries = len(queries)

    res_eval = []
    for index, query in enumerate(queries):
        
        sql = query["SQL"]
    
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"],
            "ground_truth" : sql
        }
        question = query["question"]
        print(f"Index {index} of {num_queries}, Question: {question}")
        try:
            res = llm_chain.invoke(input_file)

            data_services = res['data_services']
            pipeline =  res["pipeline"]
            output = res["output"]
            
            output_json = json.loads(open("result.json", "r").read())

            res_elem = [index, question, sql, data_services, pipeline, output, output_json]
        except Exception as e:
            print(f"Error in query {index}: {e}")
            res_elem = [index, question, sql, None, None, None, None]
        res_eval.append(res_elem)
        
        if enterprise == "Mistral":
            time.sleep(0.2)

    res_df = pd.DataFrame(res_eval, columns=["index", "question", "sql", "data_services", "pipeline", "output", "output_json"])
    safe_model = str(model.replace("-", "_"))
    res_df.to_csv(f"evaluation/evaluation_results_{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}.csv", sep=',', index=False)


def evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic):
    db = GetDataFromDatabase()
    agent = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="wo_pipeline")
    
    if automatic:
        db.open_connection(f"data_service_bird_automatic/train_databases/{database}/{database}.sqlite")
    else:
        db.open_connection(f"data_service_bird/{database}/{database}.sqlite")

    metrics_res = []
    safe_model = str(model.replace("-", "_"))
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}.csv")
    #eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    num_queries = len(queries)
    for index, query in enumerate(queries):
        metrics_res_q_idx = []
        question = query["question"]
        df1 = db.call(query=query["SQL"])
        print(f"Index {index} of {num_queries}, Question: {question}")
        res = eval_results[(eval_results["index"] == index)]
        #Check if pipeline completely failed like query 35 eval_26-11-24
        if type(res["output_json"].values[0]) != str:
            output_json = ""
        else:
            output_json = str(res["output_json"].values[0])
            output_json = output_json.replace('\\xa0', '')
            output_json = output_json.replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")
            pattern = r'(".+\$)".+"'
            output_json = re.sub(pattern, r'\1.+"', output_json)
        if verbose:
            print("Raw JSON output is:")
            print(output_json)
            
        if output_json != "":
            try:
                output_res = json.loads(output_json)
                df2 = pd.DataFrame(output_res)
            except Exception as e:
                print("Exception while load json")
                print(e)
                df2 = pd.DataFrame()
        else:
            df2 = pd.DataFrame()

        if not df2.empty:
            #Full name extrapolation
            words = agent.check_word_simliarity("fullname", list(df2.columns), similarity_treshold=0.55)
            if len(words) > 0 :
                words = sorted(words, key=lambda x: x[1])
                if 'fullname' != words[0][0]:
                    try:
                        df2 = df2.rename(columns={words[0][0] : "fullname"})
                        df2[['firstname','lastname']] = df2["fullname"].str.split(expand=True)
                        del df2["fullname"]
                        print(f"Changed {words[0][0]}")
                        #print(df2)
                    except:
                        print("Cannot decompose fullname column")
                        pass
            
            df1.name = "table_1"
            df2.name = "table_2"
            
            try:
                precision, recall, acc_cell, acc_row = match_similarity(df1, df2)
            except Exception as e:
                print(f"Exception for query {index}")
                print(f"Exception: {e}")
                precision, recall, acc_cell, acc_row = 0, 0, 0, 0
        else:
            print("Empty pipeline result, probably failed execution. Not performing any match.")
            precision, recall, acc_cell, acc_row = 0, 0, 0, 0
            
        metrics_res_q_idx.append([precision, recall, acc_cell, acc_row])
        
        # average metrics
        metrics_res_q_idx = pd.DataFrame(metrics_res_q_idx, columns=["precision", "recall", "acc_cell", "acc_row"])
        metrics_res_q_idx = metrics_res_q_idx.mean()

        # append to the final results with the query index
        metrics_res_q_idx = [index] + metrics_res_q_idx.to_list()
        metrics_res.append(metrics_res_q_idx)

    metrics_res = pd.DataFrame(metrics_res, columns=["index", "precision", "recall", "acc_cell", "acc_row"])
    averages = average_results(metrics_res, "pipeline")
    safe_model = str(model.replace("-", "_"))
    metrics_res.to_csv(f"evaluation/metrics_results_{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}.csv", sep=',', index=False)
    averages.to_csv(f"evaluation/summarized_results_{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}.csv", sep=',', index=False)
    print("Detailed metrics are:")
    print(metrics_res)
    print("Summarized metrics are:")
    print(averages)
    return

if __name__ == "__main__":
    
    enterprise="Mistral"
    model = "mistral-large-latest"
    database="chicago_crime"
    print(f"Model: {model}, Database: {database}")
    
    generation = True
    force_generation = False
    ground_truth_check = False
    pipeline_check = True
    automatic = True
    verbose = False
    
    pipeline_mode = "wo_pipeline_view"
    evidence_mode = "standard_evidence"
    dataservice_mode = "ground_truth"
    similarity_treshold = 0.9
    
    queries = get_queries(database)
    print(f"Got {len(queries)} queries")
    
    if generation:
        safe_model = str(model.replace("-", "_"))
        exist = os.path.exists(f"data_service_bird_automatic/train_databases/{database}/data_services/{enterprise}/{safe_model}/")
        if not exist or (exist and force_generation):
            print("Generating Data services")
            databases = None
            with open(databases_description_location) as f:
                databases = json.load(f)
            database_info = [db for db in databases if db["db_id"] == database][0]
            dataservice_generator = DataServiceGenerator(enterprise, model)
            dataservice_generator.create_data_services(database_info)

    if ground_truth_check:
        print("Performing ground truth check")
        run_evaluation_ground_truth(database, enterprise, model, queries, automatic=automatic, verbose = verbose)
        evaluate_ground_truth(database, enterprise, model)
        
    if pipeline_check:
        print("Performing pipeline check")
    
        with open("result.json", "w") as file:
        # Use the `truncate()` method to clear the file's content
            file.truncate()
        
        run_evaluation(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode=dataservice_mode, automatic=automatic, similarity_treshold=similarity_treshold, verbose=verbose)
        evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic=automatic)
    
    
