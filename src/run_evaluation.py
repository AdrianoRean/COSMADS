import ast
import json
import math
import os
import re
import time
import pandas as pd

from data_service_bird.database import GetDataFromDatabase
from result_averager import average_results

from data_service_generator import DataServiceGenerator, databases_description_location
from main import LLMAgent, get_queries
from evaluation.match_similarity import match_similarity
from judge import Judge

def run_evaluation_ground_truth(database, enterprise, mode, model, queries, automatic, verbose = False):
    llm = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="check_ground_truth", dataservice_mode=mode, automatic=automatic, database=database, verbose=verbose)
    llm_chain = llm.get_chain_truth()
    
    num_queries = len(queries)

    res_eval = []
    for index, query in enumerate(queries):
        sql = query["SQL"]
    
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"],
            "sql" : sql
        }
        question = query["question"]
        print(f"Index {index} of {num_queries}, Question: {question}")
        
        try:
            res = llm_chain.invoke(input_file)
        except:
            print("Probably LLM rate exceeded. Waiting 2 seconds and retrying.")
            time.sleep(2)
            res = llm_chain.invoke(input_file)
            
        res_eval.append([index, res["output"], res["ground_truth"]])
        
        
        if enterprise == "Mistral":
            time.sleep(0.2)
    res_df = pd.DataFrame(res_eval, columns=["index", "tools_prediction", "ground_truth"])
    safe_model = str(model.replace("-", "_"))
    res_df.to_csv(f"evaluation/evaluation_results_check_ground_truth_{mode}_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
    
def evaluate_ground_truth(database, enterprise, model, mode):
    metrics_res = []
    safe_model = str(model.replace("-", "_"))
    eval_results = pd.read_csv(f"evaluation/evaluation_results_check_ground_truth_{mode}_{database}_{enterprise}_{safe_model}.csv")

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
    metrics_res.to_csv(f"evaluation/detailed_results_check_ground_truth_{mode}_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
    averages.to_csv(f"evaluation/summarized_results_check_ground_truth_{mode}_{database}_{enterprise}_{safe_model}.csv", sep=',', index=False)
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
            try:
                res = llm_chain.invoke(input_file)
            except:
                print("Probably LLM rate exceeded. Waiting 2 seconds and retrying.")
                time.sleep(2)
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

def metrics_valentine(index, sql, db, res, fullname_split, agent, verbose = False):
    df1 = db.call(sql)
    #Check if pipeline completely failed like query 35 eval_26-11-24
    if type(res["output_json"].values[0]) != str:
        output_json = ""
    else:
        output_json = res["output_json"].values[0]
        output_json = output_json.replace('\\xa0', '')
        output_json = output_json.replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")
        
    if verbose:
        print("Raw JSON output is:")
        print(output_json)
        
    if output_json == "":
        df2 = pd.DataFrame()
    else:
        try:
            output_res = json.loads(output_json)
            df2 = pd.DataFrame(output_res)
            num_of_entries = df2.shape[0]
        except Exception as e:
            print("Exception while load json")
            print(e)
            df2 = pd.DataFrame()

    if not df2.empty:
        
        if fullname_split:
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
        
    return [index, precision, recall, acc_cell, acc_row]

def averaging_saving_print_results(results, columns, averaging_mode, partial_file_path):
    df_results = pd.DataFrame(results, columns=columns)
    averages = average_results(df_results, averaging_mode)
    df_results.to_csv(f"evaluation/metrics_results_{averaging_mode}_{partial_file_path}.csv", sep=',', index=False)
    averages.to_csv(f"evaluation/summarized_results_{averaging_mode}_{partial_file_path}.csv", sep=',', index=False)
    print(f"Detailed {averaging_mode} metrics are:")
    print(df_results)
    print(f"Summarized {averaging_mode} metrics are:")
    print(averages)
    return df_results

def check_all_zeros(list):
    for value in list:
        if value > 0:
            return False
    return True

def evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic, valentine = True, llm = False, unified = False, fullname_split=False):
    
    safe_model = str(model.replace("-", "_"))
    partial_file_path = f"{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}"
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{partial_file_path}.csv")
    
    if valentine:
        db = GetDataFromDatabase()
        if automatic:
            db.open_connection(f"data_service_bird_automatic/train_databases/{database}/{database}.sqlite")
        else:
            db.open_connection(f"data_service_bird/{database}/{database}.sqlite")
            
        agent = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="wo_pipeline")
        metrics_res = []
        
    if llm:
        judge = Judge(enterprise, model, mode="verdict")
        verdict_res = []

    #eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    num_queries = len(queries)
    
    if valentine or llm:
        for index, query in enumerate(queries):
            
            question = query["question"]
            print(f"Index {index} of {num_queries}, Question: {question}")
            
            res = eval_results[(eval_results["index"] == index)]
            sql = query["SQL"]
            
            if valentine:    
                metrics_res.append(metrics_valentine(index, sql, db, res, fullname_split, agent, verbose))
            
            if llm:
                try:
                    verdict = judge.judge(res["pipeline"], sql, question)
                    verdict_res.append([index, verdict])
                except:
                    print("Probably LLM rate exceeded. Waiting 2 seconds and retrying.")
                    time.sleep(2)
                    verdict = judge.judge(res["pipeline"], sql, question)
                    verdict_res.append([index, verdict])
                    
                print(f"Verdict is: {verdict}")
                if enterprise == "Mistral":
                    time.sleep(0.3)
        
    if valentine:
        columns = ["index", "precision", "recall", "acc_cell", "acc_row"]
        metrics_res = averaging_saving_print_results(metrics_res, columns, "valentine", partial_file_path)
    
    if llm:
        columns = ["index", "verdict"]
        verdict_res = averaging_saving_print_results(verdict_res, columns, "llm", partial_file_path)
    
    if unified:
        print("Unifying valentin and llm results.\n    Ignoring MISLEADING results and setting metrics to 1 if TRUE")
        if not valentine:
            print("Reading past valentine results")
            metrics_res = pd.read_csv(f"evaluation/metrics_results_valentine_{partial_file_path}.csv")
        if not llm:
            print("Reading past llm results")
            verdict_res = pd.read_csv(f"evaluation/metrics_results_llm_{partial_file_path}.csv")
        unified_metric_res = []
        for index in range(num_queries):
            verdict = verdict_res.iloc[index]["verdict"]
            metrics = metrics_res.iloc[index][["precision", "recall", "acc_cell", "acc_row"]].to_list()
            
            if verdict == "SQL-WRONG" and check_all_zeros(metrics):
                continue
            elif verdict == "EQUIVALENT":
                metrics = metrics_res.iloc[index].to_list()
                new_metrics = [index,1,1,1,1]
                for i in range(1, len(metrics)):
                    if not math.isnan(metrics[i]):
                        new_metrics[i] = new_metrics[i]*0.5 + metrics[i]*0.5
                unified_metric_res.append(new_metrics)
            else:
                unified_metric_res.append(metrics_res.iloc[index].to_list())
        
        columns = ["index", "precision", "recall", "acc_cell", "acc_row"]
        averaging_saving_print_results(unified_metric_res, columns, "unified", partial_file_path)
    
    return

if __name__ == "__main__":
    
    enterprise="Mistral"
    model = "mistral-large-latest"
    database="chicago_crime"
    print(f"Model: {model}, Database: {database}")
    
    generation = False
    force_generation = False
    print(f"Generation: {generation}, Forcing regeneration: {force_generation}")
    
    automatic = True
    print(f"Data Services are generated: {automatic}")
    
    similarity_treshold = 0.9
    print(f"Bert similarity treshold: {similarity_treshold}")
    
    verbose = False
    print(f"Verbose: {verbose}")
    
    only_metrics = False
    valentine = True
    llm = False
    unified = False
    print(f"Only calculating metrics: {only_metrics}, Valentine metrics: {valentine}, Judge metrics: {llm}, Unified metrics: {unified}")

    ground_truth_check = True
    ground_truth_check_mode = "no_view" # "no_view"
    print(f"Data service selection evaluation: {ground_truth_check}, Data service selection mode: {ground_truth_check_mode}")
    
    pipeline_check = False
    pipeline_mode = "wo_pipeline_view" # "wo_pipeline"
    evidence_mode = "standard_evidence"
    dataservice_mode = "ground_truth"
    print(f"Pipeline evaluation: {pipeline_check}, Pipeline mode: {pipeline_mode}, Evidence mode: {evidence_mode}, Data service mode: {dataservice_mode}")
    
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
        if not only_metrics:
            run_evaluation_ground_truth(database, enterprise, ground_truth_check_mode, model, queries, automatic=automatic, verbose = verbose)
        evaluate_ground_truth(database, enterprise, model, ground_truth_check_mode)
        
    if pipeline_check:
        print("Performing pipeline check")
    
        with open("result.json", "w") as file:
        # Use the `truncate()` method to clear the file's content
            file.truncate()
        
        if not only_metrics:
            run_evaluation(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode=dataservice_mode, automatic=automatic, similarity_treshold=similarity_treshold, verbose=verbose)
        evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic=automatic, fullname_split=False, valentine=valentine, llm=llm, unified=unified)
    
    
