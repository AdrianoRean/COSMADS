import ast
import json
import math
import os
import time
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

from data_service_bird.database import GetDataFromDatabase
from result_averager import average_results

from data_service_generator import DataServiceGenerator, databases_description_location
from main import LLMAgent, get_queries
from evaluation.match_similarity import match_similarity
from judge import Judge


PIPELINE_GENERATION_DELAY_SEC = 5 
PIPELINE_GENERATION_RETRY_DELAY_SEC = 10

def run_evaluation_ground_truth(database, enterprise, mode, model, queries, automatic, verbose = False):
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="check_ground_truth", dataservice_mode=mode, automatic=automatic, database=database, verbose=verbose)
    ## modalità check ground truth significa che sto usando il selector, devo anche mettere get_chain_truth (la catena giusta)
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
        except Exception as e:
            print(f"Error in query {index}: {e}")
            print(f"Probably LLM rate exceeded. Waiting {PIPELINE_GENERATION_RETRY_DELAY_SEC} seconds and retrying.")
            time.sleep(PIPELINE_GENERATION_RETRY_DELAY_SEC)
            res = llm_chain.invoke(input_file)
            
        res_eval.append([index, res["output"], res["ground_truth"]])
        
        
        if enterprise == "Mistral":
            time.sleep(0.2)
    res_df = pd.DataFrame(res_eval, columns=["index", "tools_prediction", "ground_truth"])
    safe_model = str(model.replace("-", "_"))
    res_df.to_csv(result_dir / f"evaluation_results_check_ground_truth__{mode}__{database}__{enterprise}__{safe_model}.csv", sep=',', index=False)
    
def evaluate_ground_truth(database, enterprise, model, mode):
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_res = []
    safe_model = str(model.replace("-", "_"))
    eval_results = pd.read_csv(result_dir / f"evaluation_results_check_ground_truth__{mode}__{database}__{enterprise}__{safe_model}.csv")

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
        
        prediction_num = len(prediction)

        accuracy = correct_choices/(ground_num + wrong_choises) if ground_num + wrong_choises > 0 else 0
        precision = correct_choices/prediction_num if prediction_num > 0 else 0
        recall = correct_choices/ground_num if ground_num > 0 else 0
    
        metrics_res.append([index, accuracy, precision, recall])
    
    metrics_res = pd.DataFrame(metrics_res, columns=["index", "accuracy", "precision", "recall"])
    averages = average_results(metrics_res, "ground_truth_check")
    metrics_res.to_csv(result_dir / f"detailed_results_check_ground_truth__{mode}__{database}__{enterprise}__{safe_model}.csv", sep=',', index=False)
    averages.to_csv(result_dir / f"summarized_results_check_ground_truth__{mode}__{database}__{enterprise}__{safe_model}.csv", sep=',', index=False)
    print("Detailed metrics are:")
    print(metrics_res)
    print("Summarized metrics are:")
    print(averages)
        

def run_evaluation(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = None, automatic=False, similarity_treshold = 0.9, verbose=False, data_service_gen_enterprise="Openai", data_service_gen_model="gpt-4o"):
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)
    # skip if there are files in the directory
    if len(list(result_dir.glob("*.csv"))) > 0:
        print(f"Results already present, skipping running evaluation with {enterprise} on {database}")
        return

    llm = LLMAgent(enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = dataservice_mode, similarity_treshold=similarity_treshold, automatic=automatic, database=database, verbose=verbose, data_service_gen_enterprise=data_service_gen_enterprise, data_service_gen_model=data_service_gen_model)
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
                # sleep for a few seconds to avoid exceeding the LLM rate
                time.sleep(PIPELINE_GENERATION_DELAY_SEC)
            except Exception as e:
                print(f"Error in query {index}: {e}")
                print(f"Probably LLM rate exceeded. Waiting {PIPELINE_GENERATION_RETRY_DELAY_SEC} seconds and retrying.")
                time.sleep(PIPELINE_GENERATION_RETRY_DELAY_SEC)
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
    res_df.to_csv(result_dir / f"evaluation_results__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv", sep=',', index=False)


def compute_execution_accuracy(index, sql, db, res, verbose = False):
    execution_accuracy = 0
    # get the ground truth table
    ground_truth_sql_table = db.call(sql)
    # parse the result table
    if type(res["output_json"].values[0]) != str:
        if verbose:
            print("Empty pipeline result, probably failed execution. Not performing any match.")
        return [index, execution_accuracy]
    output_json = res["output_json"].values[0]
    output_json = output_json.replace('\\xa0', '')
    output_json = output_json.replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")

    if output_json == "":
        if verbose:
            print("Empty pipeline result, probably failed execution. Not performing any match.")
        return [index, execution_accuracy]
    
    try:
        output_res = json.loads(output_json)
        output_table = pd.DataFrame(output_res)

        if output_table.shape != ground_truth_sql_table.shape:
            if verbose:
                print("Different shapes of the two tables")
            return [index, execution_accuracy]
        # sort the columns alphabetically for comparison
        output_table = output_table[sorted(output_table.columns)]
        ground_truth_sql_table = ground_truth_sql_table[sorted(ground_truth_sql_table.columns)]
        # convert both tables to list of tuples
        output_table = [tuple(x) for x in output_table.to_numpy()]
        ground_truth_sql_table = [tuple(x) for x in ground_truth_sql_table.to_numpy()]
        if set(output_table) == set(ground_truth_sql_table):
            execution_accuracy = 1

    except Exception as e:
        print(f"Exception while load json: {e}")
    return [index, execution_accuracy]


    

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

def averaging_saving_print_results(results, columns, averaging_mode, partial_file_path, result_dir):
    df_results = pd.DataFrame(results, columns=columns)
    if averaging_mode not in ["execution_accuracy", "table_verdict"]:
        averages = average_results(df_results, averaging_mode)
        averages.to_csv(result_dir / f"summarized_results__{averaging_mode}__{partial_file_path}.csv", sep=',', index=False)
    df_results.to_csv(result_dir / f"metrics_results__{averaging_mode}__{partial_file_path}.csv", sep=',', index=False)
    print(f"Detailed {averaging_mode} metrics are:")
    print(df_results)
    if averaging_mode not in ["execution_accuracy", "table_verdict"]:
        print(f"Summarized {averaging_mode} metrics are:")
        print(averages)
    return df_results

def check_all_zeros(list):
    for value in list:
        if value > 0:
            return False
    return True

def evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic, valentine = True, llm = False, unified = False, fullname_split=False, execution_accuracy=True, judge_table_result=True):
    # create the result dir folder
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)

    # skip if there are files in the directory that start with "metrics_results" or "summarized_results"
    if len(list(result_dir.glob("metrics_results*.csv"))) > 0 or len(list(result_dir.glob("summarized_results*.csv"))) > 0:
        print(f"Results already present, skipping computing metrics with {enterprise} on {database}")
        return


    safe_model = str(model.replace("-", "_"))
    partial_file_path = f"{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}"
    eval_results = pd.read_csv(result_dir / f"evaluation_results__{partial_file_path}.csv")
    
    if valentine or execution_accuracy:
        db = GetDataFromDatabase()
        if automatic:
            db.open_connection(f"data_service_bird_automatic/train_databases/{database}/{database}.sqlite")
        else:
            db.open_connection(f"data_service_bird/{database}/{database}.sqlite")
    
    if valentine:
        agent = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="wo_pipeline")
        metrics_res = []
        
    if llm:
        judge = Judge(enterprise, model, mode="verdict")
        verdict_res = []

    if execution_accuracy:
        execution_accuracy_res = []

    if judge_table_result:
        judge_table = Judge(enterprise, model, mode="verdict_no_sql")
        table_verdict_res = []

    #eval_results = pd.read_csv(result_dir / f"evaluation_results_{mode}.csv")
    num_queries = len(queries)

    
    if valentine or llm or judge_table_result or execution_accuracy:
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
                except Exception as e:
                    print(f"Error in query {index}: {e}")
                    print(f"Probably LLM rate exceeded. Waiting {PIPELINE_GENERATION_RETRY_DELAY_SEC} seconds and retrying.")
                    time.sleep(PIPELINE_GENERATION_RETRY_DELAY_SEC)
                    verdict = judge.judge(res["pipeline"], sql, question)
                    verdict_res.append([index, verdict])
                    
                print(f"Verdict is: {verdict}")
                if enterprise == "Mistral":
                    time.sleep(0.3)
            
            if execution_accuracy:
                execution_accuracy_res.append(compute_execution_accuracy(index, sql, db, res, verbose))
            
            if judge_table_result:
                verdict = "NON-CORRECT"
                # parse the result table
                output = res["output_json"]
                #Check if pipeline completely failed like query 35 eval_26-11-24
                if type(res["output_json"].values[0]) != str:
                    output_json = ""
                else:
                    output_json = res["output_json"].values[0]
                    output_json = output_json.replace('\\xa0', '')
                    output_json = output_json.replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")
                    
                if output_json == "":
                    if verbose:
                        print("Empty pipeline result, probably failed execution. Not performing any match.")
                    table_verdict_res.append([index, verdict])
                    continue
                else:
                    try:
                        # truncate the output to 5 rows
                        num_of_entries = 5
                        output_res = json.loads(output_json)[:num_of_entries]
                    except Exception as e:
                        print(e)
                        print("Exception while load json")
                        table_verdict_res.append([index, verdict])
                        continue
                try:
                    verdict = judge_table.judge(pipeline=res["pipeline"], sql=sql, question=question, view=output_res)
                except Exception as e:
                    print(f"Error in query {index}: {e}")
                    print(f"Probably LLM rate exceeded. Waiting {PIPELINE_GENERATION_RETRY_DELAY_SEC} seconds and retrying.")
                    time.sleep(PIPELINE_GENERATION_RETRY_DELAY_SEC)
                    verdict = judge_table.judge(pipeline=res["pipeline"], sql=sql, query=question, view=output_res)

                table_verdict_res.append([index, verdict])


    if valentine:
        columns = ["index", "precision", "recall", "acc_cell", "acc_row"]
        metrics_res = averaging_saving_print_results(metrics_res, columns, "valentine", partial_file_path, result_dir)
    
    if llm:
        columns = ["index", "verdict"]
        verdict_res = averaging_saving_print_results(verdict_res, columns, "llm", partial_file_path, result_dir)

    if execution_accuracy:
        columns = ["index", "execution_accuracy"]
        execution_accuracy_res = averaging_saving_print_results(execution_accuracy_res, columns, "execution_accuracy", partial_file_path, result_dir)
    
    if judge_table_result:
        columns = ["index", "verdict"]
        table_verdict_res = averaging_saving_print_results(table_verdict_res, columns, "table_verdict", partial_file_path, result_dir)
    
    if unified:
        print("Unifying valentin and llm results.\n    Ignoring MISLEADING results and setting metrics to 1 if TRUE")
        if not valentine:
            print("Reading past valentine results")
            metrics_res = pd.read_csv(result_dir / f"metrics_results_valentine_{partial_file_path}.csv")
        if not llm:
            print("Reading past llm results")
            verdict_res = pd.read_csv(result_dir / f"metrics_results_llm_{partial_file_path}.csv")
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
    # create the parser and add the arguments
    parser = ArgumentParser()
    parser.add_argument('--database', type=str, default="chicago_crime")
    parser.add_argument('--model', type=str, choices=["openai", "mistral", "anthropic", "deepseek"], default="openai")
    parser.add_argument('--generate', type=bool, default=True)

    # parse the arguments
    args = parser.parse_args()

    # get enterprise and model
    parsed_model = args.model
    if parsed_model == "openai":
        enterprise = "Openai"
        model = "gpt-4o"
    elif parsed_model == "mistral":
        enterprise = "Mistral"
        model = "mistral-large-latest"
    elif parsed_model == "anthropic":
        enterprise = "Anthropic"
        model = "claude-3-5-sonnet-latest"
    elif parsed_model == "deepseek":
        enterprise = "Deepseek"
        model = "deepseek-chat"

    # get the database
    database= args.database

    # get the generate
    generate_again = args.generate
    print(f"Model: {model}, Database: {database}, Generate: {generate_again}")

    ## quali data services sono utilizzati (lasciare a True perchè utilizziamo quelli generati automaticamente)
    automatic = True
    print(f"Data Services are generated: {automatic}")
    
    ## per la stampa in output
    verbose = True
    print(f"Verbose: {verbose}")
    
    ## evaluation sul selector
    ground_truth_check = False  # per fare l'evaluation sul selector
    ground_truth_check_mode = "no_view" # "no_view" non gli passo la view, "with_view" gli passo la view
    print(f"Data service selection evaluation: {ground_truth_check}, Data service selection mode: {ground_truth_check_mode}")
    
    queries = get_queries(database)
    print(f"Got {len(queries)} queries")

    generate_again = False
    print("Performing ground truth check")  # evaluation sul selector!!!!
    if generate_again:
        run_evaluation_ground_truth(database, enterprise, ground_truth_check_mode, model, queries, automatic=automatic, verbose = verbose)
    evaluate_ground_truth(database, enterprise, model, ground_truth_check_mode)
        
