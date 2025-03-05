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
from main import LLMAgent
from evaluation.match_similarity import match_similarity
from judge import Judge


PIPELINE_GENERATION_DELAY_SEC = 2 
PIPELINE_GENERATION_RETRY_DELAY_SEC = 5


def parse_in_action_queries():
    # open the file with the queries
    with open("queries/in_action_queries.json", "r") as f:
        queries = json.load(f)
    parsed_queries = []
    for query_id in queries:
        query_list = queries[query_id]
        for query in query_list:
            parsed_query = {
                "question": query,
                "SQL": query_id,
                "evidence": ""
            }
            parsed_queries.append(parsed_query)
    return parsed_queries
        

def run_evaluation(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = None, automatic=False, similarity_treshold = 0.9, verbose=False):
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(model.replace("-", "_"))
    result_filepath = result_dir / f"evaluation_results__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv"
    print(f"Result file path: {result_filepath}")

    # skip if the file already exists
    if result_filepath.exists():
        print(f"Results already present, skipping generating the pipeline with {enterprise} on {database}")
        return

    llm = LLMAgent(enterprise, model, pipeline_mode, evidence_mode, dataservice_mode = dataservice_mode, similarity_treshold=similarity_treshold, automatic=automatic, database=database, verbose=verbose)
    llm_chain = llm.get_chain_in_action()
    
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

    res_df = pd.DataFrame(res_eval, columns=["index", "question", "sql", "data_services", "pipeline", "output", "output_json"])
    res_df.to_csv(result_filepath, sep=',', index=False)


def compute_execution_accuracy(index, sql, res, verbose = False):
    execution_accuracy = 0
    # retrieve the ground truth table
    ground_truth_filepath = Path("ground_truth") / "in_action" / sql / "result.json"
    ground_truth_df = pd.read_json(ground_truth_filepath)
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

        if output_table.shape != ground_truth_df.shape:
            if verbose:
                print("Different shapes of the two tables")
            return [index, execution_accuracy]
        # sort the columns alphabetically for comparison
        output_table = output_table[sorted(output_table.columns)]
        ground_truth_df = ground_truth_df[sorted(ground_truth_df.columns)]
        # convert both tables to list of tuples
        output_table = [tuple(x) for x in output_table.to_numpy()]
        ground_truth_df = [tuple(x) for x in ground_truth_df.to_numpy()]
        if set(output_table) == set(ground_truth_df):
            execution_accuracy = 1

    except Exception as e:
        print(f"Exception while load json: {e}")
    return [index, execution_accuracy]


def metrics_valentine(index, sql, res, fullname_split, agent, verbose = False):
    ground_truth_filepath = Path("ground_truth") / "in_action" / sql / "result.json"
    df1 = pd.read_json(ground_truth_filepath)
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

def evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic, valentine = True, unified = False, fullname_split=False, execution_accuracy=True, judge_table_result=True):
    # create the result dir folder
    result_dir = Path(__file__).parent / "evaluation" / database / enterprise
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(model.replace("-", "_"))


    # flag to check if the result files are present
    valentine_result_filepath = result_dir / f"metrics_results__valentine__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv"
    valentine_summarized_result_filepath = result_dir / f"summarized_results__valentine__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv"
    is_valentine_result_present = valentine_result_filepath.exists() and valentine_summarized_result_filepath.exists()

    execution_accuracy_result_filepath = result_dir / f"metrics_results__execution_accuracy__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv"
    is_execution_accuracy_result_present = execution_accuracy_result_filepath.exists()

    judge_table_result_filepath = result_dir / f"metrics_results__table_verdict__{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}.csv"
    is_judge_table_result_present = judge_table_result_filepath.exists()

    # if all the results are present, then return
    if is_valentine_result_present and \
        is_execution_accuracy_result_present and \
            is_judge_table_result_present:
        print(f"Results already present, skipping computing metrics with {enterprise} on {database}")
        return


    partial_file_path = f"{database}__{enterprise}__{safe_model}__{pipeline_mode}__{evidence_mode}__{dataservice_mode}"
    eval_results = pd.read_csv(result_dir / f"evaluation_results__{partial_file_path}.csv")
    
    if valentine and not is_valentine_result_present:
        agent = LLMAgent(enterprise=enterprise, model=model, pipeline_mode="wo_pipeline")
        metrics_res = []

    if execution_accuracy and not is_execution_accuracy_result_present:
        execution_accuracy_res = []

    if judge_table_result and not is_judge_table_result_present:
        judge_table = Judge(enterprise, model, mode="verdict_no_sql")
        table_verdict_res = []

    num_queries = len(queries)

    for index, query in enumerate(queries):
        question = query["question"]
        print(f"Index {index} of {num_queries}, Question: {question}")
        
        res = eval_results[(eval_results["index"] == index)]
        sql = query["SQL"]
        
        if valentine and not is_valentine_result_present:   
            metrics_res.append(metrics_valentine(index, sql, res, fullname_split, agent, verbose))
        
        if execution_accuracy and not is_execution_accuracy_result_present:
            execution_accuracy_res.append(compute_execution_accuracy(index, sql, res, verbose))
        
        if judge_table_result and not is_judge_table_result_present:
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


    if valentine and not is_valentine_result_present:
        columns = ["index", "precision", "recall", "acc_cell", "acc_row"]
        metrics_res = averaging_saving_print_results(metrics_res, columns, "valentine", partial_file_path, result_dir)

    if execution_accuracy and not is_execution_accuracy_result_present:
        columns = ["index", "execution_accuracy"]
        execution_accuracy_res = averaging_saving_print_results(execution_accuracy_res, columns, "execution_accuracy", partial_file_path, result_dir)
    
    if judge_table_result and not is_judge_table_result_present:
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
    parser.add_argument('--model', type=str, choices=["openai", "mistral", "anthropic", "deepseek"], default="openai")
    
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

    # set the database name
    database= "cardboard_production"
    print(f"Model: {model}, Database: {database}")
    
    automatic = False
    print(f"Data Services are generated: {automatic}")
    
    ## per bert similarity (quanto voglio che siano simili gli embeddings per farglieli cambiare)
    similarity_treshold = 0.9
    print(f"Bert similarity treshold: {similarity_treshold}")
    
    ## per la stampa in output
    verbose = True
    print(f"Verbose: {verbose}")
    
    ## per la valutazione
    only_metrics = False    # se è true, allora runno solo evaluation (metrics) sia per il selector che per la pipeline, se è false runno tutto (rigenero anche i risultati)
    valentine = True    # se le metriche devono essere valutate su valentine
    llm = True # se le metriche devono essere valutate su llm judge
    unified = False # misto tra i due
    execution_accuracy = True 
    judge_table_result = False
    print(f"Only calculating metrics: {only_metrics}, Valentine metrics: {valentine}, Judge metrics: {llm}, Judge table result: {judge_table_result}, Unified metrics: {unified}")
    
    ## evaluation sulla pipeline
    pipeline_mode = "in_action" # "wo_pipeline_view" non gli passo la pipeline ma gli passo la view, "wo_pipeline" non gli passo la pipeline e neanche la view (NON TOCCARE!)
    evidence_mode = "standard_evidence" # "standard_evidence" gli passo ciò che sta in bird, "added_evidence" DA IGNORARE
    dataservice_mode = "tutti"   # "ground_truth" gli passo il ground truth (da SQL), l'altro non ha nome ma significa che gli passo tutti i data services
    print(f"Pipeline mode: {pipeline_mode}, Evidence mode: {evidence_mode}, Data service mode: {dataservice_mode}")
    
    queries = parse_in_action_queries()
    print(f"Got {len(queries)} queries")
        
    print("Performing pipeline check")

    with open("result.json", "w") as file:
    # Use the `truncate()` method to clear the file's content
        file.truncate()
    
    if not only_metrics:
        run_evaluation(database, 
                        queries, 
                        enterprise, 
                        model, 
                        pipeline_mode, 
                        evidence_mode, 
                        dataservice_mode=dataservice_mode, 
                        automatic=automatic, 
                        similarity_treshold=similarity_treshold, 
                        verbose=verbose)
    evaluate_results(database, queries, enterprise, model, pipeline_mode, evidence_mode, dataservice_mode, automatic=automatic, fullname_split=False, valentine=valentine, unified=unified, execution_accuracy=execution_accuracy, judge_table_result=judge_table_result)
    
    
