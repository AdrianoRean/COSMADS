import json
import os
import re
import pandas as pd

from data_service_bird.database import GetDataFromDatabase

from main import LLMAgent
from evaluation.match_similarity import match_similarity

def run_evaluation(mode, second_mode, num_of_pipelines):
    llm = LLMAgent(mode, second_mode)
    llm_chains = llm.get_chain_all(num_of_pipelines)

    main_queries = json.load(open("queries/test/human_resources.json"))

    res_eval = []
    for index, query in enumerate(main_queries):
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"]
        }
        question = query["question"]
        print(question)
        for index_pipeline in range(num_of_pipelines):
            print(index_pipeline)
            try:
                res = llm_chains[index_pipeline].invoke(input_file)
                data_services = res['data_services']
                pipeline =  res["pipeline"]
                output = res["output"]

                example_query = res["example_query"]
                example_pipeline = res["example_pipeline"]
                
                try:
                    advice = json.loads(open(f"advice_{index_pipeline}.json", "r").read())
                except:
                    advice = "No advice or error"
                output_json = json.loads(open(f"result_{index_pipeline}.json", "r").read())

                res_elem = [index, index_pipeline, question, data_services, advice, pipeline, output, output_json, example_query, example_pipeline]
            except Exception as e:
                print(f"Error in query {index} pipe {index_pipeline}: {e}")
                res_elem = [index, index_pipeline, question, None, None, None, None, None, None, None]
            res_eval.append(res_elem)

    res_df = pd.DataFrame(res_eval, columns=["index", "index_pipeline", "question", "data_services", "advice", "pipeline", "output", "output_json", "example_query", "example_pipeline"])
    res_df.to_csv(f"evaluation/evaluation_results_{mode}.csv", sep=',', index=False)


def evaluate_results(mode, num_of_pipelines):
    queries = json.load(open("queries/test/human_resources.json"))
    db = GetDataFromDatabase()
    db.open_connection("data_service_bird/human_resources/human_resources.sqlite")
    

    metrics_res = []
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    
    for pipe in range(num_of_pipelines):
        metrics_res = []
        print(f"Pipeline {pipe}")
        for index, query in enumerate(queries):
            metrics_res_q_idx = []
            question = query["question"]
            df1 = db.call(query=query["SQL"])
            #print(f"Query {question} index {index}")
            res = eval_results[(eval_results["index"] == index) & (eval_results["index_pipeline"] == pipe)]
            #print(res)
            #Check if pipeline completely failed like query 35 eval_26-11-24
            if type(res["output_json"].values[0]) != str:
                output_json = "[]"
            else:
                output_json = res["output_json"].values[0].replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")
                pattern = r'(".+\$)".+"'
                output_json = re.sub(pattern, r'\1.+"', output_json)
            #print(output_json)
            output_res = json.loads(output_json)
            df2 = pd.DataFrame(output_res)

            df1.name = "table_1"
            df2.name = "table_2"
            try:
                precision, recall, acc_cell, acc_row = match_similarity(df1, df2)
            except:
                #print(f"Exception for query {index}")
                #print(f"Exception for query {index}")
                precision, recall, acc_cell, acc_row = 0, 0, 0, 0
            metrics_res_q_idx.append([precision, recall, acc_cell, acc_row])
            
            # average metrics
            metrics_res_q_idx = pd.DataFrame(metrics_res_q_idx, columns=["precision", "recall", "acc_cell", "acc_row"])
            metrics_res_q_idx = metrics_res_q_idx.mean()

            # append to the final results with the query index
            metrics_res_q_idx = [index] + metrics_res_q_idx.to_list()
            metrics_res.append(metrics_res_q_idx)

        metrics_res = pd.DataFrame(metrics_res, columns=["index", "precision", "recall", "acc_cell", "acc_row"])
        metrics_res.to_csv(f"evaluation/metrics_results_{pipe}_{mode}.csv", sep=',', index=False)
        print(metrics_res)
    return

if __name__ == "__main__":

    
    '''modes = ["standard", "wo_pipeline", "wrong"]
    for mode in modes:
        run_evaluation(mode)
        evaluate_results(mode)
    
    evaluate_results("copilot")'''
    
    modes = ["standard_view"]
    second_mode = "standard_evidence"
    
    pipe_dir = "pipelines_bird/human_resources"
    num_of_pipelines = len([name for name in os.listdir(pipe_dir) if os.path.isfile(os.path.join(pipe_dir, name))])
    
    for pipe in range(num_of_pipelines):
        with open(f"advice_{pipe}.json", "w") as file:
        # Use the `truncate()` method to clear the file's content
            file.truncate()
        with open(f"result_{pipe}.json", "w") as file:
        # Use the `truncate()` method to clear the file's content
            file.truncate()
    
    for mode in modes:
        #run_evaluation(mode, second_mode, num_of_pipelines)
        evaluate_results(mode, num_of_pipelines)
