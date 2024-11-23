import json
import pandas as pd
import os

from data_service_bird.database import GetDataFromDatabase

from main import LLMAgent
from evaluation.match_similarity import match_similarity

def run_evaluation(mode):
    llm = LLMAgent(mode)
    if mode == "wrong":
        llm_chain = llm.get_chain_wrong()
    else:
        llm_chain = llm.get_chain()

    main_queries = json.load(open("queries/test/human_resources.json"))

    res_eval = []
    for index, query in enumerate(main_queries):
        question = query["question"]
        print(question)
        try:
            res = llm_chain.invoke(question)

            data_services = res['data_services']
            pipeline =  res["pipeline"]
            output = res["output"]

            example_query = res["example_query"]
            example_pipeline = res["example_pipeline"]

            output_json = json.loads(open("result.json", "r").read())

            res_elem = [index, question, data_services, pipeline, output, output_json, example_query, example_pipeline]
        except Exception as e:
            print(f"Error in query {index}: {e}")
            res_elem = [index, question, None, None, None, None, None, None]
        res_eval.append(res_elem)

    res_df = pd.DataFrame(res_eval, columns=["index", "question", "data_services", "pipeline", "output", "output_json", "example_query", "example_pipeline"])
    res_df.to_csv(f"evaluation/evaluation_results_{mode}.csv", sep=',', index=False)


def evaluate_results(mode):
    queries = json.load(open("queries/test/human_resources.json"))
    db = GetDataFromDatabase()
    db.open_connection("data_service_bird/human_resources/human_resources.sqlite")

    metrics_res = []
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    for index, query in enumerate(queries):
        metrics_res_q_idx = []
        question = query["question"]
        df1 = db.call(query=query["SQL"])
        print(f"Query {question} index {index}")
        res = eval_results[(eval_results["index"] == index)]
        
        try:
            output_json = res["output_json"].values[0].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false")
            output_res = json.loads(output_json)
            df2 = pd.DataFrame(output_res)
            print("-------------")
            print(df1)
            print("-------------")
            print(df2)
            
            exit()
            precision, recall, acc_cell, acc_row = match_similarity(df1, df2)
        except:
            precision, recall, acc_cell, acc_row = 0, 0, 0, 0
        metrics_res_q_idx.append([precision, recall, acc_cell, acc_row])
        
        # average metrics
        metrics_res_q_idx = pd.DataFrame(metrics_res_q_idx, columns=["precision", "recall", "acc_cell", "acc_row"])
        metrics_res_q_idx = metrics_res_q_idx.mean()

        # append to the final results with the query index
        metrics_res_q_idx = [index] + metrics_res_q_idx.to_list()
        metrics_res.append(metrics_res_q_idx)

    metrics_res = pd.DataFrame(metrics_res, columns=["index", "precision", "recall", "acc_cell", "acc_row"])
    metrics_res.to_csv(f"evaluation/metrics_results_{mode}.csv", sep=',', index=False)
    print(metrics_res)
    return

if __name__ == "__main__":
    '''modes = ["standard", "wo_pipeline", "wrong"]
    for mode in modes:
        run_evaluation(mode)
        evaluate_results(mode)
    
    evaluate_results("copilot")'''
    
    modes = ["standard"]
    for mode in modes:
        #run_evaluation(mode)
        evaluate_results(mode)
