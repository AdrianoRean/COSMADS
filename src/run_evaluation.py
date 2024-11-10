import json
import pandas as pd
import os

from main import LLMAgent
from evaluation.match_similarity import match_similarity

def run_evaluation(model, mode):
    llm = LLMAgent(model, mode)
    if mode == "wrong":
        llm_chain = llm.get_chain_wrong()
    else:
        llm_chain = llm.get_chain()

    main_queries = json.load(open("queries_pipelines.json"))

    queries = json.load(open("evaluation/rephrased_queries.json"))
    q_idxs = list(queries.keys())

    res_eval = []
    for q_idx in q_idxs:
        queries_idx = queries[q_idx]
        main_query = main_queries[q_idx]["query"]
        queries_idx = [main_query] + queries_idx
        variant = 0
        for query in queries_idx:
            print(query)
            try:
                res = llm_chain.invoke(query)

                data_services = res['data_services']
                pipeline =  res["pipeline"]
                output = res["output"]

                example_query = res["example_query"]
                example_pipeline = res["example_pipeline"]

                output_json = json.loads(open("result.json", "r").read())

                res_elem = [q_idx, variant, query, data_services, pipeline, output, output_json, example_query, example_pipeline]
            except Exception as e:
                print(f"Error in query {q_idx} variant {variant}: {e}")
                res_elem = [q_idx, variant, query, None, None, None, None, None, None]
            res_eval.append(res_elem)
            variant += 1

    res_df = pd.DataFrame(res_eval, columns=["q_idx", "variant", "query", "data_services", "pipeline", "output", "output_json", "example_query", "example_pipeline"])
    res_df.to_csv(f"evaluation/evaluation_results_{mode}.csv", sep=',', index=False)


def evaluate_results(mode):
    queries = json.load(open("evaluation/rephrased_queries.json"))
    q_idxs = list(queries.keys())

    metrics_res = []
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    variants = 10
    for q_idx in q_idxs:
        output_ref = json.loads(open(f"evaluation/ground_truth/{q_idx}/result.json", "r").read())
        metrics_res_q_idx = []
        df1 = pd.DataFrame(output_ref)
        for var in range(variants):
            print(f"Query {q_idx} variant {var}")
            res = eval_results[(eval_results["q_idx"] == q_idx) & (eval_results["variant"] == var)]
            output_json = res["output_json"].values[0].replace("'", "\"").replace("None", "null").replace("True", "true").replace("False", "false")
            output_res = json.loads(output_json)

            df2 = pd.DataFrame(output_res)

            try:
                precision, recall, acc_cell, acc_row = match_similarity(df1, df2)
            except:
                precision, recall, acc_cell, acc_row = 0, 0, 0, 0
            metrics_res_q_idx.append([precision, recall, acc_cell, acc_row])
        
        # average metrics
        metrics_res_q_idx = pd.DataFrame(metrics_res_q_idx, columns=["precision", "recall", "acc_cell", "acc_row"])
        metrics_res_q_idx = metrics_res_q_idx.mean()

        # append to the final results with the query index
        metrics_res_q_idx = [q_idx] + metrics_res_q_idx.to_list()
        metrics_res.append(metrics_res_q_idx)

    metrics_res = pd.DataFrame(metrics_res, columns=["q_idx", "precision", "recall", "acc_cell", "acc_row"])
    metrics_res.to_csv(f"evaluation/metrics_results_{mode}.csv", sep=',', index=False)
    print(metrics_res)
    return

if __name__ == "__main__":
    
    modes = ["standard", "wo_pipeline", "wrong"]
    #model = "llama3.2"
    model = "mistral"
    for mode in modes:
        run_evaluation(model, mode)
        evaluate_results(mode)
    
    evaluate_results("copilot")
