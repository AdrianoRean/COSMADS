import json
import os
import re
import pandas as pd

from data_service_bird.database import GetDataFromDatabase

from main import LLMAgent
from evaluation.match_similarity import match_similarity

def run_evaluation(mode, second_mode, num_of_pipelines):
    llm = LLMAgent(mode, second_mode, combinatory=True)
    llm_chains = llm.get_chain_combinatory(num_of_pipelines)

    main_queries = json.load(open("queries/test/human_resources.json"))
    
    for index, query in enumerate(main_queries):
        if index < 0:
            continue
        res_eval = []
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"]
        }
        question = query["question"]
        print(f"Index: {index}, QUestion {question}")
        for index_pipeline in range(num_of_pipelines):
            for index_pipeline_2 in range(num_of_pipelines):
                if index_pipeline == index_pipeline_2:
                    continue
                print(f"{index_pipeline} - {index_pipeline_2}")
                try:
                    res = llm_chains[index_pipeline][index_pipeline_2].invoke(input_file)
                    data_services = res['data_services']
                    pipeline =  res["pipeline"]
                    output = res["output"]

                    example_query = res["example_query"]
                    example_pipeline = res["example_pipeline"]
                    
                    try:
                        advice = json.loads(open(f"results/advice_{index_pipeline}_{index_pipeline_2}.json", "r").read())
                    except:
                        advice = "No advice or error"
                    output_json = json.loads(open(f"results/result_{index_pipeline}_{index_pipeline_2}.json", "r").read())

                    res_elem = [index, index_pipeline, index_pipeline_2, question, data_services, advice, pipeline, output, output_json, example_query, example_pipeline]
                except Exception as e:
                    print(f"Error in query {index} pipe {index_pipeline}: {e}")
                    res_elem = [index, index_pipeline, index_pipeline_2, question, None, None, None, None, None, None, None]
                res_eval.append(res_elem)

        res_df = pd.DataFrame(res_eval, columns=["index", "index_pipeline", "index_pipeline_2", "question", "data_services", "advice", "pipeline", "output", "output_json", "example_query", "example_pipeline"])
        res_df.to_csv(f"evaluation/combinatory/evaluation_results_{mode}_{index}.csv", sep=',', index=False)


def evaluate_results(mode, num_of_pipelines):
    queries = json.load(open("queries/test/human_resources.json"))
    db = GetDataFromDatabase()
    db.open_connection("data_service_bird/human_resources/human_resources.sqlite")
    

    for index, query in enumerate(queries):
        eval_results = pd.read_csv(f"evaluation/combinatory/evaluation_results_{mode}_{index}.csv")
        print(f"Query {index}")
        
        metrics_res_q_pipes = [[] for pipe in range(num_of_pipelines) for pipe_2 in range(num_of_pipelines)]
        
        for pipe in range(num_of_pipelines):
            for pipe_2 in range(num_of_pipelines):
                if pipe == pipe_2:
                        continue
                print(f"Pipeline {pipe} - {pipe_2}")
                df1 = db.call(query=query["SQL"])
                #print(f"Query {question} index {index}")
                res = eval_results[(eval_results["index"] == index) & (eval_results["index_pipeline"] == pipe) & (eval_results["index_pipeline_2"] == pipe_2)]
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
                metrics_res_q_pipes[num_of_pipelines * pipe + pipe_2].append([precision, recall, acc_cell, acc_row])
                
    total_metric_res = []
    for pipe in range(num_of_pipelines):
        for pipe_2 in range(num_of_pipelines):
            
            metrics_res = metrics_res_q_pipes[num_of_pipelines * pipe + pipe_2]
            print(metrics_res)
            metrics_df = pd.DataFrame(metrics_res, columns=["precision", "recall", "acc_cell", "acc_row"])
            averages = metrics_df.mean()

            total_metric_res = total_metric_res + metrics_res

            # Printing the results
            print(f"Average values for {pipe} - {pipe_2}:")
            print(f"Average acc_cell: {averages['acc_cell']:.6f}")
            print(f"Average acc_row: {averages['acc_row']:.6f}")
            print(f"Average recall: {averages['recall']:.6f}")
            
            metrics_df.to_csv(f"evaluation/combinatory/metrics_results_{mode}_{pipe}_{pipe_2}.csv", sep=',', index=False)
            
    total_metric_res = pd.DataFrame(total_metric_res, columns=["precision", "recall", "acc_cell", "acc_row"])
    averages = total_metric_res[['acc_cell', 'acc_row', 'recall']].mean()

    # Printing the results
    print(f"Average values for all combinatory:")
    print(f"Average acc_cell: {averages['acc_cell']:.6f}")
    print(f"Average acc_row: {averages['acc_row']:.6f}")
    print(f"Average recall: {averages['recall']:.6f}")
    
    total_metric_res.to_csv(f"evaluation/combinatory/metrics_results_{mode}.csv", sep=',', index=False)
            
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
    
    '''for pipe in range(num_of_pipelines):
        for pipe_2 in range(num_of_pipelines):
            with open(f"results/advice_{pipe}_{pipe_2}.json", "w") as file:
            # Use the `truncate()` method to clear the file's content
                file.truncate()
            with open(f"results/result_{pipe}_{pipe_2}.json", "w") as file:
            # Use the `truncate()` method to clear the file's content
                file.truncate()'''
    
    for mode in modes:
        #run_evaluation(mode, second_mode, num_of_pipelines)
        evaluate_results(mode, num_of_pipelines)
