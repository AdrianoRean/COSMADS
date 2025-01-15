import ast
import json
import re
import pandas as pd

from data_service_bird.database import GetDataFromDatabase

from main import LLMAgent
from evaluation.match_similarity import match_similarity

def run_evaluation_ground_truth():
    llm = LLMAgent(mode="check_ground_truth")
    llm_chain = llm.get_chain_truth()

    main_queries = json.load(open("queries/test/human_resources.json"))

    res_eval = []
    for index, query in enumerate(main_queries):
        sql = query["SQL"]
    
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"],
            "ground_truth" : sql
        }
        question = query["question"]
        print(question)
        res = llm_chain.invoke(input_file)
        res_eval.append([index, res["output"], res["ground_truth"]])
    res_df = pd.DataFrame(res_eval, columns=["index", "tools_prediction", "ground_truth"])
    res_df.to_csv(f"evaluation/evaluation_results_check_ground_truth.csv", sep=',', index=False)
    
def evaluate_ground_truth():
    metrics_res = []
    eval_results = pd.read_csv(f"evaluation/evaluation_results_check_ground_truth_fixed.csv")
    
    def modify_tools_prediction(row):
        # Convert string to list
        try:
            words = row[1:-1].split(",")
            # Add double quotes to each word
            modified = [word.strip() for word in words]
            # Convert back to string
            return modified
        except Exception as e:
            print(f"Error processing row: {row}, error: {e}")
            return []

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
    metrics_res.to_csv(f"evaluation/metrics_results_check_ground_truth.csv", sep=',', index=False)
    print(metrics_res)
        

def run_evaluation(mode, second_mode, services_mode = None):
    llm = LLMAgent(mode, second_mode, services_mode = services_mode)
    if mode == "wrong":
        llm_chain = llm.get_chain_wrong()
    else:
        llm_chain = llm.get_chain()

    main_queries = json.load(open("queries/test/human_resources.json"))

    res_eval = []
    for index, query in enumerate(main_queries):
        sql = query["SQL"]
    
        input_file = {
            "query" : query["question"],
            "evidence" : query["evidence"],
            "ground_truth" : sql
        }
        question = query["question"]
        print(question)
        try:
            res = llm_chain.invoke(input_file)

            data_services = res['data_services']
            pipeline =  res["pipeline"]
            output = res["output"]

            example_query = res["example_query"]
            example_pipeline = res["example_pipeline"]
            
            try:
                advice = json.loads(open("advice.json", "r").read())
            except:
                advice = "No advice or error"
            output_json = json.loads(open("result.json", "r").read())

            res_elem = [index, question, sql, data_services, advice, pipeline, output, output_json, example_query, example_pipeline]
        except Exception as e:
            print(f"Error in query {index}: {e}")
            res_elem = [index, question, sql, None, None, None, None, None, None, None]
        res_eval.append(res_elem)

    res_df = pd.DataFrame(res_eval, columns=["index", "question", "sql", "data_services", "advice", "pipeline", "output", "output_json", "example_query", "example_pipeline"])
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
        #Check if pipeline completely failed like query 35 eval_26-11-24
        if type(res["output_json"].values[0]) != str:
            output_json = "[]"
        else:
            output_json = res["output_json"].values[0].replace("'", "\"").replace("None", "null").replace("nan", "\"nan\"").replace("True", "true").replace("False", "false").replace("\"\"", "\"")
            pattern = r'(".+\$)".+"'
            output_json = re.sub(pattern, r'\1.+"', output_json)
        print(output_json)
        output_res = json.loads(output_json)
        df2 = pd.DataFrame(output_res)

        df1.name = "table_1"
        df2.name = "table_2"
        try:
            precision, recall, acc_cell, acc_row = match_similarity(df1, df2)
        except:
            print(f"Exception for query {index}")
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
    with open("advice.json", "w") as file:
    # Use the `truncate()` method to clear the file's content
        file.truncate()
    with open("result.json", "w") as file:
    # Use the `truncate()` method to clear the file's content
        file.truncate()
        
    '''modes = ["standard", "wo_pipeline", "wrong"]
    for mode in modes:
        run_evaluation(mode)
        evaluate_results(mode)
    
    evaluate_results("copilot")'''
    
    modes = ["wo_pipeline_view"]
    second_mode = "added_evidence"
    for mode in modes:
        run_evaluation(mode, second_mode, services_mode="ground_truth")
        evaluate_results(mode)
        
    #run_evaluation_ground_truth()
    #evaluate_ground_truth()
