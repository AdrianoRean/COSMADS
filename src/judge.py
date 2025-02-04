import ast
import time
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
import dotenv
import os
import json
import pandas as pd
from main import get_queries
from templates import JUDGE_PROMPT, JUDGE_VERDICT, JUDGE_EXPLAIN
from model import getModel

class CustomOutputParser(BaseOutputParser):
    """The output parser for the LLM."""
    def parse(self, text: str) -> str:
        text = text.strip("\n")
        text = text.strip()
        # count how many ``` are in the text
        back_count = text.count("```")
        if back_count != 2:
            print(text)
            raise ValueError("The string should contain exactly two triple backticks")
        verdict = text.split("```")[1]
        #print(code)
        return verdict

class ChainGeneratorAgent:
    def __init__(self, enterprise, model):
        """Initialize the agent."""
        prompt_template = JUDGE_PROMPT
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # define the LLM
        self.model = model
        self.llm = getModel(enterprise, model)
        # define the output parser
        self.output_parser = CustomOutputParser()

    def get_chain(self):
        # generate the python function
        agent_chain = self.prompt | self.llm | self.output_parser
        return agent_chain
    
class Judge:
    def __init__(self, enterprise, model, mode="verdict"):
        self.enterprise = enterprise
        self.model = model
        self.mode = mode
        if mode == "verdict":
            self.judge_mode_prompt = JUDGE_VERDICT
        elif mode == "explain":
            self.judge_mode_prompt = JUDGE_EXPLAIN
        
        self.generator_chain_output = {
            "output": ChainGeneratorAgent(self.enterprise, self.model).get_chain(),
            "inputs": RunnablePassthrough()
            }
        
    def judge(self, pipeline, sql, query):
        chain = (
            RunnableLambda( 
                    lambda x: {
                        "pipeline": x[0],
                        "sql": x[1],
                        "query": x[2],
                        "judge_mode_prompt" : self.judge_mode_prompt
                    }
                )
                | self.generator_chain_output
        )
        
        verdict = chain.invoke((pipeline, sql, query))["output"].strip()
        if self.mode == "verdict" and verdict not in ["EQUIVALENT", "NOT-EQUIVALENT", "SQL-WRONG"]:
            print("Verdict failed: ")
            print(verdict)
            verdict = "FAILED"
        
        return verdict
    
if __name__ == "__main__":
    
    enterprise="Mistral"
    model = "mistral-large-latest"
    database="chicago_crime"
    print(f"Model: {model}, Database: {database}")
    
    pipeline_mode = "wo_pipeline_view"
    evidence_mode = "standard_evidence"
    dataservice_mode = "ground_truth"
    print(f"Pipeline mode: {pipeline_mode}, Evidence mode: {evidence_mode}, Data service mode: {dataservice_mode}")
    
    queries = get_queries(database)
    print(f"Got {len(queries)} queries")
    
    safe_model = str(model.replace("-", "_"))
    partial_file_path = f"{database}_{enterprise}_{safe_model}_{pipeline_mode}_{evidence_mode}_{dataservice_mode}"
    eval_results = pd.read_csv(f"evaluation/evaluation_results_{partial_file_path}.csv")
    
    judge = Judge(enterprise, model, mode="explain")
    verdict_res = []

    #eval_results = pd.read_csv(f"evaluation/evaluation_results_{mode}.csv")
    num_queries = len(queries)
    
    for index, query in enumerate(queries):
            
        question = query["question"]
        print(f"Index {index} of {num_queries}, Question: {question}")
        
        sql = query["SQL"]
        res = eval_results[(eval_results["index"] == index)]
        
        try:
            result = judge.judge(res["pipeline"], sql, question)
        except:
            print("Probably LLM rate exceeded. Waiting 2 seconds and retrying.")
            time.sleep(2)
            result = judge.judge(res["pipeline"], sql, question)
            
        print("Explanation and result:")
        print(result)
        print("\n ******************* \n")
        verdict_res.append([index, result])
        
        if enterprise == "Mistral":
            time.sleep(0.2)
            
    verdict_res = pd.DataFrame(verdict_res, columns=["index", "Explaination and result"])
    verdict_res.to_csv(f"evaluation/metrics_results_explanation_{partial_file_path}.csv", sep=',', index=False)