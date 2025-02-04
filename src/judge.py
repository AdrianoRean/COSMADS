import ast
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableLambda, RunnablePassthrough
import dotenv
import os
import json
import pandas as pd
from templates import JUDGE_PROMPT
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
        verdict = text.split("```")[1].strip()
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
    def __init__(self, enterprise, model):
        self.enterprise = enterprise
        self.model = model
        
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
                        "query": x[2]
                    }
                )
                | self.generator_chain_output
        )
        
        verdict = chain.invoke((pipeline, sql, query))["output"]
        if verdict not in ["TRUE", "FALSE", "MISLEADING"]:
            print("Verdict failed: ")
            print(verdict)
            verdict = "FAILED"
        
        return verdict
    
    