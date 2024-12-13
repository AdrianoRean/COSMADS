from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from templates import *

class CustomOutputParser(BaseOutputParser):
    """The output parser for the LLM."""
    def parse(self, text: str) -> str:
        #print("----------------------------------------------")
        #print(text)
        text = text.strip("\n")
        text = text.strip()
        # count how many ``` are in the text
        back_count = text.count("```")
        if back_count != 2:
            print(text)
            raise ValueError("The string should contain exactly two triple backticks")
        splitted_text = text.split("```")
        action = splitted_text[0]
        code = splitted_text[1]
        #print(code)
        return action, code


class PipelineGeneratorAgent:
    """The agent that designs the pipeline."""

    def __init__(self, openai_key, mode = "standard"):
        """Initialize the agent."""
        # define the prompt
        if mode == "standard":
            prompt_template = TEMPLATE_WITH_DOCUMENT
        elif mode == "chain_of_tables":
            prompt_template = TEMPLATE_WITH_DOCUMENT_TABLES
        elif mode == "chain_of_thoughs_post":
            prompt_template = TEMPLATE_WITH_DOCUMENT_POST_REASONING
        elif mode == "chain_of_thoughs":
            prompt_template = TEMPLATE_WITH_DOCUMENT_PRE_REASONING
        elif mode == "wo_pipeline":
            prompt_template = TEMPLATE_WITHOUT_PIPELINE
        elif mode == "wrong":
            prompt_template = TEMPLATE_WITH_DOCUMENT
        else:
            raise ValueError(f"Mode {mode} is not recognized.")
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # define the LLM
        self.llm = ChatOpenAI(model="gpt-4o",
                              api_key=openai_key,
                              temperature=0.0)
        # define the output parser
        self.output_parser = CustomOutputParser()

    def get_chain(self):
        # generate the python function
        agent_chain = self.prompt | self.llm | self.output_parser
        return agent_chain


if __name__ == "__main__":
    generator = PipelineGeneratorAgent()
    chain = generator.get_chain()
    
    query ={
        "query": "Generate a table containing the max speed of the diemachine with id 25 over a time span of 30 seconds.",
        "data_services": []
    }

    result = chain.invoke(query)
    print(result)