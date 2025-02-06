from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.output_parsers import StrOutputParser
from templates import *
from model import getModel

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

    def __init__(self, enterprise, model, mode = "standard", additional_mode = None):
        """Initialize the agent."""
        # define the prompt
        if mode == "check_ground_truth":
            if additional_mode != "with_view":
                prompt_template = TEMPLATE_WITH_DOCUMENT_TOOLS
            else:
                prompt_template = TEMPLATE_WITH_DOCUMENT_TOOLS_VIEW
        elif mode == "wo_pipeline_view":
            prompt_template = TEMPLATE_WITHOUT_PIPELINE_BUT_VIEW
        elif mode == "wo_pipeline":
            prompt_template = TEMPLATE_WITHOUT_PIPELINE
        else:
            raise ValueError(f"Mode {mode} is not recognized.")
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        # define the LLM
        self.llm = getModel(enterprise, model)
        # define the output parser
        self.output_parser = CustomOutputParser()

    def get_chain(self):
        # generate the python function
        agent_chain = self.prompt | self.llm | self.output_parser
        return agent_chain


if __name__ == "__main__":
    import os
    import dotenv
    dotenv.load_dotenv()
    generator = PipelineGeneratorAgent(model="Mistral", key=os.getenv("MISTRAL_API_KEY"), mode="wo_pipeline")
    chain = generator.get_chain()
    
    query ={
        "query": "Generate a table containing the max speed of the diemachine with id 25 over a time span of 30 seconds.",
        "data_services": [],
        "evidence" : "",
        "DATA_SERVICE_SECTION" : DATA_SERVICE_SECTION
    }

    result = chain.invoke(query)
    print(result)