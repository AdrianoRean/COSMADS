import os
import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableBranch

PIPELINE_RESULT_FILEPATH = Path(__file__).parent / "result.json"

class PipelineRunner:

    def run_pipeline(self, pipeline_filepath: str) -> dict:
        execution_ok = False

        cwd = Path.cwd()    # original directory path
        
        os.chdir(Path(pipeline_filepath).parent)    # change to pipeline directory
        
        execution_result = os.system(f"python {pipeline_filepath}")
        
        if execution_result == 0:
            execution_ok = True

        os.chdir(cwd)

        return execution_ok

    def parse_pipeline_result(self, pipeline_result_filepath: str) -> dict:
        with open(pipeline_result_filepath, "r") as f:
            result = json.load(f)
        
        result = pd.DataFrame(result)
        result = tabulate(result, headers='keys', tablefmt='psql')
        return result

    def get_chain(self) -> Runnable:
        path = str(PIPELINE_RESULT_FILEPATH)
            
        runner_chain = (
            RunnableLambda(lambda x: {
                "execution_ok": self.run_pipeline(x["pipeline_filepath"])
            })
            | RunnableBranch(
                (lambda x: x["execution_ok"], RunnableLambda(lambda x: self.parse_pipeline_result(path))),
                (RunnableLambda(lambda x: "The pipeline did not run successfully"))
            )
        )
        return runner_chain
