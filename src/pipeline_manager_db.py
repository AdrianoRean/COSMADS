from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
#from langchain_mistralai import MistralAIEmbeddings
import importlib.util
import os
import dotenv
import json


class PipelineStore():

    @staticmethod
    def tool_summary(cls, file_name):
        return cls.__name__ + " " + file_name + " " + cls.description['description']

    def __init__(self):
        print(os.getcwd())
        json_setup = json.load(open('queries/queries_pipelines_human_resources.json'))
        queries = json_setup.keys()
        self.docs = []
        for q in queries:
            self.docs.append(
                Document(
                    page_content = json_setup[q]['query_example'],
                    metadata = {"pipeline":json_setup[q]['pipeline']}
                )
            )

    def embed_docs(self, embedding_function):
        self.embedding_function = embedding_function

        self.db = DocArrayInMemorySearch.from_documents(documents=self.docs, embedding=embedding_function)

    # cls is for instance "capture_image"
    def extract_input_output(self, cls, file_name):
        tool_info = {}
        basename = os.path.basename(self.tools_dir)
        module = importlib.import_module(f'{basename}.{file_name}')
        tool_class = getattr(module, cls)
        tool_info = {
            'name': tool_class.__name__,
            'description': tool_class.description['description'] + tool_class.description['more details'],
            'input_parameters': tool_class.description['input_parameters'],
            'output_parameters': tool_class.description['output_parameters'],
            'actor': tool_class.description['actor']
        }
        return tool_info


    def search(self, query):
        best_result = None
        try:
            # cosine distance is used to find the closest vector
            # best_match contains the list of the closest vectors (the first element is the closest one)            
            best_match = self.db.similarity_search(query)
            best_result = best_match[0]
            print(best_result)
        except Exception as e:
            exception = str(e)
            return {'api_name': self.__class__.__name__, 'input': query, 'output': best_result, 'exception': exception}
        else:
            return {'api_name': self.__class__.__name__, 'input': query, 'output': best_result, 'exception': None}


class PipelineManagerDB:

    def __init__(self, model, key):
        self.pipeline_store = PipelineStore()
        if model == "GPT":
            embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=key)
            pass
        elif model == "Mistral":
            #embedding_function = MistralAIEmbeddings( model="mistral-embed", api_key=key)
            pass
        
        self.pipeline_store.embed_docs(embedding_function)

    def command_line(self):
        print("ciao")
        while True:
            query = input('Please enter the query (\'exit\' to exit):\n')
            if query == 'exit':
                break
            response = self.pipeline_store.search(query)['output']
            print(response)


if __name__ == '__main__':
    dotenv.load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')

    tools_manager = PipelineManagerDB(openai_key=openai_key)
    tools_manager.command_line()
