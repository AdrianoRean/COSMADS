#models = ["gpt-4o", "mistral-large-latest", "claude-3-5-sonnet-latest", "deepseek-chat"]

import os
import dotenv


def getModel(enterprise, model):
    dotenv.load_dotenv()
    key = os.getenv(f"{enterprise.upper()}_API_KEY")
    llm = None
    if enterprise == "Openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model,
                            api_key=key,
                            temperature=0.0)
    elif enterprise == "Mistral":
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(model=model,
                            api_key=key,
                            temperature=0.0)
    elif enterprise == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model,
                            api_key=str(key),
                            temperature=0.0)
    elif enterprise == "Deepseek":
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(model=model,
                            api_key=str(key),
                            temperature=0.0)
    return llm