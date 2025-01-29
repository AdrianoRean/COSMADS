def getModel(model, key):
    llm = None
    if model == "Openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o",
                            api_key=key,
                            temperature=0.0)
    elif model == "Mistral":
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(model="mistral-large-latest",
                            api_key=key,
                            temperature=0.0)
    elif model == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3-5-sonnet-latest",
                            api_key=str(key),
                            temperature=0.0)
    elif model == "Deepseek":
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(model="deepseek-chat",
                            api_key=str(key),
                            temperature=0.0)
    return llm