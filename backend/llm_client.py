from langchain_openai import ChatOpenAI

# Define LLM instance here so it always exists
llm = ChatOpenAI(model_name="gpt-5", temperature=0.2, max_tokens=None)
