from langchain_community.chat_models import ChatOpenAI

# Define LLM instance here so it always exists
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, max_tokens=None)
