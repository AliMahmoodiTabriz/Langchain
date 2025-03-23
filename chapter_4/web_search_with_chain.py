from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=search_wrapper, output_format="list")

def web_search(question: str) -> list:
    return search.invoke(question)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
web_result = RunnableLambda(web_search)
message = """
Answer this question using the provided context only.

{question}

Context: {context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context": web_result, "question": RunnablePassthrough()} | prompt | model

tools = [search]
load_dotenv()

if __name__ == "__main__":
    search_result = chain.invoke("what is the weather in istanbul today?")
    print(search_result.content)
