from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=search_wrapper, output_format="list")
checkpointer = MemorySaver()
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tools = [search]

react_agent = create_react_agent(model, tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

if __name__ == "__main__":
    # input_data = {
    #     "messages": [("human", "500 dolar kac lira eder? bugunun tarihinde?"),
    #                  ("system","Please use DuckDuckGo to search for 'human input' and return the search results.")]
    # }
    while (True):
        user_input = input(">")
        if user_input == "exit":
            break
        for chunk in react_agent.stream(
            {"messages": [HumanMessage(content=user_input)
                ,("system","DuckDuckGo kullanarak guncel cevablar ver gerdi zamanlar kulanici sorularina")]}
        , config):
            print(chunk)
