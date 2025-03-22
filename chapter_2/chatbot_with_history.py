from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


class ChatBotWithHistory:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)
        self.stor = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "you are chatbot assistant please help users with ability"),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.model | self.parser
        self.config = {"configurable": {"session_id": "ali"}}
        self.with_chat_history = RunnableWithMessageHistory(self.chain, self.get_session_history)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.stor:
            self.stor[session_id] = InMemoryChatMessageHistory()
        return self.stor[session_id]




if __name__ == "__main__":
    # Hi! may name is ali and im working on langchain and LLM so can you help me about that?
    bot = ChatBotWithHistory()
    while(True):
        user_input = input(">")
        if user_input=="exit":
            break

        # response = bot.with_chat_history.invoke([
        #     HumanMessage(content=user_input)
        # ], bot.config)
        #
        # print(response)

        for r in  bot.with_chat_history.stream([
            HumanMessage(content=user_input)
        ], bot.config):
            print(r)

