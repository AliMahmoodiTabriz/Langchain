from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class SimpleMessageWithOutputParser:
    def __init__(self, human_message:str):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)
        self.messages = [
            SystemMessage("Translate the user sentence to Turkish."),
            HumanMessage(human_message),
        ]
        self.parser = StrOutputParser()
        self.chain = self.model | self.parser

    def proses_without_chain(self) -> str:
        response = self.model.invoke(self.messages)
        return self.parser.invoke(response)

    def proses_with_chain(self) -> str:
        return self.chain.invoke(self.messages)

if __name__ == "__main__":
    sample = SimpleMessageWithOutputParser("Hi! Im expert software engineer")
    print("proses without chain:", sample.proses_without_chain())
    print("proses with chain:", sample.proses_with_chain())
