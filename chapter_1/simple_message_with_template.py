from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class SimpleMessageWithTemplate:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Translate the user sentence to {language}."), ("user", "{text}")
        ])
        self.parser = StrOutputParser()
        self.chain = prompt_template | self.model | self.parser

    def proses(self, template: dict[str, str]) -> str:
        return self.chain.invoke(template)

def main():
    sample = SimpleMessageWithTemplate()
    print(sample.proses(template={
        "language": "Turkish", "text": "HI! my name is Ali Mahmoodi so im a software engineer"
    }))

if __name__ == "__main__":
    main()

