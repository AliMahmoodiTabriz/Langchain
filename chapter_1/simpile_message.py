from http.client import responses

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)
messages = [
    SystemMessage("Translate the user sentence to Turkish."),
    HumanMessage("I love programming."),
]
if __name__ == "__main__":
    response = model.invoke(messages)
    print(response.content)
