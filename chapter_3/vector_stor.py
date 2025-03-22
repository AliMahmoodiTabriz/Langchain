from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_stor = Chroma.from_documents(
    documents=documents,
    embedding=embedding
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
retriever = RunnableLambda(vector_stor.similarity_search).bind(k=1)
message = """
Answer this question using the provided context only.

{question}

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model
if __name__ == "__main__":
    # embeddog = embedding.embed_query("dog")
    # embedcat = embedding.embed_query("shark")
    # print(vector_stor.similarity_search_by_vector(embed)[0])
    # print(retriever.batch(["dog", "shark"]))
    response =chain.invoke("tell me about cat.")
    print(response.content)
