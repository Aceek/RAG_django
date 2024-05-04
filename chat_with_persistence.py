import dotenv

dotenv.load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader


# load the llm
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 1. Load Retriever
loader = PyPDFium2Loader("django-readthedocs-io-en-latest.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace=embeddings.model
)
vector_store = FAISS.from_documents(documents, cached_embedder)
retriever = vector_store.as_retriever()
template = """Awnser the following question based on the provided context. \
If the answer is not in the context, just say that you don't know. \
Use three sentences maximum and keep the answer concise. \
  context: \
  {context}
  
  Question: {question}"""
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# # langserve
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(app, rag_chain)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
