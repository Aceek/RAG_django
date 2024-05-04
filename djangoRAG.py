import dotenv
import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment variables from .env file
dotenv.load_dotenv()

# Set the environment variable to enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

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

# 2. Load llm from openai
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. create prompt
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. create document chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

