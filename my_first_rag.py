from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI, HTTPException, Request, Depends
from langchain.memory import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import re
from typing import Callable, Union
from pathlib import Path
from langchain_core.runnables.history import RunnableWithMessageHistory
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langserve import add_routes
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever






def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))

def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history

# Define the base directory for storing chat histories
base_dir = Path("./chat_histories")
session_factory = create_session_factory(base_dir)


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

# 2. load LLM ensure that ollama server is running and the model is available
llm = Ollama(model="llama2", temperature=0)


# 3. create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant, answer the following question based on the provided context."),
        ("system", "You can keep a trace of previous questions and answers in history."),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="context"),

        ("human", "{input}"),
    ]
)

# 4. create document chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)



# 5. create chain history

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input"}},
    )


chain_history = RunnableWithMessageHistory(
    retrieval_chain,
    create_session_factory("chat_histories"),
    input_messages_key="input",
    history_messages_key="history",
).with_types(input_type=InputChat)


# 6. create app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# 7. add routes
add_routes(
    app,
    chain_history,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

