from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pickle
import os


def get_pdf_text(pdf_files):
    
    text = ""

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    # if os.path.exists(f"./data/{file_name}.pkl"):
    #     with open(f"./data/{file_name}.pkl", "rb") as f:
    #         vectorstore = pickle.load(f)
    # else:
    #     embeddings = OpenAIEmbeddings()
    #     vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    #     with open(f"./data/{file_name}.pkl", "wb") as f:
    #         pickle.dump(vectorstore, f)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    
    # OpenAI Model

    llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )

    return conversation_chain