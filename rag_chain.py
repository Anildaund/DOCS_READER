from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from mistral_llm import MistralLLM

def create_rag_chain(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embedder)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"
    )

    llm = MistralLLM()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, output_key="answer"
    )
    return chain
