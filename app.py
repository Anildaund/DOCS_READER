import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ── LangChain components ────────────────────────────────────────────────────
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.llms import LLM

# ── Mistral SDK ─────────────────────────────────────────────────────────────
from mistralai import Mistral
from pydantic import PrivateAttr

# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit setup
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="🤖 Conversational PDF Chatbot", layout="wide")
st.title("📄 Conversational PDF ChatBot (Mistral + RAG)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Custom Mistral Wrapper (LangChain-compatible)
# ─────────────────────────────────────────────────────────────────────────────
class MistralLLM(LLM):
    _client: Mistral = PrivateAttr()
    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()

    def __init__(
        self,
        model: str = "mistral-small",
        temperature: float = 0.1,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))
        self._model = model
        self._temperature = temperature

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.complete(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            stop=stop,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "mistral"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self._model, "temperature": self._temperature}


# ─────────────────────────────────────────────────────────────────────────────
#  PDF Upload & RAG Setup
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload your PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    st.info("📄 Extracting text...")
    docs = PyMuPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    #st.success(f"✅ Extracted {len(chunks)} text chunks.")

    # ── Generate summary ───────────────────────────────────────────────
    st.info("📝 Generating 3–4 line summary...")
    summary_prompt = (
        "Summarize the following PDF content in 3 to 4 lines:\n\n"
        + "\n".join([chunk.page_content for chunk in chunks[:3]])
    )
    summary = MistralLLM().invoke(summary_prompt)
    st.subheader("📝 PDF Summary")
    st.markdown(summary)

    # ── Vector DB ──────────────────────────────────────────────────────
    #st.info("🔍 Creating vector index...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embedder)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # ── Conversational RAG Chain ──────────────────────────────────────
    st.info("🔁 Building conversational agent...")
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=MistralLLM(),
        retriever=retriever,
        memory=st.session_state.chat_memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False
    )

    st.success("🚀 Ready! Start chatting about your PDF.")

    # ── User Chat ─────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            result = conv_chain.invoke({"question": user_input})
            answer = result["answer"]
            sources = result["source_documents"]

        # Save to session history
        st.session_state.chat_history.append((user_input, answer))

        # Display current answer
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(answer)

        # Show sources
        with st.expander("📄 Source Passages"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}**")
                st.write(doc.page_content)

    # Show chat history
    # if st.session_state.chat_history:
    #     st.subheader("💬 Full Chat History")
    #     for q, a in st.session_state.chat_history:
    #         st.markdown(f"**🧑 You:** {q}")
    #         st.markdown(f"**🤖 Bot:** {a}")
