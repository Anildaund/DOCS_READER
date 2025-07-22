from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from mistral_llm import MistralLLM
from utils import process_pdf
from rag_chain import create_rag_chain
import model
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

# Optional: allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    model.chunks = process_pdf(await file.read())

    # Generate 3–4 line summary from first 3 chunks
    prompt = "Summarize the following PDF content in 3 to 4 lines:\n\n" + "\n".join(
        [chunk.page_content for chunk in model.chunks[:3]]
    )
    summary = MistralLLM()._call(prompt)

    # Create RAG chain and store it globally
    model.rag_chain = create_rag_chain(model.chunks)

    return {"summary": summary}


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if model.rag_chain is None:
        return {"error": "Please upload a PDF first."}
    
    result = model.rag_chain.invoke({"question": question})
    return {
        "question": question,
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["source_documents"]],
    }
