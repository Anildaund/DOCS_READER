
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from mistral_llm import MistralLLM
from utils import process_pdf
from rag_chain import create_rag_chain
import model
from dotenv import load_dotenv
load_dotenv()
from fastapi.responses import FileResponse

app = FastAPI()



# Optional: allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_html():
    return FileResponse("index.html", media_type="text/html")

SUGGESTED_QUESTIONS = [
    "What is the main topic of this document?",
    "Can you summarize the key points?",
    "What are the important definitions or concepts?",
    "List out any steps, procedures, or instructions mentioned.",
    "What are the conclusions or final findings?",
    "Are there any statistics or data insights?",
    "What questions can I expect from this PDF in a test?",
    "Can you generate 5 MCQs from this document?",
    "What is the tone or writing style of the document?",
    "List all the chapters or sections from the PDF.",
]

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

@app.get("/suggested_questions/")
async def get_suggested_questions():
    return {"suggested_questions": SUGGESTED_QUESTIONS}