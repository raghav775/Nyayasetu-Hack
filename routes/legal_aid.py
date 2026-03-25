from fastapi import APIRouter, HTTPException
from models.schemas import LegalAidRequest, LegalAidResponse, SearchSource
from services.rag import search_drafts
from services.llm import call_llm

router = APIRouter()


@router.post("/ask", response_model=LegalAidResponse)
def ask_legal_aid(req: LegalAidRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = search_drafts(req.question, n_results=req.n_results)

    context = ""
    if results:
        context = "\n\n---\n\n".join([
            f"Reference: {r['metadata']['filename']} (Category: {r['metadata']['category']})\n{r['text']}"
            for r in results
        ])

    system_prompt = """You are NyayaSetu, a knowledgeable Indian legal aid assistant.
You provide helpful, accurate legal guidance based on Indian law.
Always clarify that your answers are for informational purposes and not a substitute for professional legal advice.
Be clear, structured, and cite relevant laws or documents when possible."""

    user_message = f"""Answer this legal question:

Question: {req.question}

{"Relevant Legal References:" + chr(10) + context if context else "Answer based on your knowledge of Indian law."}

Provide a clear, helpful answer with relevant legal provisions."""

    answer = call_llm(system_prompt, user_message)

    sources = [
        SearchSource(
            filename=r["metadata"]["filename"],
            category=r["metadata"]["category"],
            score=round(r["score"], 3)
        )
        for r in results
    ]

    return LegalAidResponse(
        question=req.question,
        answer=answer,
        sources=sources
    )