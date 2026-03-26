from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from models.database import get_db, User, QueryLog
from models.schemas import LegalAidRequest, LegalAidResponse, SearchSource
from services.rag import search_drafts
from services.llm import call_llm
from utils.auth import get_current_user
from utils.encryption import encrypt
from services.scraper import scrape_indian_kanoon

router = APIRouter()


@router.post("/ask", response_model=LegalAidResponse)
def ask_legal_aid(
    req: LegalAidRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = search_drafts(req.question, n_results=req.n_results)
    kanoon_results = scrape_indian_kanoon(req.question)
    print("KANOON RESULTS:", kanoon_results)

    context_parts = []

    # 🔹 Local RAG results
    for r in results:
        context_parts.append(
            f"Reference: {r['metadata']['filename']} | Category: {r['metadata']['category']}\n{r['text']}"
        )

    # 🔹 Indian Kanoon results (🔥 THIS WAS MISSING)
    for k in kanoon_results:
        context_parts.append(
            f"Case Title: {k['title']}\nSnippet: {k['snippet']}\nLink: {k['link']}"
        )

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""
    system_prompt = """You are NyayaSetu, a knowledgeable Indian legal aid assistant.
You provide clear, accurate, and helpful legal guidance based on Indian law.

Always structure your response EXACTLY like this:
DIRECT ANSWER:
[Clear answer to the question]

LEGAL BASIS:
[Relevant sections, acts, and legal provisions]

BINDING PRECEDENTS:
[Relevant case citations if applicable, or "No direct precedent required"]

ACTIONABLE INSIGHT:
[Practical next steps or warnings]

DISCLAIMER:
This is for informational purposes only. Please consult a qualified advocate for legal advice.

Be precise, empathetic, and use clear language."""

    user_message = f"""Legal Question: {req.question}

{"Relevant Legal References:" + chr(10) + context if context else "Answer based on your knowledge of Indian law."}"""

    # 🔥 SAFE LLM CALL
    try:
        answer = call_llm(system_prompt, user_message)
    except Exception as e:
        print("LLM ERROR:", e)  # 👈 shows error in terminal
        answer = "⚠️ AI service temporarily unavailable. Please try again."

    try:
        log = QueryLog(
            user_id=current_user.id,
            query_type="legal_aid",
            encrypted_query=encrypt(req.question),  
        )
        db.add(log)
        db.commit()
    except Exception:
        pass

    sources = [
        SearchSource(
            filename=r["metadata"]["filename"],
            category=r["metadata"]["category"],
            score=round(r["score"], 3),
        )
        for r in results
    ]

    return LegalAidResponse(
        question=req.question,
        answer=answer,
        sources=sources,
    )
