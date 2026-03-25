from fastapi import APIRouter, HTTPException
from models.schemas import CaseSearchRequest, CaseSearchResponse, SearchSource
from services.rag import search_drafts
from services.scraper import scrape_indian_kanoon
from services.llm import call_llm

router = APIRouter()


@router.post("/search", response_model=CaseSearchResponse)
def search_cases(req: CaseSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    rag_results = search_drafts(req.query, n_results=req.n_results)
    live_results = scrape_indian_kanoon(req.query, max_results=3)

    if not rag_results and not live_results:
        raise HTTPException(status_code=404, detail="No relevant cases found")

    rag_context = "\n\n---\n\n".join([
        f"Document: {r['metadata']['filename']} (Category: {r['metadata']['category']})\n{r['text']}"
        for r in rag_results
    ])

    live_context = "\n\n".join([
        f"Case: {r['title']}\nLink: {r['link']}\nSummary: {r['snippet']}"
        for r in live_results
    ])

    system_prompt = """You are NyayaSetu, an expert Indian legal assistant.
You help advocates and legal interns find relevant case law and legal precedents.
Always cite the source document names and case links in your answer.
Be precise, professional, and structured."""

    user_message = f"""Find relevant legal cases and precedents for this query:

Query: {req.query}

Legal Documents from Database:
{rag_context}

Live Cases from Indian Kanoon:
{live_context}

Provide a structured answer with relevant legal principles and cite all sources."""

    answer = call_llm(system_prompt, user_message)

    sources = [
        SearchSource(
            filename=r["metadata"]["filename"],
            category=r["metadata"]["category"],
            score=round(r["score"], 3)
        )
        for r in rag_results
    ]

    return CaseSearchResponse(
        query=req.query,
        answer=answer,
        sources=sources
    )