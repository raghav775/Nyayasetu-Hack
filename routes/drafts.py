from fastapi import APIRouter, HTTPException
from models.schemas import DraftRequest, DraftResponse, SearchSource
from services.rag import search_drafts
from services.llm import call_llm

router = APIRouter()


@router.post("/generate", response_model=DraftResponse)
def generate_draft(req: DraftRequest):
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    query = req.description
    if req.category:
        query = f"{req.category}: {req.description}"

    results = search_drafts(query, n_results=req.n_results)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant templates found")

    context = "\n\n---\n\n".join([
        f"Template: {r['metadata']['filename']} (Category: {r['metadata']['category']})\n{r['text']}"
        for r in results
    ])

    system_prompt = """You are NyayaSetu, an expert Indian legal document drafter.
You generate accurate, professional legal documents based on established Indian legal templates.
Follow proper legal formatting, use correct legal terminology, and ensure the document is complete.
Do not leave any critical fields blank — use placeholder text like [PARTY NAME], [DATE], [AMOUNT] where specific details are needed."""

    user_message = f"""Draft a legal document based on this request:

Request: {req.description}
{f"Category: {req.category}" if req.category else ""}

Reference Templates:
{context}

Generate a complete, properly formatted legal document."""

    draft = call_llm(system_prompt, user_message)

    sources = [
        SearchSource(
            filename=r["metadata"]["filename"],
            category=r["metadata"]["category"],
            score=round(r["score"], 3)
        )
        for r in results
    ]

    return DraftResponse(
        description=req.description,
        draft=draft,
        sources=sources
    )