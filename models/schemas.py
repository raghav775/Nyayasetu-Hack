from pydantic import BaseModel
from typing import List, Optional


class SearchSource(BaseModel):
    filename: str
    category: str
    score: float


class CaseSearchRequest(BaseModel):
    query: str
    n_results: int = 5


class CaseSearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchSource]


class DraftRequest(BaseModel):
    description: str
    category: Optional[str] = None
    n_results: int = 3


class DraftResponse(BaseModel):
    description: str
    draft: str
    sources: List[SearchSource]


class LegalAidRequest(BaseModel):
    question: str
    n_results: int = 3


class LegalAidResponse(BaseModel):
    question: str
    answer: str
    sources: List[SearchSource]