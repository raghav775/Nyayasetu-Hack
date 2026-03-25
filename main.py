from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes import cases, drafts, legal_aid

load_dotenv()

app = FastAPI(
    title="NyayaSetu API",
    description="AI-powered legal assistant for Indian advocates and interns",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router, prefix="/cases", tags=["Case Finder"])
app.include_router(drafts.router, prefix="/drafts", tags=["Draft Assistant"])
app.include_router(legal_aid.router, prefix="/legal-aid", tags=["Legal Aid"])

@app.get("/")
def root():
    return {"message": "NyayaSetu API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}