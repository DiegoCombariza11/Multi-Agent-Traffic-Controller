from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from reporter.core.config import get_settings
from reporter.services.data_loader import load_data
from reporter.core.indexes import Indexes
from reporter.core.cache import AnswerCache
from reporter.services.llm_client import LLMClient
from reporter.services.analyzer import TrafficAnalyzer

settings = get_settings()
DATA = load_data(settings.data_path)
INDEXES = Indexes(DATA)
CACHE = AnswerCache(settings.cache_size)
LLM = LLMClient(settings.ollama_model)
ANALYZER = TrafficAnalyzer(INDEXES, DATA)

app = FastAPI(title="Traffic Reporter API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(question: str):
    cached = CACHE.get(question)
    if cached is not None:
        return {"question": question, "answer": cached, "cached": True}
    answer = ANALYZER.analyze(question, LLM)
    CACHE.set(question, answer)
    return {"question": question, "answer": answer, "cached": False}

@app.get("/cache-stats")
def cache_stats():
    return CACHE.stats()

@app.get("/health")
def health():
    return {"status": "ok"}
