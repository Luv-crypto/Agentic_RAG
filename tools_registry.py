# tools_registry.py
# -------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import List, Any
import google.generativeai as genai
from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from domain_routing import choose_domain
from rag_scipdf_core import ingest_documents, smart_query
from dotenv import load_dotenv
import os


load_dotenv()

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_API_KEY:
    raise RuntimeError("Set SERPAPI_API_KEY")


_GEMINI_CHAT = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"), 
)

# 1) classify_document tool (delegates to choose_domain, which may read a PDF)
def _classify(inp: str) -> str:
    return choose_domain(inp)

# 2) ingest_pdf tool
def _ingest(path: str, user_id: int = 0) -> str:
    p = Path(path)
    if not p.is_file():
        return f"❌ File not found: {path}"
    ingest_documents(str(p), user_id=user_id)
    return f"✅ Ingested {p.name}"

# 3) retrieve_rag tool (wrap smart_query)
def _retrieve(q: str, user_id: int = 0, top_k: int = 3) -> str:
    ans, _ = smart_query(q, user_id=user_id, top_k=top_k, return_media=False)
    return ans

# 4) search_web tool (SerpAPI)
_search = SerpAPIWrapper(params={"hl": "en", "gl": "us"})
def _search_web(q: str) -> str:
    dom = choose_domain(q)
    return _search.run(f"{q} ({dom})")[:4000]

# 5) NEW – code_writer tool (Gemini writes complete code)
def _code_writer(task: str) -> str:
    dom = choose_domain(task)
    prompt = (
        f"You are an expert {dom} software engineer.\n"
        "Write clean, well-commented code (Python unless user specifies). "
        "Return only a fenced code block."
    )
    return _GEMINI_CHAT.predict(prompt + "\n\nTASK:\n" + task)

# 6) Factory
def build_tools(user_id: int):
    return [
        Tool(name="classify_document", func=_classify,
             description="Classify raw text or a PDF path into a domain."),
        Tool(name="ingest_pdf", func=lambda p: _ingest(p, user_id),
             description="Ingest a PDF into the proper domain DB."),
        Tool(name="retrieve_rag", func=lambda q: _retrieve(q, user_id),
             description="Run RAG retrieval & answer from ingested PDFs."),
        Tool(name="search_web", func=_search_web,
             description="Web search via SerpAPI (domain-aware)."),
        Tool(name="code_writer", func=_code_writer,
             description="Generate complete domain-aware code."),
        PythonREPLTool(),
    ]
