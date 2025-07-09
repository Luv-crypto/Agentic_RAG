
from __future__ import annotations
import os
import google.generativeai as genai
from pathlib import Path
from config import ALL_DOMAINS
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import os


load_dotenv()

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
_LLM = genai.GenerativeModel(MODEL_GEN)

_KEYS = list(ALL_DOMAINS.keys())      #  ["GENOMIC", "CYBERSEC"]

_PROMPT = f"""
You are a **strict one-word domain classifier**.

Allowed answers (exact, case-preserved):
    {_KEYS}

If the text fits **neither** domain, reply exactly:
    none

Do not output anything else.

### Examples
TEXT: ```We analyse gene expression and protein folding.```   -> GENOMIC
TEXT: ```Latest CVE for Cisco ASA firewall.```                -> CYBERSEC
TEXT: ```Random poem about butterflies.```                    -> none

### Task
TEXT: ```{{}}```
Reply with one token: GENOMIC, CYBERSEC, or none.
""".strip()


def _peek_text(pdf_path: Path, n_chars: int = 2000) -> str:
    """Use Docling to extract first 2 000 chars of markdown."""
    d = DocumentConverter().convert(pdf_path).document
    return d.export_to_markdown()[:n_chars]

def choose_domain(text_or_path: str | Path) -> str:
    """Accept raw text *or* a PDF path, return domain string."""
    if isinstance(text_or_path, (str, Path)) and Path(text_or_path).is_file():
        text = _peek_text(Path(text_or_path))
    else:
        text = str(text_or_path)[:2000]
    rsp = _LLM.generate_content(_PROMPT.format(text)).text.strip()
    return rsp if rsp in ALL_DOMAINS else "No domain"


