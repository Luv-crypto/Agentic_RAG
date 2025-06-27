from numpy import dot
from numpy.linalg import norm
import json
import time
from typing import Dict, List, Tuple 
import re
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")
_gem = genai.GenerativeModel(MODEL_GEN)



# ─────────────────── helper functions ─────────────────────────

def _gem_chat(prompt: str, retry: int = 3) -> str:
    """
    Simple wrapper around Gemini chat. Retries up to `retry` times on failure.
    """
    for i in range(retry):
        try:
            return _gem.generate_content(prompt).text.strip()
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(1 + i)
    # should never reach here
    return ""



# utils.py
def _embed(texts: List[str], model: str | None = None) -> List[List[float]]:
    """
    texts : list[str]
    model : overrides the default embedding model when provided
    """
    model_name = model    # ← falls back to global default
    return genai.embed_content(
        model=model_name,
        content=texts,
        task_type="retrieval_document"
    )["embedding"]



def _safe_json(raw: str) -> Dict:
    """
    Strip any ```json fences and attempt to json.loads. On failure, return {}.
    """
    raw = re.sub(r"^```json|```$", "", raw, flags=re.I).strip()
    try:
        return json.loads(raw)
    except:
        return {}

def _flatten_meta(meta: Dict) -> Dict:
    """
    Convert a metadata dict so that every value is a scalar (string/int/float/bool).
    - Lists become semicolon-joined JSON dumps of each element (if not a simple string).
    - Dicts become JSON dumps.
    """
    flat: Dict[str, object] = {}
    for k, v in meta.items():
        if isinstance(v, list):
            parts: List[str] = []
            for x in v:
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, dict):
                    # turn nested dict→ JSON string
                    parts.append(json.dumps(x, ensure_ascii=False))
                else:
                    parts.append(str(x))
            flat[k] = "; ".join(parts)
        elif isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------
# Helper: craft a single human-readable sentence from list-like metadata
# ---------------------------------------------------------------------
def _fmt_list(xs: List[str]) -> str:
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f", and {xs[-1]}"


CAP_RE = re.compile(r"^(table|tab\.)\s+[ivxlcdm\d]+\b", re.I)   # Table II / Tab. 3 …

def _find_caption(lines, direction="below", max_scan=8):
    """Scan up to `max_scan` non-blank lines above/below the grid."""
    seq = lines if direction == "below" else reversed(lines)
    seen = 0
    for ln in seq:
        txt = ln.strip().strip("| ").strip()      # strip leading '|' if inside grid
        if not txt:
            continue
        if CAP_RE.match(txt):
            return txt
        seen += 1
        if seen >= max_scan:
            break
    return ""


def _zip_ids_meta(res) -> List[Dict]:
    """
    Convert a Chroma `get()` or `query()` result into a list of dicts,
    each containing the metadata plus an “id” field. If no hits, return [].
    """
    if not res or not res.get("ids"):
        return []

    # Chroma’s “query” returns nested lists; “get” returns flat lists.
    ids_raw   = res["ids"][0]    if isinstance(res["ids"][0], list) else res["ids"]
    metas_raw = res["metadatas"][0] if isinstance(res["metadatas"][0], list) else res["metadatas"]

    out: List[Dict] = []
    for _id, meta in zip(ids_raw, metas_raw):
        if meta is None:
            continue
        d = dict(meta)
        d["id"] = _id
        out.append(d)
    return out

def _fetch_media_linked(chunk_ids: List[str], collection_img, collection_tbl,user_id:int) -> Tuple[List[Dict], List[Dict]]:
    """
    Given a list of text‐chunk IDs, fetch all images/tables in Chroma whose
    `parent_chunk_id` is in that list. Returns two lists of dicts (imgs, tables).
    """
    if not chunk_ids:
        return [], []

    user_clause = {"user_id": user_id}     # simple equality form
    where_clause = {
        "$and": [
            user_clause,
            {"parent_chunk_id": {"$in": chunk_ids}}
        ]
    }
    imgs = _zip_ids_meta(collection_img.get(where=where_clause, include=["metadatas"]))
    tbls = _zip_ids_meta(collection_tbl.get(where=where_clause, include=["metadatas"]))
    return imgs, tbls

def _candidate_filters(meta: Dict) -> List[Dict]:
    """
    Given a parsed metadata‐dict from the LLM (e.g. {"Diseases":["Hepatitis"], "title":None, ...}),
    produce a list of “single‐field” where‐clauses to try in Chroma. E.g.:
      • if v is a list → use {"field": {"$in": v}}
      • if v is a non‐empty scalar → use {"field": {"$eq": v}}
    Finally always append a None (meaning “no metadata filter”).
    """
    out: List[Dict] = []
    for k, v in meta.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            out.append({k: {"$in": v}})
        else:
            out.append({k: {"$eq": v}})
    out.append(None)  # fallback: no filter
    return out


def _top_media_by_similarity(question_vec: List[float],
                             media: Dict[str, Dict],
                             top_n: int = 2) -> List[str]:
    """
    Given a dict of media (key=media_id, value=metadata including “summary”),
    compute cosine similarity between question_vec and each media["summary"] embedding,
    return the top_n media_ids, sorted by descending similarity.
    """
    if not media:
        return []

    ids_list    = list(media.keys())
    summaries   = [media[mid]["summary"] for mid in ids_list]
    sum_vecs    = _embed(summaries)   # embeddings for all summaries
    q = question_vec
    sims = [ dot(q, v) / (norm(q)*norm(v) + 1e-9) for v in sum_vecs ]

    id_sims = list(zip(ids_list, sims))
    id_sims.sort(key=lambda t: t[1], reverse=True)
    return [mid for mid, _ in id_sims[:top_n]]


def ensure_dirs(dirs: Dict[str, Path]) -> None:
    for p in dirs.values(): p.mkdir(parents=True, exist_ok=True)

def get_chroma_collections(cfg) -> tuple:
    import chromadb
    client = chromadb.PersistentClient(path=str(cfg.chroma_root))
    return (
        client.get_or_create_collection(cfg.collection_names["text"]),
        client.get_or_create_collection(cfg.collection_names["image"]),
        client.get_or_create_collection(cfg.collection_names["table"]),
    )


