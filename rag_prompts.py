import json
from typing import Dict, List, Tuple, Any
from utils import _gem_chat, _safe_json ,_fmt_list
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()





# ---------------------------------------------------------------------
# 1)  IMAGE  — richer retrieval-aware summary
# ---------------------------------------------------------------------
def _gen_image_summary(path: str,
                       caption: str,
                       meta: Dict[str, Any],
                       max_words: int = 200) -> str:
    """
    Produce a figure summary that *naturally* weaves in the paper’s context
    (title, diseases, methodology, keywords) so that vector search later
    binds the image back to the right document.
    """
    title      = meta.get("title", "")            # may be empty
    diseases   = _fmt_list(meta.get("Diseases", []))
    keywords   = _fmt_list(meta.get("keywords", []))

    context_bits = [
        f"from the paper titled “{title}”" if title else "",
        f"focused on {diseases}"            if diseases else "",
        f"({keywords})"                     if keywords else ""
    ]
    context = " ".join([b for b in context_bits if b]).strip()

    prompt_header = (
        "You are an expert science writer helping a RAG system.\n"
        "Write a concise, retrieval-friendly figure summary (≤ "
        f"{max_words} words).\n\n"
        "✱ What to include\n"
        "  • The scientific context (disease/topic, method) in one phrase.\n"
        "  • What the image visually shows (axes, flows, key elements).\n"
        "  • Any numerical results or qualitative comparisons visible.\n"
        "  • Mention the provided caption if it clarifies symbols.\n"
        "✱ What to avoid\n"
        "  • Guessing beyond image + caption + metadata.\n"
        "  • Generic filler (e.g., “This is a figure…”).\n\n"
    )
    print(title)
    with open(path, "rb") as f:
        parts = [
            {"mime_type": "image/png", "data": f.read()},
            prompt_header +
            f"Context  : {context or 'N/A'}\n"
            f"Caption   : {caption or 'N/A'}\n"
            f"Metadata  : {json.dumps(meta, ensure_ascii=False)}\n\n"
            "Write the summary:"
        ]
    return _gem_chat(parts) 



# ---------------------------------------------------------------------
# 2)  TABLE — richer retrieval-aware summary
# ---------------------------------------------------------------------
def _gen_table_summary(table_md: str,
                       caption: str,
                       meta: Dict[str, Any],
                       max_words: int = 200) -> str:
    """
    Produce a table summary that embeds the scientific context so the
    RAG system can later retrieve the correct document by content.
    """
    title     = meta.get("title", "")
    diseases  = _fmt_list(meta.get("Diseases", []))
    method    = meta.get("Methodology", "")
    keywords  = _fmt_list(meta.get("keywords", []))

    context_bits = [
        f"from “{title}”"     if title else "",
        f"on {diseases}"      if diseases else "",
        f"using {method}"     if method else "",
        f"({keywords})"       if keywords else ""
    ]
    context = " ".join([b for b in context_bits if b]).strip()

    prompt = f"""
You are an expert science writer helping a RAG system.

Task: Write a succinct (≤ {max_words} words) yet retrieval-friendly table
summary that *naturally* embeds the study context and key metrics.

✱ Must cover
  • Scientific context (topic/disease, method) in a single clause.
  • What variables or metrics the table reports (accuracy, F1, etc.).
  • Any standout values or comparisons (e.g., “Proposed method reaches 97% vs. MobileNetV2’s 91%”).
  • Clarify the caption if it uses abbreviations.

✱ Data provided
  • Table (first 4000 chars of Markdown):
{table_md[:]}

  • Caption  : {caption or 'N/A'}
  • Context  : {context or 'N/A'}
  • Full metadata (JSON for reference, don’t dump): {json.dumps(meta, ensure_ascii=False)}

Write the summary now:
"""
    return _gem_chat(prompt).strip()

# rag_prompts.py


def ctx_builder_genomic(docs: list[str],
                        metas: list[dict],
                        imgs_final: dict,
                        tbls_final: dict) -> list[str]:
    
    ctx: List[str] = []
    for i, (doc_text, meta) in enumerate(zip(docs, metas), start=1):
        section = (
            f"\n### Doc {i} (chunk {meta['chunk_id'][:8]})"
            f"\nTitle   : {meta.get('title','')}"
            f"\nAuthors : {meta.get('authors','')}"
            f"\nAbstract : {meta.get('abstract','')}"
            f"\Keywords : {meta.get('keywords','')}"
            f"\n---\n{doc_text[:1500]}\n"
        )
        ctx.append(section)

    if imgs_final:
        ctx.append("\n## Linked images")
        for im in imgs_final.values():
            # Always show full 36-char UUID in the prompt
            ctx.append(f"* (img:{im['id']}) {im['summary']}")

    if tbls_final:
        ctx.append("\n## Linked tables")
        for tb in tbls_final.values():
            ctx.append(f"* (tbl:{tb['id']}) {tb['summary']}")
    
    
    return ctx



def genomic_meta_from_text(question: str) -> Dict[str, Any]:
    """
    Wraps your existing QUERY_PROMPT block *unchanged* and returns
    a JSON dict with whatever fields Gemini extracts.
    """
    META_PROMPT = textwrap.dedent("""\
        Extract the following fields from the first-page text of a paper.
        Return ONLY valid JSON:
        { "title":string, "authors":[…], "abstract":string, "keywords":[…],
        "Diseases":[…], "Methodology":string }
        Text:
        """)

    raw_json = _gem_chat(META_PROMPT + question)
    return _safe_json(raw_json)



# ───────────────────────────────────────────────────────────────
#  1) QUERY-METADATA   →  returns a Python dict
# ───────────────────────────────────────────────────────────────
def genomic_meta_from_question(question: str) -> Dict[str, Any]:
    """
    Wraps your existing QUERY_PROMPT block *unchanged* and returns
    a JSON dict with whatever fields Gemini extracts.
    """
    QUERY_PROMPT = textwrap.dedent("""\
    Extract any of these fields from the user query (return valid JSON):
    { "Diseases":[…], "title":string, "authors":[…],
      "keywords":[…], "methodology":string }
    Query:
    """).strip()
   
    raw_json = _gem_chat(QUERY_PROMPT + question)
    return _safe_json(raw_json)



# ───────────────────────────────────────────────────────────────
#  2) FULL-ANSWER PROMPT   →  returns a prompt string
# ───────────────────────────────────────────────────────────────
def build_full_prompt_genomic(ctx_chunks: List[str], question: str) -> str:
    """
    ctx_chunks : list of strings (built earlier)
    question   : user’s raw question
    Returns    : the long prompt EXACTLY as you wrote it, with ctx & question inserted
    """
    FULL_PROMPT_HEADER = textwrap.dedent("""
        You are given text chunks (academic paper extracts) plus
        concise summaries of images and tables that might belong to them.

        • Answer strictly using ONLY the provided material. 
        • If the answer is not available in chunks and table simply say "Sorry, The  text does not contain information about your question"
        • Cite chunks as (Doc 1), (Doc 2), etc.
        • If an image/table is essential, output exactly
            <<img:FULL_UUID>>   or   <<tbl:FULL_UUID>>
          on its own line (no other text).

        --- EXAMPLE 1 ---

        CONTEXT:
        ### Doc 1
        Title   : Hepatitis Subtype—Encoding
        ---
        “Each DNA sequence is converted via EIIP coding (A→0.1260, C→0.1340, G→0.0806, T→0.1335) to numeric form.”

        ### Doc 2
        Title   : Hepatitis Subtype—Transforms
        ---
        “After EIIP, a Discrete Sine Transform (DST) is applied, then a level-4 Haar wavelet, and SVD retains the top 5 singular vectors.”

        ## Linked images
        * (img:936a6f5a-b0d3-4526-9b7b-1c917e730a03) Pipeline flowchart showing EIIP→DST→Haar→SVD.

        ## Linked tables
        * (tbl:f062ae4e-c43a-4018-a400-7e9f3674f255) The tables displays the metrics of the models performance accross different combinations.

        QUESTION:
        Explain the signal-processing pipeline for hepatitis subtype classification.

    ANSWER:
    First, raw DNA is mapped via EIIP coding (Doc 1). Next, a Discrete Sine Transform (DST) is applied to those numeric vectors (Doc 2). Then a level 4 Haar wavelet extracts multiresolution coefficients (Doc 2). Finally, SVD is performed and the top 5 singular vectors feed into the classifier (Doc 2).  
    <<img:936a6f5a-b0d3-4526-9b7b-1c917e730a03>>
    In the above template you can see that the results part was excluded in the final answer as the question and the table id was not tagged as it was not specifically asking for that information.
                                                              
    > Follow the template accordigly without including any irrelevant information that might have been provided accidently and donot cite the table or image that is not relevant. 
    • Cite chunks as (Doc 1), (Doc 2)… .
    • If an image/table is essential, output exactly
        <<img:FULL_UUID>>   or   <<tbl:FULL_UUID>>
      on its own line (no other text on that line).
    """).strip()

    prompt = textwrap.dedent(f"""{FULL_PROMPT_HEADER}

    --- MATERIAL ---
    {''.join(ctx_chunks)} 
    --- END MATERIAL ---

    Question: "{question}"
    """).strip()


    return _gem_chat(prompt)

 