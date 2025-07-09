# ----------------------------------------
# rag_scipdf_core.py  – ingestion + retrieval
from __future__ import annotations                      # later you can switch domains
import glob
import re
import uuid
from pathlib import Path
from typing import List, Tuple
from threading import Event
import nest_asyncio
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from IPython.display import display, Markdown, Image
from utils import get_chroma_collections, ensure_dirs,_flatten_meta, _embed, CAP_RE, _find_caption,_candidate_filters, _fetch_media_linked,_zip_ids_meta,_top_media_by_similarity
from config import ALL_DOMAINS
from domain_routing import choose_domain
from logging_config import logger





nest_asyncio.apply()


# ─────────────────── Docling converter ────────────────────────
pipe_opts = PdfPipelineOptions(
    do_table_structure=True,
    generate_page_images=True,
    generate_picture_images=True,
    save_picture_images=True,
    images_scale=2.0
)
pipe_opts.table_structure_options.mode = TableFormerMode.ACCURATE
converter = DocumentConverter(
    format_options={ InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts) }
)



# ═══════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════
def ingest_documents(pattern: str,user_id : int, chunk_size: int = 1500, stop_event: Event | None = None) -> None:
    """
    Ingest all PDFs matching `pattern` into three Chroma collections:
      • scientific_chunks   (text chunks, embeddings & metadata)
      • image_summaries     (figure summaries, embeddings & metadata)
      • table_summaries     (table summaries, embeddings & metadata)

    Also saves:
      - PNG figures under object_store/images/
      - Markdown tables under object_store/tables/
    """
    stop_event = stop_event or Event()   # use dummy flag if caller passed None

    pdfs = glob.glob(pattern, recursive=True)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs matched pattern: {pattern}")

    for pdf in pdfs:
        if stop_event.is_set():          # ← graceful early-exit
            print("▶ Ingestion cancelled by user")
            return

        p = Path(pdf)
        print(f"\n▶ Processing {p.name} …")
        # 1) Use Docling to convert → Markdown + page images + saved PNGs
        ddoc = converter.convert(p).document
        md   = ddoc.export_to_markdown()
        
        # 2) Pick domain & CFG for this document --------------------------
        doc_domain = choose_domain(md[:2000])
        if doc_domain is None:
            print(f" No domain found for {p.name} – skipped.")
            continue
        CFG = ALL_DOMAINS[doc_domain]
        logger.info(f"Started ingesting {p.name} for domain {doc_domain}")


        # one-time per PDF: make sure object-store dirs exist
        ensure_dirs(CFG.object_store_dirs)
        collection_txt, collection_img, collection_tbl = get_chroma_collections(CFG)
        OBJ_DIR_IMG = CFG.object_store_dirs["image"]
        OBJ_DIR_TBL = CFG.object_store_dirs["table"]

        # 3) Extract “global” metadata (title/authors/etc) from first ~1500 chars


        # 2) Extract “global” metadata (title/authors/etc) from first ~1500 chars
        meta_dict = CFG.prompt_builders["meta_generation"](md[:1500])
        meta_dict["path"] = str(p)
        meta_flat = _flatten_meta(meta_dict)


        # 3) Split the full Markdown into ~chunk_size pieces
        text_chunks = [md[i : i + chunk_size] for i in range(0, len(md), chunk_size)]
        chunk_ids   = [str(uuid.uuid4()) for _ in text_chunks]

        # 4) Embed & store each text chunk into `collection_txt`
        for cid, chunk in zip(chunk_ids, text_chunks):
            vec = _embed([chunk], model=CFG.embed_models["text"])[0]
            flat = {
                **meta_flat,
                "chunk_id": cid,
                "chunk_preview": chunk[:400],
                "user_id" :user_id
            }
            collection_txt.add(
                ids=[cid],
                embeddings=[vec],
                documents=[chunk],
                metadatas=[flat]
            )
        
        # 5) Process each figure in ddoc.pictures → save PNG + embed its 200-word summary
        page_numbers = [pic.prov[0].page_no for pic in ddoc.pictures if pic.prov]
        max_pg = max(page_numbers) if page_numbers else 1
        for pic in ddoc.pictures:
            img = pic.get_image(ddoc)
            if img is None:
                continue
            pg = pic.prov[0].page_no if pic.prov else 1
            # Save PNG to object_store/images/
            img_id = str(uuid.uuid4())                       # one UUID for both
            fn     = f"{img_id}_{p.stem}_p{pg}.png"
            fp     = OBJ_DIR_IMG / fn
            img.save(fp, "PNG")
            caption_image = pic.caption_text(ddoc) or "" 
            # Determine which text‐chunk “owns” this page:
            idx = min(int((pg - 1) / max_pg * len(chunk_ids)), len(chunk_ids) - 1)
            parent = chunk_ids[idx]
            # Summarize that figure (send PNG → Gemini)
            summ = CFG.prompt_builders["image"](**{ "path": str(fp),"caption": caption_image,"meta": meta_flat})

            embed_text = f"{caption_image}\n\n{summ}" if caption_image else summ

            collection_img.add(
                ids=[img_id],
                embeddings=[_embed([embed_text],model=CFG.embed_models["image"])[0]],
                documents=[summ],
                metadatas=[{
                    **meta_flat,
                    "id": img_id,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "caption": caption_image,
                    "summary": summ,
                    "user_id" :user_id
                }]
            )
        
        print("Images ingested")
        page_nums_tbl = [t.prov[0].page_no for t in ddoc.tables   if t.prov]
        max_pg_tbl = max(page_nums_tbl) if page_nums_tbl else 1

    

# --- Table summary generation -------------
        for tbl in ddoc.tables:
            tbl_md  = tbl.export_to_markdown(ddoc).strip()
            pos     = md.find(tbl_md)

            # 1) Docling’s own caption if it already starts with “Table …”
            caption = (tbl.caption_text(ddoc) or "").strip()
            if not CAP_RE.match(caption):
                # 2) search ↑ above the grid
                caption = _find_caption(md[:pos].splitlines(), "above") or caption

            if not CAP_RE.match(caption):
                # 3) search ↓ below the grid
                caption = _find_caption(md[pos + len(tbl_md):].splitlines(), "below") or caption

            # ---------- everything below is what you already had ---------------
            # page → owning chunk
            pg   = tbl.prov[0].page_no if tbl.prov else 1
            idx  = min(int((pg - 1) / max_pg_tbl * len(chunk_ids)), len(chunk_ids) - 1)
            parent = chunk_ids[idx]

            tid = str(uuid.uuid4())
            fp  = OBJ_DIR_TBL / f"{tid}.md"
            fp.write_text(tbl_md, encoding="utf-8")


            summ       = summ = CFG.prompt_builders["table"](**{"table_md": tbl_md, "caption" : caption, "meta" : meta_flat})
            embed_text = f"{caption}\n\n{summ}" if caption else summ
            collection_tbl.add(
                ids        =[tid],
                embeddings =[_embed([embed_text],model=CFG.embed_models["table"])[0]],
                documents  =[summ],
                metadatas  =[{
                    **meta_flat,
                    "id": tid,
                    "parent_chunk_id": parent,
                    "path": str(fp),
                    "caption": caption,
                    "summary": summ,
                    "user_id" :user_id
                }]
            )
        logger.info(f"Ingested {len(chunk_ids)} chunks, {len(ddoc.pictures)} images")

# ═══════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def smart_query(
        question: str,
        user_id: int,
        top_k: int = 3,
        return_media: bool = False   # ← new optional kw-arg
    ) -> str | tuple[str, list[tuple[str,str]]]: 
    """
    Perform a “smart” RAG:
     1) Metadata‐aware + semantic search in `scientific_chunks` to get top_k text chunks.
     2) Fetch media linked by chunk_id (images + tables).
     3) Semantic‐nearest search on `image_summaries` + `table_summaries` to add any “closest” media.
     4) Re‐rank all candidate media by cosine similarity of their summary embeddings (keep top1 image & top2 tables).
     5) Build a single Gemini prompt that contains:
         • The top text chunks (with chunk_id, title, authors, chunk preview).
         • A “## Linked images” section listing each figure’s 200-word summary, prefaced with `<<img:FULL_UUID>>`.
         • A “## Linked tables” section listing each table’s 200-word summary, prefaced with `<<tbl:FULL_UUID>>`.
     6) Send to Gemini. If Gemini needs to actually show a figure or table, it writes exactly `<<img:ID8>>` or `<<tbl:ID8>>`
        (8 hex chars) or the full UUID (36 chars). We catch either format, look up path, and render inline.
    """

    query_domain = choose_domain(question)
    if query_domain is None:
        return "❌ No domain found for this query.", [] if return_media else "❌ No domain found for this query."
    CFG = ALL_DOMAINS[query_domain]
    collection_txt, collection_img, collection_tbl = get_chroma_collections(CFG)

    # ── 1) Embed question + attempt metadata filters one by one ───────────
    q_vec = _embed([question],model=CFG.embed_models["text"])[0]
    meta_raw = CFG.prompt_builders["meta_extraction"](question)
    
    hits_txt = None

    # ------------------------------------------------------------------
# Build a user-scoped WHERE clause and query Chroma
# ------------------------------------------------------------------
    for i, flt in enumerate(_candidate_filters(meta_raw), start=1):
        # 1️⃣ Always restrict to the current user

        # --------------------------------------------------------------
#`````` 1) metadata-aware search, always scoped by user_id
#`````` --------------------------------------------------------------
        user_clause = {"user_id": user_id}                # simple equality form
        hits_txt    = None


        for flt in _candidate_filters(meta_raw):
            # ----- build legal WHERE clause ---------------------------
            if flt is None:                           # last pass = “no meta filter”
                where_clause = user_clause            # user only
            else:
                where_clause = {                      # user AND metadata
                    "$and": [user_clause, flt]
                }

            hits_txt = collection_txt.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas"]
            )
            # stop at first non-empty result set
            if hits_txt and hits_txt["ids"] and hits_txt["ids"][0]:
                break

        # --------------------------------------------------------------
        # 2) pure semantic fallback (but still user-scoped)
        # --------------------------------------------------------------
        if not hits_txt or not hits_txt["ids"] or not hits_txt["ids"][0]:
            hits_txt = collection_txt.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where=user_clause,            # <-- user only, no metadata filter
                include=["documents", "metadatas"]
            )
    docs  = hits_txt["documents"][0]
    metas = hits_txt["metadatas"][0]
    chunk_ids = [m["chunk_id"] for m in metas]

    # ── 2) Fetch media directly linked by chunk_id ───────────────────────
    imgs_link, tbls_link = imgs_link, tbls_link = _fetch_media_linked(
        chunk_ids,
        collection_img,          #To be added
        collection_tbl,
        user_id=user_id
        )

    # ── 3) Semantic‐nearest search in media stores ──────────────────────
    imgs_sem_res = collection_img.query(
        [q_vec],
        n_results=top_k,
        where={"user_id": user_id},  
        include=["metadatas"]
    )
    tbls_sem_res = collection_tbl.query(
        [q_vec],
        n_results=top_k,
        where={"user_id": user_id},  
        include=["metadatas"]
    )
    imgs_sem = _zip_ids_meta(imgs_sem_res)
    tbls_sem = _zip_ids_meta(tbls_sem_res)

    # Combine linked + nearest, keyed by full “id”
    imgs_all = {m["id"]: m for m in (imgs_link + imgs_sem)}
    tbls_all = {t["id"]: t for t in (tbls_link + tbls_sem)}


    # ── 4) Re‐rank media by cosine similarity of their summary embeddings ──
    top_img_ids = _top_media_by_similarity(q_vec, imgs_all, CFG.embed_models["image"],1)   # keep best 1 image
    top_tbl_ids = _top_media_by_similarity(q_vec, tbls_all, CFG.embed_models["table"],2)   # keep best 2 tables

    imgs_final = {mid: imgs_all[mid] for mid in top_img_ids if mid in imgs_all}
    tbls_final = {tid: tbls_all[tid] for tid in top_tbl_ids if tid in tbls_all}
    
    ctx_chunks = CFG.ctx_builder(docs, metas, imgs_final, tbls_final)
    answer = CFG.prompt_builders["query"](ctx_chunks, question)
    

  

    # ── 6) Inline render (Jupyter/VS Code) if Gemini emitted any media tokens ───
    #    We match either 8-hex chars OR full 36-char UUID (with hyphens).
    show: List[Tuple[str, str]] = []
    pattern = r"<<(img|tbl):([0-9A-Fa-f]{8}|[0-9A-Fa-f\-]{32,36})>>"
    for kind, token in re.findall(pattern, answer):
        kind = kind.lower()
        # If 8 hex chars, find the first media whose ID startswith token
        if len(token) == 8:
            if kind == "img":
                match = next((m for m in imgs_final.values() if m["id"].startswith(token)), None)
            else:
                match = next((t for t in tbls_final.values() if t["id"].startswith(token)), None)
        else:
            # 32–36 chars → treat as full UUID
            match = (imgs_final.get(token) if kind == "img" else tbls_final.get(token))

        if match:
            path = match["path"]
            if Path(path).exists():
                if show and (kind, path) in show:
                    pass
                else:
                    show.append((kind, path))

    # Display answer + inline media in Jupyter / VS Code if available
    try:
        display(Markdown(answer))
        for kind, p in show:
            if kind == "img":
                display(Image(filename=p))
            else:
                md_text = Path(p).read_text(encoding="utf-8")
                display(Markdown(md_text))
    except ImportError: 
        # If not in notebook, just print text  paths
        print(answer)
        for kind, p in show:
            print(f"[{kind.upper()}]: {p}")

    # Return the tuple: (answer_text, list_of_(kind,path))
    if return_media:
        return answer, show
    else:
        answer


        