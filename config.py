from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List
from rag_prompts import (
    _gen_image_summary, _gen_table_summary,   # low-level
    ctx_builder_genomic, genomic_meta_from_question,
    build_full_prompt_genomic, genomic_meta_from_text
)
from pathlib import Path

@dataclass
class DomainConfig:
    name: str
    chroma_root: Path
    collection_names: Dict[str, str]          # text/table/image
    embed_models:   Dict[str, str]            # text/table/image
    object_store_dirs: Dict[str, Path]        # image/table
    meta_schema:    Dict[str, Callable[[Any], Any]]
    allowed_meta_keys: List[str]
    ctx_builder:    Callable
    prompt_builders: Dict[str, Callable]      
  




GENOMIC = DomainConfig(
    name="genomic",
    chroma_root=Path("chroma_scipdfs"),
    collection_names={
        "text":  "scientific_chunks",
        "image": "image_summaries",
        "table": "table_summaries",
    },
    embed_models={
        "text":  "models/text-embedding-004",
        "image": "models/text-embedding-004",
        "table": "models/text-embedding-004",
    },
    object_store_dirs={
    "image": Path("object_store/genomic/images"),
    "table": Path("object_store/genomic/tables")},
    meta_schema={
        "title": str,
        "authors": list,
        "abstract": str,
        "Diseases": list,
        "keywords": list,
        "Methodology": str,
    },
    allowed_meta_keys=["title","authors","Diseases","keywords","methodology","abstract"],
    ctx_builder=ctx_builder_genomic,
    prompt_builders={
        "image": _gen_image_summary,
        "table": _gen_table_summary,
        "meta_generation" : genomic_meta_from_text,
        "meta_extraction" : genomic_meta_from_question,
        "query": build_full_prompt_genomic,
    },
)



CYBERSEC = DomainConfig(
    name="cybersecurity",
    chroma_root=Path("chroma_cyberpdfs"),
    collection_names={
        "text":  "scientific_chunks",
        "image": "image_summaries",
        "table": "table_summaries",
    },
    embed_models={
        "text":  "models/text-embedding-004",
        "image": "models/text-embedding-004",
        "table": "models/text-embedding-004",
    },
    object_store_dirs={
    "image": Path("object_store/cybersec/images"),
    "table": Path("object_store/cybersec/tables")},
    meta_schema={
        "title": str,
        "authors": list,
        "abstract": str,
        "keywords": list,
        "Methodology": str,
    },
    allowed_meta_keys=["title","authors","keywords","methodology","abstract"],
    ctx_builder=ctx_builder_genomic,
    prompt_builders={
        "image": _gen_image_summary,
        "table": _gen_table_summary,
        "meta_generation" : genomic_meta_from_text,
        "meta_extraction" : genomic_meta_from_question,
        "query": build_full_prompt_genomic,
    },
)


ALL_DOMAINS = {"GENOMIC": GENOMIC, "CYBERSEC":CYBERSEC}
