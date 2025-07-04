import logging

logging.basicConfig(
    filename="rag_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RAG")
