
from __future__ import annotations
import os
import functools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
import google.generativeai as genai
from tools_registry import build_tools
from dotenv import load_dotenv
import os
from logging_config import logger
from langchain.callbacks.base import BaseCallbackHandler

class StepLogger(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        logger.info(f"Thought/Action: {action.log}")   # prints Thought + Action

    def on_chain_end(self, outputs, **kwargs):
        # final answer
        logger.info(f"Final: {outputs}")

load_dotenv()

# ─────────────────── API keys & model names ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")      # ← Set your own Gemini API key here
MODEL_GEN      = "models/gemini-1.5-flash-latest"
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# ---- 1) Build tool set for the single-user demo -----------------------
_llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"), 
    callbacks=[StepLogger()],
)

SYSTEM_PREFIX = """
You are an agent for a Retrieval-Augmented-Generation system with the following tools:

• classify_document   – classify a PDF or raw text into a domain key
• ingest_pdf          – ingest a PDF into the vector DB
• retrieve_rag        – ⟨ALWAYS CALL THIS FIRST⟩ to answer questions from ingested PDFs
• search_web          – use ONLY if retrieve_rag returns NO_RESULTS
• code_writer, python_repl – programming tools (use if question explicitly needs code)

Protocol:
1. Call retrieve_rag first.
2. If it returns 'NO_RESULTS', then call search_web.
3. If retrieve_rag returns >1 candidate papers, ask the user to choose.
4. When you have the final answer, respond in markdown.

Begin!
""".strip()
# ────────────────────────────────────────────────────────────────
# 2)  Cached agent factory per user_id
# ────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def get_agent(user_id: int):
    """
    Return a ReAct agent whose tool wrappers carry this user_id.
    Cached (LRU) so repeated calls for the same user reuse the object.
    """
    tools = build_tools(user_id)
    ag = initialize_agent(
        tools,
        _llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=6,
        prefix = SYSTEM_PREFIX
    )
    return ag

# ────────────────────────────────────────────────────────────────
# 3)  Optional CLI demo  →  python agentic_rag_agent.py
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔧 Agentic RAG CLI (Gemini).  Type quit to exit.")
    agent0 = get_agent(0)
    while True:
        try:
            q = input("\n🟢 You: ")
            if q.lower() in {"quit", "exit"}:
                break
            print("🔵 Agent:")
            resp = agent0.run(q)
            print(f"\n🟢 Final: {resp}")
        except KeyboardInterrupt:
            break