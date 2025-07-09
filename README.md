
# Agentic\_RAG

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#requirements)
[📄 Changelog](CHANGELOG.md) • [🤝 Contributing](#contributing)

---

## 🚀 Overview

Agentic\_RAG transforms your **static RAG pipeline** into a **dynamic AI-powered agent** that:

* **Intelligently routes** user queries and PDF ingestions
  across **multiple domains** (e.g. Genomic analysis, Cyber-security).
* Leverages **Google Gemini** for:

  * **High-quality embeddings** (`gemini-embedding-004`)
  * **Conversational and prompt-based summarization** (`gemini-1.5-flash-latest`)
* Stores vectors in **ChromaDB** per domain for **super-fast retrieval**.
* Provides a **web UI** (Flask + simple chat frontend) for uploads & Q\&A.
* Offers a **LangChain agent** that decides “genomic vs. cybersec” at runtime.

![Agentic\_RAG Architecture](docs/architecture_diagram.png)

---

## ⭐ Key Features

1. ### Multi-Domain Ingestion

   * **Genomic papers**: Extract sequences, diseases, methodologies.
   * **Cyber-security reports**: Extract threats, CSIRT processes, metrics.
   * Auto-classify uploads and ingest into the appropriate Chroma collection.

2. ### Domain-Specific Prompting

   * **Image summaries** that “weave in” contextual metadata.
   * **Table summaries** that emphasize key metrics & comparisons.
   * **Full-document summarization** with few-shot examples tailored
     to each domain’s style.

3. ### Agentic Query Routing

   * A **LangChain ZERO-SHOT agent** with two tools:

     * `genomic_search`
     * `cybersec_search`
   * Routes questions to the proper vector DB + prompt builder
     for **120% relevant answers**.

4. ### Seamless UI & CLI

   * **Flask webapp** (`app.py` + `templates/` + `static/`) for:

     * PDF uploads
     * Interactive chat with inline figure/table render
   * **CLI agent** (`cyber_agent.py`) for quick terminal queries.

5. ### Configurable & Extensible

   * **`config.py`** drives:

     * Chroma collection names
     * Embedding models per modality
     * Prompt builders, metadata schemas
     * Object-store directories
   * **Add new domains** by mirroring the `GENOMIC` & `CYBERSEC` configs.

---

## ⚙️ Getting Started

### Prerequisites

* **Python 3.10+**
* **Google Gemini API key**
  Obtain an API key and set it in your environment (see below).
* **ChromaDB** (no external setup: file-backed)

### Installation

1. **Clone** the repo and switch to the `main` branch:

   ```bash
   git clone https://github.com/Luv-crypto/Agentic_RAG.git
   cd Agentic_RAG
   git checkout main
   ```

2. **Create & activate** a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## 💻 Usage


### 1. Run the Web App

```bash
python app.py
```

Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)
Upload PDFs, then chat!

---

## 🛠️ Configuration

All domain settings live in [`config.py`](config.py):

```python
GENOMIC  = DomainConfig(
  name="genomic",
  chroma_root=Path("chroma_scipdfs"),
  collection_names={…},
  embed_models={…},
  object_store_dirs={…},
  meta_schema={…},
  prompt_builders={
    "image": _gen_image_summary,
    "table": _gen_table_summary,
    …
  }
)

CYBERSEC = DomainConfig(
  name="cybersec",
  chroma_root=Path("chroma_cyberpdfs"),
  …
)
```

To **add a new domain**:

1. Define prompt functions in `rag_prompts.py`:

   * `_gen_image_summary_<domain>`
   * `_gen_table_summary_<domain>`
   * `ctx_builder_<domain>`
   * `meta_from_text_<domain>`
   * `meta_from_question_<domain>`
   * `build_full_prompt_<domain>`

2. Mirror the `DomainConfig` block in `config.py` and append to `ALL_DOMAINS`.

---

## 📚 Project Structure

```
.
├── app.py               # Flask app + chat UI
├── config.py            # DomainConfig definitions
├── rag_prompts.py       # All prompt-builder functions
├── rag_scipdf_core.py   # Ingestion + retrieval pipeline
├── utils.py             # Gemini + Chroma helper functions
├── cyber_agent.py       # LangChain agent entrypoint
├── requirements.txt     # ✨ pinned dependencies
├── templates/           # Flask chat/upload/login views
├── static/              # CSS + JS assets
└── LICENSE              # MIT
```

---

## 🤝 Contributing

1. Fork the repo and create your branch:
   `git checkout -b feature/my-new-domain`
2. Commit your changes & push:

   ```bash
   git add .
   git commit -m "Add <domain> support"
   git push origin feature/my-new-domain
   ```
3. Open a Pull Request, describe your domain and prompt logic.

---

## 📝 License

This project is released under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

> ⭐️ If you find Agentic\_RAG useful, please give it a ⭐️ on GitHub!
