# multirag — Multi-LLM RAG System with Fallback Chain

> A production-hardened RAG system that keeps working even when your primary LLM goes down.

Most RAG demos assume the LLM is always available. This one doesn't.

`multirag` is a legal document Q&A system built around **resilience** — a 3-tier model fallback chain, hybrid retrieval combining vector search and BM25 keyword search, multi-query expansion for better recall, OCR for scanned documents, and multilingual support for Telugu, Hindi, and English.

---

## Architecture

```
PDF Upload
    │
    ▼
OCR (Tesseract) ──── scanned pages
    │
    ▼
Text Chunking + Preprocessing
    │
    ▼
Dual Indexing ──── Vector Embeddings (FAISS/HuggingFace)
                └── BM25 Keyword Index
    │
    ▼
Multi-Query Expansion ──── 3 query variants per question
    │
    ▼
Hybrid Retrieval + Reranking
    │
    ▼
LLM Fallback Chain
    ├── Groq (primary — fastest)
    ├── OpenRouter (secondary)
    └── HuggingFace (fallback — always available)
    │
    ▼
Answer + Source Pages → Streamlit UI + Chat History
```

---

## Key features

**Resilient LLM orchestration**  
Groq fails → automatically retries with OpenRouter → then HuggingFace. No manual intervention. No silent failures.

**Hybrid search**  
Vector similarity (semantic) + BM25 (keyword) combined. Neither alone is sufficient for legal/technical documents — together they handle both conceptual and exact-term queries.

**Multi-query retrieval**  
Each user question spawns 3 related queries. Retrieved chunks are deduplicated and reranked. Dramatically improves recall on complex or ambiguous questions.

**OCR preprocessing**  
Scanned PDFs are rasterized page-by-page and processed through Tesseract before chunking. Tables and structured content are handled.

**Multilingual support**  
Auto-detects document language (Telugu, Hindi, English) and translates queries/answers accordingly.

**Chat history**  
Full Q&A log with source page numbers stored in the sidebar across the session.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM (primary) | Groq — `llama-3.1-8b-instant` |
| LLM (secondary) | OpenRouter |
| LLM (fallback) | HuggingFace Inference API |
| Embeddings | HuggingFace sentence-transformers |
| Vector store | FAISS |
| Keyword search | BM25 (rank_bm25) |
| OCR | Tesseract + pdf2image |
| UI | Streamlit |
| Language detection | langdetect |

---

## Quickstart

```bash
git clone https://github.com/Muneshshaganti/multirag.git
cd multirag
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

Run:

```bash
streamlit run multirag.py
```

---

## Why this design

Three separate RAG repos in this profile represent an intentional progression:

1. `rag-document-advisor` — cloud-based, single model, simple Chroma store
2. `Rag_Ollama` — fully offline, local LLM, DB-backed persistence
3. `multirag` (this repo) — multi-cloud, resilient, production-grade retrieval

Each solves a real constraint: API cost, data privacy, and uptime reliability.

---

## Topics

`rag` `llm` `langchain` `retrieval-augmented-generation` `groq` `openrouter` `huggingface` `ocr` `hybrid-search` `bm25` `faiss` `streamlit` `python` `document-ai` `multilingual`
# Legal RAG System

This is a Retrieval-Augmented Generation (RAG) system designed to handle legal or structured documents (PDFs) and answer questions based on their content. It uses multiple language models in fallback order, handles scanned documents via OCR, and supports multi-language translation.

## Features

- Upload PDF documents (including scanned or structured tables)
- Multi-query generation for improved retrieval accuracy
- Hybrid retrieval combining vector search (embeddings) and BM25 keyword search
- Fallback language models:
  - Groq (primary)
  - OpenRouter (secondary)
  - HuggingFace (fallback)
- OCR for scanned pages or tables using Tesseract
- Language detection and translation support (Telugu, Hindi)
- Chat history stored in the sidebar with full questions and answers along with source pages

## How It Works

1. Upload a PDF document.
2. The system extracts text using PDF parsing and OCR (if needed).
3. When you ask a question, it generates multiple related queries to retrieve the most relevant passages.
4. The retrieved passages are re-ranked, and the language model generates an answer.
5. The full Q&A with source pages is stored in the sidebar history.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
