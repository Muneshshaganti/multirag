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