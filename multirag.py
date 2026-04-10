
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import streamlit as st
# import tempfile
# import re
# import requests
# from dotenv import load_dotenv
# load_dotenv()

# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image

# from langdetect import detect
# from sentence_transformers import CrossEncoder

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.schema import Document
# from langchain_groq import ChatGroq


# # ---------------- KEY LOADER ----------------
# def get_key(key_name):
#     value = os.getenv(key_name)
#     if not value:
#         raise ValueError(f"{key_name} not found. Set it in .env")
#     return value


# # ---------------- ENV ----------------
# load_dotenv(dotenv_path=".env")
# import os

# print("GROQ:", os.getenv("GROQ_API_KEY"))

# GROQ_API_KEY = get_key("GROQ_API_KEY")
# OPENAI_API_KEY = get_key("OPENAI_API_KEY")   # OpenRouter
# HUGGINGFACE_API_KEY = get_key("HUGGINGFACE_API_KEY")

# openrouter_available = bool(OPENAI_API_KEY)


# # ---------------- GROQ ----------------
# if "groq" not in st.session_state:
#     st.session_state.groq = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0
#     )


# # ---------------- MULTI LLM ----------------
# def multi_llm(prompt):
    
#     # 1️⃣ GROQ FIRST
#     try:
#         st.info("🟠 Using Groq...")
#         response = st.session_state.groq.invoke(prompt)

#         if response.content and response.content.strip():
#             st.session_state["model_used"] = "Groq"
#             return response.content

#     except Exception as e:
#         st.warning(f"⚠️ Groq failed: {e}")

#     # 2️⃣ OPENROUTER FALLBACK
#     if openrouter_available:
#         try:
#             st.info("🔵 Using OpenRouter...")

#             response = requests.post(
#                 url="https://openrouter.ai/api/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {OPENAI_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": "openai/gpt-4o-mini",
#                     "messages": [
#                         {"role": "user", "content": prompt}
#                     ]
#                 }
#             )

#             result = response.json()

#             if "choices" in result and result["choices"]:
#                 text = result["choices"][0]["message"]["content"]

#                 if text.strip():
#                     st.session_state["model_used"] = "OpenRouter"
#                     return text

#         except Exception as e:
#             st.warning(f"⚠️ OpenRouter failed: {e}")

#     # 3️⃣ HUGGINGFACE FALLBACK
#     try:
#         from transformers import pipeline

#         st.info("🟢 Using HuggingFace...")

#         generator = pipeline(
#             "text2text-generation",
#             model="google/flan-t5-base"
#         )

#         result = generator(prompt, max_length=512)
#         text = result[0]["generated_text"]

#         if text.strip():
#             st.session_state["model_used"] = "HuggingFace"
#             return text

#     except Exception as e:
#         st.warning(f"⚠️ HuggingFace failed: {e}")

#     st.session_state["model_used"] = "None"
#     return "❌ All AI APIs failed."

# # ---------------- SESSION ----------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# if "reranker" not in st.session_state:
#     st.session_state.reranker = None


# # ---------------- FUNCTIONS ----------------

# def detect_language(text):
#     try:
#         return detect(text)
#     except:
#         return "en"


# def clean_text(text):
#     text = re.sub(r'\n+', '\n', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# def preprocess_image(img):
#     img = np.array(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 11, 2
#     )
#     return Image.fromarray(thresh)


# def process_pdf(uploaded_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         path = tmp.name

#     docs = []

#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):

#             text = page.extract_text()

#             if not text or len(text.strip()) < 50:
#                 try:
#                     img = page.to_image(resolution=300).original
#                     img = preprocess_image(img)
#                     text = pytesseract.image_to_string(img)
#                 except:
#                     text = ""

#             text = clean_text(text)

#             if len(text) > 50:
#                 docs.append(Document(page_content=text, metadata={"page": i+1}))

#     if not docs:
#         st.error("❌ No text extracted from PDF")
#         return None, None

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=300
#     )

#     chunks = splitter.split_documents(docs)

#     if not chunks:
#         st.error("❌ Chunking failed")
#         return None, None

#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5"
#     )

#     vectordb = Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=None
#     )

#     vector_retriever = vectordb.as_retriever(search_kwargs={"k": 40})

#     bm25 = BM25Retriever.from_documents(chunks)
#     bm25.k = 25

#     retriever = EnsembleRetriever(
#         retrievers=[bm25, vector_retriever],
#         weights=[0.5, 0.5]
#     )

#     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     return retriever, reranker


# def rerank(query, docs, reranker, top_k=8):
#     pairs = [[query, d.page_content] for d in docs]
#     scores = reranker.predict(pairs)
#     scored = list(zip(docs, scores))
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [x[0] for x in scored[:top_k]]


# # ---------------- UI ----------------

# st.title("⚖️ Legal RAG System")

# file = st.file_uploader("Upload PDF", type="pdf")

# if file:

#     if st.session_state.retriever is None:
#         with st.spinner("Processing PDF..."):
#             r, rr = process_pdf(file)

#             if r is None:
#                 st.stop()

#             st.session_state.retriever = r
#             st.session_state.reranker = rr

#         st.success("✅ PDF processed")

#     q = st.chat_input("Ask your question...")

#     if q:

#         lang = detect_language(q)

#         if lang == "te":
#             q = multi_llm(f"Translate Telugu to English:\n{q}")
#         elif lang == "hi":
#             q = multi_llm(f"Translate Hindi to English:\n{q}")

#         queries_text = multi_llm(f"Generate 5 queries:\n{q}")
#         queries = [x.strip() for x in queries_text.split("\n") if x.strip()]
#         queries.append(q)

#         all_docs = []
#         for query in queries:
#             all_docs.extend(st.session_state.retriever.invoke(query))

#         docs = list({d.page_content: d for d in all_docs}.values())
#         docs = rerank(q, docs, st.session_state.reranker)

#         if docs:
#             context = ""
#             for d in docs:
#                 context += f"\n--- PAGE {d.metadata.get('page')} ---\n{d.page_content}\n"

#             prompt = f"""
# Use ONLY the context. Quote law. Show page.

# Context:
# {context}

# Question:
# {q}

# Answer:
# """
#             ans = multi_llm(prompt)
#         else:
#             ans = "Not enough information"

#         st.session_state.chat_history.append(("user", q))
#         st.session_state.chat_history.append(("bot", ans))

#     for role, msg in st.session_state.chat_history:
#         st.chat_message("user" if role=="user" else "assistant").write(msg)

#     if st.session_state.chat_history:
#         last = st.session_state.chat_history[-1][1]

#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("Telugu"):
#                 st.write(multi_llm(f"Translate to Telugu:\n{last}"))

#         with col2:
#             if st.button("Hindi"):
#                 st.write(multi_llm(f"Translate to Hindi:\n{last}"))

#     if "model_used" in st.session_state:
#         st.info(f"🤖 Model Used: {st.session_state['model_used']}")



















# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import streamlit as st
# import tempfile
# import re
# import requests
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=".env")

# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image

# from langdetect import detect
# from sentence_transformers import CrossEncoder

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.schema import Document
# from langchain_groq import ChatGroq

# import uuid


# # ---------------- KEY LOADER ----------------
# def get_key(key_name):
#     value = os.getenv(key_name)
#     if not value:
#         raise ValueError(f"{key_name} not found. Set it in .env")
#     return value


# # ---------------- ENV ----------------
# GROQ_API_KEY = get_key("GROQ_API_KEY")
# OPENAI_API_KEY = get_key("OPENAI_API_KEY")
# HUGGINGFACE_API_KEY = get_key("HUGGINGFACE_API_KEY")

# openrouter_available = bool(OPENAI_API_KEY)


# # ---------------- GROQ ----------------
# if "groq" not in st.session_state:
#     st.session_state.groq = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0
#     )


# # ---------------- MULTI LLM ----------------
# def multi_llm(prompt):

#     # 1️⃣ GROQ FIRST
#     try:
#         st.info("🟠 Using Groq...")
#         response = st.session_state.groq.invoke(prompt)

#         if response.content and response.content.strip():
#             st.session_state["model_used"] = "Groq"
#             return response.content

#     except Exception as e:
#         st.warning(f"⚠️ Groq failed: {e}")

#     # 2️⃣ OPENROUTER
#     if openrouter_available:
#         try:
#             st.info("🔵 Using OpenRouter...")

#             response = requests.post(
#                 url="https://openrouter.ai/api/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {OPENAI_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": "openai/gpt-4o-mini",
#                     "messages": [{"role": "user", "content": prompt}]
#                 }
#             )

#             result = response.json()

#             if "choices" in result and result["choices"]:
#                 text = result["choices"][0]["message"]["content"]

#                 if text.strip():
#                     st.session_state["model_used"] = "OpenRouter"
#                     return text

#         except Exception as e:
#             st.warning(f"⚠️ OpenRouter failed: {e}")

#     # 3️⃣ HUGGINGFACE
#     try:
#         from transformers import pipeline

#         st.info("🟢 Using HuggingFace...")

#         generator = pipeline(
#             "text2text-generation",
#             model="google/flan-t5-base"
#         )

#         result = generator(prompt, max_length=512)
#         text = result[0]["generated_text"]

#         if text.strip():
#             st.session_state["model_used"] = "HuggingFace"
#             return text

#     except Exception as e:
#         st.warning(f"⚠️ HuggingFace failed: {e}")

#     return "❌ All AI APIs failed."


# # ---------------- SESSION ----------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# if "reranker" not in st.session_state:
#     st.session_state.reranker = None


# # ---------------- FUNCTIONS ----------------

# def detect_language(text):
#     try:
#         return detect(text)
#     except:
#         return "en"


# def clean_text(text):
#     text = re.sub(r'\n+', '\n', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# def preprocess_image(img):
#     img = np.array(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 11, 2
#     )
#     return Image.fromarray(thresh)


# def process_pdf(uploaded_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         path = tmp.name

#     docs = []

#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):

#             text = page.extract_text()

#             if not text or len(text.strip()) < 50:
#                 try:
#                     img = page.to_image(resolution=300).original
#                     img = preprocess_image(img)
#                     text = pytesseract.image_to_string(img)
#                 except:
#                     text = ""

#             text = clean_text(text)

#             if len(text) > 50:
#                 docs.append(Document(page_content=text, metadata={"page": i+1}))

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=300
#     )

#     chunks = splitter.split_documents(docs)

#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5"
#     )

#     db_path = f"chroma_{uuid.uuid4().hex}"

#     vectordb = Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=db_path
#     )

#     vector_retriever = vectordb.as_retriever(search_kwargs={"k": 15})

#     bm25 = BM25Retriever.from_documents(chunks)
#     bm25.k = 25

#     retriever = EnsembleRetriever(
#         retrievers=[bm25, vector_retriever],
#         weights=[0.5, 0.5]
#     )

#     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     return retriever, reranker


# def rerank(query, docs, reranker, top_k=8):
#     pairs = [[query, d.page_content] for d in docs]
#     scores = reranker.predict(pairs)
#     scored = list(zip(docs, scores))
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [x[0] for x in scored[:top_k]]


# # ---------------- UI ----------------

# st.title("⚖️ Legal RAG System")

# file = st.file_uploader("Upload PDF", type="pdf")

# if file:

#     if "current_file" not in st.session_state:
#         st.session_state.current_file = None

#     if st.session_state.current_file != file.name:
#         st.session_state.retriever = None
#         st.session_state.reranker = None
#         st.session_state.current_file = file.name

#     if st.session_state.retriever is None:
#         with st.spinner("Processing PDF..."):
#             r, rr = process_pdf(file)
#             st.session_state.retriever = r
#             st.session_state.reranker = rr

#         st.success("✅ PDF processed")

#     q = st.chat_input("Ask your question...")

#     if q:

#         all_docs = []

#         for query in [q]:
#             if st.session_state.retriever is None:
#                 st.error("Retriever not initialized")
#                 st.stop()

#             all_docs.extend(st.session_state.retriever.invoke(query))

#         docs = list({d.page_content: d for d in all_docs}.values())
#         docs = rerank(q, docs, st.session_state.reranker)

#         sources = list(set([d.metadata.get("page", "N/A") for d in docs]))

#         context = "\n".join([d.page_content for d in docs])

#         prompt = f"Context:\n{context}\n\nQuestion:\n{q}\nAnswer:"
#         ans = multi_llm(prompt)

#         # 🔥 store history (max 10)
#         st.session_state.chat_history.append({
#             "question": q,
#             "answer": ans,
#             "sources": sources
#         })

#         if len(st.session_state.chat_history) > 10:
#             st.session_state.chat_history.pop(0)

#         st.write(ans)


# # ---------------- SIDEBAR HISTORY ----------------
# st.sidebar.title("📚 Chat History")

# if st.session_state.chat_history:
#     for i, item in enumerate(reversed(st.session_state.chat_history)):
#         with st.sidebar.expander(f"Q{i+1}: {item['question'][:40]}..."):
#             st.markdown(f"**Question:** {item['question']}")
#             st.markdown(f"**Answer:** {item['answer']}")
#             st.markdown(f"**Source Pages:** {item['sources']}")
# else:
#     st.sidebar.info("No history yet")



















# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import streamlit as st
# import tempfile
# import re
# import requests
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=".env")

# import pdfplumber
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image

# from langdetect import detect
# from sentence_transformers import CrossEncoder

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.schema import Document
# from langchain_groq import ChatGroq

# import uuid


# # ---------------- KEY LOADER ----------------
# def get_key(key_name):
#     value = os.getenv(key_name)
#     if not value:
#         raise ValueError(f"{key_name} not found. Set it in .env")
#     return value


# # ---------------- ENV ----------------
# GROQ_API_KEY = get_key("GROQ_API_KEY")
# OPENAI_API_KEY = get_key("OPENAI_API_KEY")
# HUGGINGFACE_API_KEY = get_key("HUGGINGFACE_API_KEY")

# openrouter_available = bool(OPENAI_API_KEY)


# # ---------------- GROQ ----------------
# if "groq" not in st.session_state:
#     st.session_state.groq = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0
#     )


# # ---------------- MULTI LLM ----------------
# def multi_llm(prompt):

#     # 1️⃣ GROQ FIRST
#     try:
#         st.info("🟠 Using Groq...")
#         response = st.session_state.groq.invoke(prompt)

#         if response.content and response.content.strip():
#             st.session_state["model_used"] = "Groq"
#             return response.content

#     except Exception as e:
#         st.warning(f"⚠️ Groq failed: {e}")

#     # 2️⃣ OPENROUTER
#     if openrouter_available:
#         try:
#             st.info("🔵 Using OpenRouter...")

#             response = requests.post(
#                 url="https://openrouter.ai/api/v1/chat/completions",
#                 headers={
#                     "Authorization": f"Bearer {OPENAI_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "model": "openai/gpt-4o-mini",
#                     "messages": [{"role": "user", "content": prompt}]
#                 }
#             )

#             result = response.json()

#             if "choices" in result and result["choices"]:
#                 text = result["choices"][0]["message"]["content"]

#                 if text.strip():
#                     st.session_state["model_used"] = "OpenRouter"
#                     return text

#         except Exception as e:
#             st.warning(f"⚠️ OpenRouter failed: {e}")

#     # 3️⃣ HUGGINGFACE
#     try:
#         from transformers import pipeline

#         st.info("🟢 Using HuggingFace...")

#         generator = pipeline(
#             "text2text-generation",
#             model="google/flan-t5-base"
#         )

#         result = generator(prompt, max_length=512)
#         text = result[0]["generated_text"]

#         if text.strip():
#             st.session_state["model_used"] = "HuggingFace"
#             return text

#     except Exception as e:
#         st.warning(f"⚠️ HuggingFace failed: {e}")

#     return "❌ All AI APIs failed."


# # ---------------- SESSION ----------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# if "reranker" not in st.session_state:
#     st.session_state.reranker = None


# # ---------------- FUNCTIONS ----------------

# def detect_language(text):
#     try:
#         return detect(text)
#     except:
#         return "en"


# def clean_text(text):
#     text = re.sub(r'\n+', '\n', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# def preprocess_image(img):
#     img = np.array(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 11, 2
#     )
#     return Image.fromarray(thresh)


# def process_pdf(uploaded_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         path = tmp.name

#     docs = []

#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):

#             text = page.extract_text()

#             if not text or len(text.strip()) < 50:
#                 try:
#                     img = page.to_image(resolution=300).original
#                     img = preprocess_image(img)
#                     text = pytesseract.image_to_string(img)
#                 except:
#                     text = ""

#             text = clean_text(text)

#             if len(text) > 50:
#                 docs.append(Document(page_content=text, metadata={"page": i+1}))

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=300
#     )

#     chunks = splitter.split_documents(docs)

#     embeddings = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-base-en-v1.5"
#     )

#     db_path = f"chroma_{uuid.uuid4().hex}"

#     vectordb = Chroma.from_documents(
#         chunks,
#         embeddings,
#         persist_directory=db_path
#     )

#     vector_retriever = vectordb.as_retriever(search_kwargs={"k": 15})

#     bm25 = BM25Retriever.from_documents(chunks)
#     bm25.k = 25

#     retriever = EnsembleRetriever(
#         retrievers=[bm25, vector_retriever],
#         weights=[0.5, 0.5]
#     )

#     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     return retriever, reranker


# def rerank(query, docs, reranker, top_k=8):
#     pairs = [[query, d.page_content] for d in docs]
#     scores = reranker.predict(pairs)
#     scored = list(zip(docs, scores))
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [x[0] for x in scored[:top_k]]


# # ---------------- UI ----------------

# st.title("⚖️ Legal RAG System")

# file = st.file_uploader("Upload PDF", type="pdf")

# if file:

#     if "current_file" not in st.session_state:
#         st.session_state.current_file = None

#     if st.session_state.current_file != file.name:
#         st.session_state.retriever = None
#         st.session_state.reranker = None
#         st.session_state.current_file = file.name

#     if st.session_state.retriever is None:
#         with st.spinner("Processing PDF..."):
#             r, rr = process_pdf(file)
#             st.session_state.retriever = r
#             st.session_state.reranker = rr

#         st.success("✅ PDF processed")

#     q = st.chat_input("Ask your question...")

#     if q:

#         all_docs = []

#         for query in [q]:
#             if st.session_state.retriever is None:
#                 st.error("Retriever not initialized")
#                 st.stop()

#             all_docs.extend(st.session_state.retriever.invoke(query))

#         docs = list({d.page_content: d for d in all_docs}.values())
#         docs = rerank(q, docs, st.session_state.reranker)

#         sources = list(set([d.metadata.get("page", "N/A") for d in docs]))

#         context = "\n".join([d.page_content for d in docs])

#         prompt = f"Context:\n{context}\n\nQuestion:\n{q}\nAnswer:"
#         ans = multi_llm(prompt)

#         # 🔥 store history (max 10)
#         st.session_state.chat_history.append({
#             "question": q,
#             "answer": ans,
#             "sources": sources
#         })

#         if len(st.session_state.chat_history) > 10:
#             st.session_state.chat_history.pop(0)

#         st.write(ans)


# # ---------------- SIDEBAR HISTORY ----------------
# st.sidebar.title("📚 Chat History")

# if st.session_state.chat_history:
#     for i, item in enumerate(reversed(st.session_state.chat_history)):
#         with st.sidebar.expander(f"Q{i+1}: {item['question'][:40]}..."):
#             st.markdown(f"**Question:** {item['question']}")
#             st.markdown(f"**Answer:** {item['answer']}")
#             st.markdown(f"**Source Pages:** {item['sources']}")
# else:
#     st.sidebar.info("No history yet")















import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
import re
import requests
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

from langdetect import detect
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.retrievers.ensemble import EnsembleRetriever
import uuid


# ---------------- KEY LOADER ----------------
def get_key(key_name):
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"{key_name} not found. Set it in .env")
    return value


# ---------------- ENV ----------------
GROQ_API_KEY = get_key("GROQ_API_KEY")
OPENAI_API_KEY = get_key("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = get_key("HUGGINGFACE_API_KEY")

openrouter_available = bool(OPENAI_API_KEY)


# ---------------- GROQ ----------------
if "groq" not in st.session_state:
    st.session_state.groq = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )


# ---------------- MULTI LLM ----------------
def multi_llm(prompt):

    try:
        response = st.session_state.groq.invoke(prompt)
        if response.content and response.content.strip():
            st.session_state["model_used"] = "Groq"
            return response.content
    except:
        pass

    if openrouter_available:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
        except:
            pass

    try:
        from transformers import pipeline
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        result = generator(prompt, max_length=512)
        return result[0]["generated_text"]
    except:
        return "❌ All AI APIs failed."


# ---------------- SESSION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None


# ---------------- FUNCTIONS ----------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):

            text = page.extract_text()

            # FIX: faster OCR
            if not text or len(text.strip()) < 50:
                try:
                    img = page.to_image(resolution=150)
                    img = preprocess_image(img.original)
                    text = pytesseract.image_to_string(img)
                except:
                    text = ""

            text = clean_text(text)

            if len(text) > 50:
                docs.append(Document(page_content=text, metadata={"page": i+1}))

    if not docs:
        st.error("❌ No text extracted from PDF")
        return None, None

    # FIX: better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ No chunks created")
        return None, None

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    db_path = f"chroma_{uuid.uuid4().hex}"

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_path
    )

    # FIX: stronger retrieval
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 30})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 25

    retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5]
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return retriever, reranker


def rerank(query, docs, reranker, top_k=8):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]


# ---------------- UI ----------------

st.title("⚖️ Legal RAG System")

file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

if file:

    if st.session_state.retriever is None:
        with st.spinner("Processing PDF..."):
            r, rr = process_pdf(file)

            if r is None:
                st.stop()

            st.session_state.retriever = r
            st.session_state.reranker = rr

        st.success("✅ PDF processed")

    q = st.chat_input("Ask your question...")

    if q:

        lang = detect_language(q)

        if lang == "te":
            q = multi_llm(f"Translate Telugu to English:\n{q}")
        elif lang == "hi":
            q = multi_llm(f"Translate Hindi to English:\n{q}")

        # FIX: multi-query + number boost
        queries_text = multi_llm(f"Generate 5 search queries for:\n{q}")
        queries = [x.strip() for x in queries_text.split("\n") if x.strip()]
        queries.append(q)

        if any(char.isdigit() for char in q):
            queries.append(q + " number details Aadhaar mobile passbook")

        all_docs = []
        for query in queries:
            all_docs.extend(st.session_state.retriever.invoke(query))

        docs = list({d.page_content: d for d in all_docs}.values())
        docs = rerank(q, docs, st.session_state.reranker)

        if docs:
            context = ""
            for d in docs:
                context += f"\n--- PAGE {d.metadata.get('page')} ---\n{d.page_content}\n"

            prompt = f"""
Use ONLY the context.
If numbers (mobile, Aadhaar, passbook) exist, extract EXACTLY.
Do NOT guess.

Context:
{context}

Question:
{q}

Answer:
"""
            ans = multi_llm(prompt)

            sources = list(set([d.metadata.get("page", "N/A") for d in docs]))

            st.session_state.chat_history.append({
                "question": q,
                "answer": ans,
                "sources": sources
            })

            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history.pop(0)

            # ✅ SHOW ONLY CURRENT Q&A (ADDED)
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(ans)

        else:
            ans = "Not enough information"

    # ❌ REMOVED OLD HISTORY LOOP (DO NOT ADD BACK)

    # Translation buttons (unchanged)
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]["answer"]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Telugu"):
                st.write(multi_llm(f"Translate to Telugu:\n{last}"))

        with col2:
            if st.button("Hindi"):
                st.write(multi_llm(f"Translate to Hindi:\n{last}"))

    if "model_used" in st.session_state:
        st.info(f"🤖 Model Used: {st.session_state['model_used']}")


# ---------------- SIDEBAR ----------------
st.sidebar.title("📚 Chat History")

if st.session_state.chat_history:
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        with st.sidebar.expander(f"Q: {item['question']}"):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(f"**Source Pages:** {item['sources']}")
else:
    st.sidebar.info("No history yet")