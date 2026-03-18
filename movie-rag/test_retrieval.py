"""
test_retrieval.py
=================
ไฟล์ทดสอบระบบ Retrieval (FAISS + BM25 Hybrid Search)
ใช้สำหรับตรวจสอบว่าระบบดึงข้อมูลถูกต้องและครบถ้วนหรือไม่
โดยไม่ผ่าน LLM (ไม่ต้องรอ Ollama)
"""

import os
import sys
import pickle
import re
from deep_translator import GoogleTranslator

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache'

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever

# ==========================================
# Test Cases
# ==========================================
TEST_QUERIES = [
    # Specific Movie Questions
    "Titanic 1997 ตอนจบเป็นยังไง",
    "The Matrix ใครคือ Neo",
    "ใน Inception ความฝันทำงานอย่างไร",

    # Genre / Recommendation Queries
    "แนะนำหนังผีหน่อย",
    "หนัง action ที่ดีมีอะไรบ้าง",
    "หนัง sci-fi ในอวกาศ",

    # Plot Identification
    "ผู้ชายติดอยู่บนดาวอังคาร",
    "ตัวตลกปีศาจออกมาทุก 27 ปี",
    "เด็กเห็นผีได้คนเดียว",

    # Edge Cases
    "ไก่กับไข่อะไรเกิดก่อน",  # Off-topic -> should return unrelated/empty
    "The Lion King Mufasa ถูกฆ่าโดยใคร",
    "หนังลิง 3 เรื่อง",
]

def is_thai(text):
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def translate_to_english(question):
    if not is_thai(question):
        return question, False

    eng_phrases = re.findall(r'[a-zA-Z0-9\s\-\:\']+', question)
    eng_phrases = [p.strip() for p in eng_phrases if p.strip()]

    temp_q = question
    placeholders = {}
    for i, phrase in enumerate(eng_phrases):
        placeholder = f"__ENG_{i}__"
        temp_q = temp_q.replace(phrase, f" {placeholder} ")
        placeholders[placeholder] = phrase

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(temp_q)
        for placeholder, phrase in placeholders.items():
            translated = translated.replace(placeholder, phrase)
        return translated, True
    except Exception as e:
        print(f"  [Translation Error] {e}")
        return question, True

def load_vectorstore(index_dir):
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    bm25_path = os.path.join(index_dir, "bm25_retriever.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 12

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    print("Vectorstore loaded successfully.\n")
    return ensemble_retriever

def run_retrieval_test(vectorstore, query):
    english_query, was_thai = translate_to_english(query)

    if was_thai and english_query != query:
        print(f"  📘 Translated : {english_query}")

    docs = vectorstore.invoke(english_query)

    if not docs:
        print("  ⚠️  No documents retrieved.\n")
        return

    # Count chunks per movie
    from collections import Counter
    title_counts = Counter([d.metadata.get("title", "Unknown") for d in docs])
    print(f"  📦 Retrieved {len(docs)} chunks from {len(title_counts)} unique movie(s):")
    for title, count in title_counts.most_common():
        year = next((d.metadata.get("year", "?") for d in docs if d.metadata.get("title") == title), "?")
        genre = next((d.metadata.get("genre", "?") for d in docs if d.metadata.get("title") == title), "?")
        print(f"     🎬 {title} ({year}) [{genre}] — {count} chunk(s)")

    print()
    print("  📄 Top 3 Chunk Previews:")
    for i, doc in enumerate(docs[:3]):
        meta = doc.metadata
        chunk_idx = meta.get("chunk_index", "?")
        total = meta.get("total_chunks", "?")
        title = meta.get("title", "?")
        content_preview = doc.page_content[:120].replace("\n", " ")
        print(f"     [{i+1}] {title} [Part {chunk_idx}/{total}]: {content_preview}...")

    print()

def main():
    # Allow overriding index directory via command line
    if len(sys.argv) > 1:
        index_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_dir = os.path.join(base_dir, "data", "faiss_index")

    if not os.path.exists(index_dir):
        print(f"❌ Error: Index not found at {index_dir}")
        print("   Please run: python embeddings/build_index.py")
        sys.exit(1)

    vectorstore = load_vectorstore(index_dir)

    print("=" * 65)
    print("   🧪 RETRIEVAL TEST — FAISS + BM25 Hybrid Search")
    print("=" * 65)
    print()

    for i, query in enumerate(TEST_QUERIES):
        print(f"── Query {i+1}/{len(TEST_QUERIES)} ──────────────────────────────────────")
        print(f"  ❓ Query     : {query}")
        run_retrieval_test(vectorstore, query)

    print("=" * 65)
    print("   ✅ All retrieval tests complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()