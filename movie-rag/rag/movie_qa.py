import os
import sys
import pickle
import re
import time
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# --- 1. ตั้งค่า Cache ---
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache'

# โหลดตัวแปรจากไฟล์ .env
load_dotenv()

# --- 2. Imports Library ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.ensemble import EnsembleRetriever

def load_vectorstore(index_dir):
    """โหลด FAISS และ BM25 เพื่อทำงานร่วมกันแบบ Hybrid Search"""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    # 1. ปรับปรุง: เพิ่มจำนวน K จาก 5 เป็น 12 เพื่อให้กวาดบริบทเริ่มต้นได้กว้างขึ้น
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    
    bm25_path = os.path.join(index_dir, "bm25_retriever.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    bm25_retriever.k = 12
    
    # 2. ปรับปรุง: ให้น้ำหนัก Semantic Search (FAISS) 60% และ Keyword (BM25) 40%
    # เพราะการถามเนื้อเรื่องหนัง มักเป็นการถามด้วยความหมายมากกว่าคำเป๊ะๆ
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )
    return ensemble_retriever

def load_llm():
    """โหลด Llama 3.2 ผ่าน Ollama"""
    print("Loading Llama 3.2 via Ollama...")
    llm = OllamaLLM(
        model="llama3.2", 
        temperature=0.0,     # 0.0 = deterministic, prevents hallucination
        num_ctx=8192,        # Set context window specifically to prevent hangs
    )
    return llm

# --- 3. Prompt Engineering ---
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Cinephile AI and a highly precise Movie Search Engine. Your sole purpose is to discuss cinema, identify movies from plot descriptions, and recommend films.

<rules>
0. CONTEXT IS YOUR BIBLE: You are strictly forbidden from using your pre-trained internet knowledge to supply movie data, storylines, or titles. You must ONLY formulate your answer using the data provided in the <movie_database> below. If the provided database does not contain the answer, you MUST refuse to answer.
1. ALWAYS ANSWER IN ENGLISH: You must generate your response in English, as it will be translated by the system later.
2. STRICTLY CINEMA (NO ENCYCLOPEDIA): Treat EVERY user query as a movie-related question. For example, if asked about "a man stuck on Mars" or "dinosaurs in a park", discuss "The Martian" or "Jurassic Park". NEVER write factual essays about real-world science, history, or general knowledge.
3. OUT-OF-SCOPE FALLBACK: If a query is completely unrelated to movies and cannot possibly be linked to any film, reply EXACTLY with this phrase (do not translate it): "ผมเป็นผู้เชี่ยวชาญด้านภาพยนตร์เท่านั้น 🎬 กรุณาถามเกี่ยวกับหนังนะครับ!"
4. RECOMMENDATION RULE: If the user asks for recommendations (e.g., "suggest action movies" or "แนะนำหนังผี"), you MUST ONLY recommend movies explicitly listed in the <movie_database> below. If you list a movie that is not in the <movie_database>, you have failed. If no movies in the <movie_database> match the request, you MUST reply EXACTLY with: "ในฐานข้อมูลตอนนี้ยังไม่มีหนังแนวนี้นะครับ ขอลองถามแนวอื่นดูนะครับ 🍿" AND DO NOT INVENT MOVIES.
5. PLOT IDENTIFICATION RULE: If the user describes a specific plot to find a movie title:
   - First, identify it using the <movie_database>.
   - If the exact movie is clearly missing from the <movie_database>, you MAY use your internal cinephile knowledge to name the ONE most likely movie, but keep the plot explanation concise.
6. CHRONOLOGICAL ACCURACY: When summarizing plots, strictly follow the `[Part X/Y]` markers in the context to ensure events are described in the correct order. Do not mix up the beginning, middle, and end.
7. TONE & FORMAT: Speak confidently. NEVER use robotic transition phrases like "Based on the provided context..." or "According to the database...". Start your answer immediately.
8. PROPER NOUN PROTECTION: You MUST wrap all movie titles, character names, and important English proper nouns in `<eng>` and `</eng>` tags so they are not translated later. Example: The <eng>Joker</eng> fights <eng>Batman</eng> in <eng>The Dark Knight</eng>.
</rules>

<movie_database>
{context}
</movie_database><|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def is_thai(text):
    """เช็คว่ามีตัวอักษรภาษาไทยในข้อความหรือไม่"""
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def answer_question(question, vectorstore, llm, filter_year=None, filter_title=None):
    """ค้นหาข้อมูล แปลภาษา และส่งให้ LLM ตอบคำถาม รองรับการกรองข้อมูลด้วย filter_year และ filter_title"""
    
    total_start = time.time()

    # ==========================================
    # 1. แปลภาษา (ขาเข้า)
    # ==========================================
    step_start = time.time()
    is_thai_query = is_thai(question)
    if is_thai_query:
        print(f"\n[Translator In] Original: {question}")
        
        # Protect English phrases (like movie titles "A Ghost Story") from being translated to Thai ("เรื่องผี")
        # Extract English phrases
        import re
        eng_phrases = re.findall(r'[a-zA-Z0-9\s\-\:]+', question)
        # Filter out empty or whitespace only strings
        eng_phrases = [p.strip() for p in eng_phrases if p.strip()]
        
        # Replace with placeholders
        temp_q = question
        placeholders = {}
        for i, phrase in enumerate(eng_phrases):
            placeholder = f"__ENG_PHRASE_{i}__"
            temp_q = temp_q.replace(phrase, f" {placeholder} ")
            placeholders[placeholder] = phrase
            
        translated_q = GoogleTranslator(source='auto', target='en').translate(temp_q)
        
        # Restore English phrases
        for placeholder, phrase in placeholders.items():
            translated_q = translated_q.replace(placeholder, phrase)
            
        english_question = translated_q
        print(f"[Translator In] Translated: {english_question}")
    else:
        english_question = question
    print(f"⏱️ Translation In: {time.time() - step_start:.2f}s")

    # ==========================================
    # 2. ค้นหาข้อมูล Hybrid (with optional filter)
    # ==========================================
    step_start = time.time()
    print("\n[Retrieval] Searching FAISS + BM25...")

    if filter_year or filter_title:
        # ถ้ามีการกรอง ให้ใช้ FAISS โดยตรงกับ metadata filter
        faiss_vs = vectorstore.retrievers[1].vectorstore
        filter_dict = {}
        if filter_year:
            filter_dict["year"] = str(filter_year)
        if filter_title:
            filter_dict["title"] = str(filter_title)
            
        # ⚠️ CRITICAL OPTIMIZATION: Reduce k from 100 to 25 to prevent context overflow
        # Even 25 chunks of 300 tokens is ~7,500 tokens, which is safer for a local LLM.
        k_val = 25 if filter_title else 8
        docs = faiss_vs.similarity_search(
            english_question, 
            k=k_val, 
            filter=filter_dict,
            fetch_k=50000
        )
        
        # Sort chunks chronologically so the story makes sense from start to finish
        if filter_title:
            docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            
        print(f"[Retrieval] Filtered FAISS by: {filter_dict}. Found {len(docs)} chunks.")
    else:
        # First Pass: Broad search to detect intent
        initial_docs = vectorstore.invoke(english_question)
        
        # Count movies in initial results
        movie_counts = {}
        for d in initial_docs:
            title = d.metadata.get("title")
            if title:
                movie_counts[title] = movie_counts.get(title, 0) + 1
                
        # Find the majority movie
        if movie_counts:
            top_movie, top_count = max(movie_counts.items(), key=lambda x: x[1])
            
            # If a single movie dominates the results (e.g., 3 or more out of the top results)
            # We assume the user is asking specifically about THIS movie (Dynamic Small-to-Big Retrieval)
            if top_count >= 3:
                print(f"[Retrieval] Auto-detected specific movie focus: '{top_movie}' (Found {top_count} chunks)")
                faiss_vs = vectorstore.retrievers[1].vectorstore
                
                docs = faiss_vs.similarity_search(
                    english_question,
                    k=25, # Reduced from 100 to prevent context hang
                    filter={"title": top_movie},
                    fetch_k=50000
                )
                # Sort chronologically so the story makes sense from start to finish
                docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
                
                # Append the original diverse chunks at the end to ensure we still get diverse posters for UI
                # Deduplicate by content so we don't pass the same chunk twice
                seen_content = set([d.page_content for d in docs])
                for init_d in initial_docs:
                    if init_d.page_content not in seen_content:
                        docs.append(init_d)
                        
                print(f"[Retrieval] Dynamic Small-to-Big: Fetched {len(docs)} focused chunks for '{top_movie}' + diverse fallback")
            else:
                # No strong majority -> Broad question (e.g., "recommend action movies")
                print(f"[Retrieval] Broad topic detected. Keeping original diverse chunks.")
                docs = initial_docs
        else:
            docs = initial_docs
    
    print(f"⏱️ Retrieval: {time.time() - step_start:.2f}s [{len(docs)} docs]")

    # ==========================================
    # 3. สร้าง Context และ Related Movies
    # ==========================================
    contexts = []
    sources = []
    
    # ดึงรายชื่อหนังทั้งหมดที่ FAISS หาเจอมาเซฟไว้เป็น "หนังที่คล้ายกันจากใน Database"
    related_movies = []
    seen_related = set()
    
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        genre = doc.metadata.get("genre", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        director = doc.metadata.get("director", "Unknown")
        cast = doc.metadata.get("cast", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        total_chunks = doc.metadata.get("total_chunks", "?")
        
        # เพิ่มเข้า related_movies ถ้ายังไม่มี
        movie_key = f"{title}_{year}"
        if movie_key not in seen_related:
            seen_related.add(movie_key)
            related_movies.append({
                "title": title,
                "year": year,
                "genre": genre
            })
            
        # 3. ปรับปรุง: แทรกตัวเลข Part X/Y ให้ LLM รู้ลำดับเหตุการณ์ของเนื้อเรื่อง
        context_str = (
            f"Title: {title} ({year})\n"
            f"Director: {director} | Cast: {cast} | Genre: {genre}\n"
            f"Plot [Part {chunk_idx}/{total_chunks}]: {doc.page_content}"
        )
        contexts.append(context_str)
        
        sources.append({
            "title": title,
            "genre": genre,
            "year": year,
            "director": director,
            "cast": cast,
            "score": "Hybrid",
            "content": doc.page_content
        })

    # ==========================================
    # 4. LLM Generation
    # ==========================================
    step_start = time.time()
    combined_context = "\n\n".join(contexts)
    final_prompt = PROMPT.format(context=combined_context, question=english_question)
    
    # Calculate approx tokens (Rough estimate: 1 word ~ 1.3 tokens)
    approx_tokens = len(final_prompt.split()) * 1.3
    print(f"\n[LLM] Generating answer via Ollama (Llama 3)...")
    print(f"[LLM] Context Chunks: {len(contexts)} | Approx Tokens: {approx_tokens:.0f}")
    
    english_answer = llm.invoke(final_prompt).strip()
    print(f"⏱️ LLM Generation: {time.time() - step_start:.2f}s")

    # ==========================================
    # 5. แปลภาษา (ขาออก)
    # ==========================================
    step_start = time.time()
    if is_thai_query:
        print("[Translator Out] Translating back to Thai...")
        
        # Protect specific movie titles and proper nouns from being translated
        import re
        
        protected_answer = english_answer
        placeholders = {}
        
        # 1. Protect <eng> tags generated by LLM (Rule 8)
        eng_tags = list(set(re.findall(r'<eng>(.*?)</eng>', protected_answer)))
        for i, phrase in enumerate(eng_tags):
            tag_str = f"<eng>{phrase}</eng>"
            placeholder = f"[[PROTECTED_ENG_{i}]]"
            protected_answer = protected_answer.replace(tag_str, placeholder)
            placeholders[placeholder] = phrase
            
        # 2. Protect contextual movie titles (Fallback)
        context_titles = list(set([src["title"] for src in sources]))
        context_titles.sort(key=len, reverse=True)
        for i, title in enumerate(context_titles):
            pattern = r'(?<!\w)' + re.escape(title) + r'(?!\w)'
            if re.search(pattern, protected_answer):
                placeholder = f"[[PROTECTED_TITLE_{i}]]"
                protected_answer = re.sub(pattern, placeholder, protected_answer)
                placeholders[placeholder] = title
                
        # Remove any lingering <eng> tags just in case
        protected_answer = re.sub(r'</?eng>', '', protected_answer)
                
        # Translate the text with placeholders
        translated = GoogleTranslator(source='en', target='th').translate(protected_answer)
        
        # Restore the original phrases
        for placeholder, original in placeholders.items():
            translated = translated.replace(placeholder, original)
            
        final_answer = translated
    else:
        # Strip <eng> tags for English queries
        import re
        final_answer = re.sub(r'</?eng>', '', english_answer)
    print(f"⏱️ Translation Out: {time.time() - step_start:.2f}s")
    
    print(f"\n✅ Total Time: {time.time() - total_start:.2f}s")
    print("=" * 50)
        
    return {
        "answer": final_answer,
        "sources": sources,
        "related_movies": related_movies
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(base_dir, "data", "faiss_index")
    
    if not os.path.exists(index_dir):
        print(f"Error: Vector index not found at {index_dir}. Run build_index.py first.")
        sys.exit(1)
        
    print("Loading resources...")
    vs = load_vectorstore(index_dir)
    llm_obj = load_llm()
    
    test_q = "แนะนำหนังแนวผีหลอกๆ ให้หน่อยครับ"
    result = answer_question(test_q, vs, llm_obj)
    
    print("\n" + "="*50)
    print(f"Question: {test_q}")
    print(f"Answer: {result['answer']}")
    print("="*50)
    print("\nSources:")
    for i, src in enumerate(result['sources']):
        print(f"{i+1}. Movie: {src['title']} ({src['year']}) - Genre: {src['genre']}")