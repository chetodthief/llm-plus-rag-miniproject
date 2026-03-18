import os
import json
import pickle
import time # เพิ่มโมดูล time สำหรับจับเวลา
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def build_vector_index(input_json, output_dir):
    print(f"Loading chunks from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    # --- 1. ปรับปรุง: ดึงข้อความมาใช้ตรงๆ ป้องกันการใส่ Metadata ซ้ำซ้อน ---
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # We use a multilingual model so users can ask questions in Thai
    # and it can match against English movie plots.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Automatically use GPU if available for massively faster embedding generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name} on {device}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={
            'batch_size': 128,          # Increase batch size for speed
        }
    )
    
    # --- 2. ปรับปรุง: เพิ่มระบบจับเวลาเพื่อให้รู้ว่าประมวลผลไปถึงไหนแล้ว ---
    print(f"\nBuilding BM25 index for {len(texts)} chunks... (This is usually fast)")
    start_time = time.time()
    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
    print(f"✅ BM25 index built in {time.time() - start_time:.2f} seconds.")

    print(f"\nBuilding FAISS index for {len(texts)} chunks... (This will take a while!)")
    start_time = time.time()
    # FAISS.from_texts generates embeddings for the texts and builds the index
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    print(f"✅ FAISS index built in {time.time() - start_time:.2f} seconds.")
    
    # Save the index locally
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    
    bm25_path = os.path.join(output_dir, "bm25_retriever.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    print(f"\n🎉 All indices successfully saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "chunked_plots.json")
    output_dir = os.path.join(base_dir, "data", "faiss_index")
    build_vector_index(input_file, output_dir)