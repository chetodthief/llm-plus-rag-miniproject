import os
import time

# ปิด Warning ของ Transformers (ถ้ามี)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag.movie_qa import setup_qa_system, answer_question, load_reranker

def main():
    print("="*60)
    print("🎬 CineRAG CLI Testing Tool")
    print("="*60)
    
    print("\n[1/3] Loading FAISS and BM25 Indexes...")
    start_time = time.time()
    vectorstore = setup_qa_system()
    print(f"✅ Indexes loaded in {time.time() - start_time:.2f}s")
    
    print("\n[2/3] Loading Cross-Encoder Re-ranker...")
    start_time = time.time()
    reranker = load_reranker()
    print(f"✅ Re-ranker loaded in {time.time() - start_time:.2f}s")
    
    print("\n[3/3] System Ready! Type 'exit' or 'quit' to stop.")
    print("="*60)
    
    while True:
        try:
            print("\n" + "-"*40)
            user_input = input("🗣️ Your Question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            print("\n🤖 Generating answer...")
            start_ans = time.time()
            result = answer_question(user_input, vectorstore, reranker=reranker)
            
            print(f"\n✅ Answer (Total Time: {time.time() - start_ans:.2f}s):")
            print("="*60)
            print(result["answer"])
            print("="*60)
            
            print("\n📚 Sources Used:")
            sources = result.get("sources", [])
            if not sources:
                print("No sources found.")
            else:
                for i, src in enumerate(sources, 1):
                    title = src.get("title", "Unknown")
                    year = src.get("year", "?")
                    genre = src.get("genre", "Unknown")
                    
                    # ตัดเนื้อหาให้สั้นลงตอน Preview
                    content_preview = src.get("content", "")
                    if len(content_preview) > 100:
                        content_preview = content_preview[:100] + "..."
                        
                    print(f"[{i}] {title} ({year}) | Genre: {genre}")
                    print(f"    Snippets: {content_preview}")
                    
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")

if __name__ == "__main__":
    main()
