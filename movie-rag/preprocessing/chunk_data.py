import os
import pandas as pd
import json
import re
from langchain_text_splitters import TokenTextSplitter

def clean_text(text):
    if pd.isna(text):
        return ""
    
    # บังคับแปลงเป็น string ป้องกันบั๊กกรณีเจอตัวเลขหลงมาใน Column เนื้อเรื่อง
    text = str(text)
    
    # 1. ลบเชิงอรรถ/อ้างอิง เช่น [1], [22], [Note 1] (Noise ตัวร้ายที่กวนค่าความหมาย)
    text = re.sub(r'\[\d+\]|\[Note\s\d+\]', '', text)
    
    # 2. ลบเครื่องหมายคำพูดแปลกๆ หรือสัญลักษณ์ที่ไม่ได้ช่วยเรื่องความหมาย
    text = text.replace('"', '').replace("'", "")
    
    # 3. จัดการช่องว่างให้คงที่ (Extra whitespaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_and_chunk_data(input_csv, output_json):
    print(f"Loading data from {input_csv}...")
    # โหลดเฉพาะ Column ที่จำเป็นเพื่อประหยัด RAM
    required_cols = ['Title', 'Genre', 'Release Year', 'Director', 'Cast', 'Plot']
    df = pd.read_csv(input_csv, usecols=required_cols)
    
    # เคลียร์ค่าว่าง
    df.fillna("Unknown", inplace=True)
    
    # คลีนเนื้อเรื่อง
    df['Plot'] = df['Plot'].apply(clean_text)
    
    # กรอง Plot ที่สั้นเกินไปออก (มักเป็นข้อมูลที่ไม่สมบูรณ์)
    df = df[df['Plot'].str.len() > 100]
    
    # ใช้ TokenTextSplitter (300 tokens กำลังดีสำหรับ MiniLM)
    # เพิ่ม Overlap เพื่อให้เนื้อหาระหว่าง Chunk ไม่ขาดตอน
    text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
    
    chunks = []
    chunk_id_counter = 0
    
    for _, row in df.iterrows():
        title = str(row['Title'])
        genre = str(row['Genre'])
        year = str(row['Release Year'])
        director = str(row['Director'])
        cast = str(row['Cast'])
        plot = str(row['Plot'])
        
        splits = text_splitter.split_text(plot)
        total_splits = len(splits)
        
        for i, split in enumerate(splits):
            # --- ปรับปรุง Metadata Injection ให้มีความเป็นภาษาธรรมชาติมากขึ้น ---
            # การใส่ "Part X of Y" ช่วยให้ Vector รู้ลำดับของเนื้อหาในเรื่องเดียวกัน
            header = f"Movie: {title} ({year}). Genre: {genre}. Director: {director}. "
            if total_splits > 1:
                header += f"(Part {i+1}/{total_splits}) "
            
            injected_text = f"{header}Content: {split}"
            
            chunks.append({
                "chunk_id": chunk_id_counter,
                "text": injected_text,
                "metadata": {
                    "title": title,
                    "genre": genre,
                    "year": year,
                    "director": director,
                    "cast": cast,
                    "chunk_index": i,
                    "total_chunks": total_splits
                }
            })
            chunk_id_counter += 1
            
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False) # ensure_ascii=False เพื่อให้อ่านไทย/สัญลักษณ์ได้ชัดเจน
        
    print(f"✅ Data processing complete. Saved {len(chunks)} chunks to {output_json}")

if __name__ == "__main__":
    # ใช้ Path แบบ Dynamic หรือตรวจสอบให้แน่ใจว่า Path ถูกต้อง
    input_file = r"D:\llm\llm-project\wiki_movie_plots_deduped\wiki_movie_plots_deduped.csv"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, "data", "chunked_plots.json")
    process_and_chunk_data(input_file, output_file)