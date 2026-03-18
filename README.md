
---

## ⚙️ Technologies Used

- **LangChain**
- **FAISS** – Vector Search
- **BM25** – Keyword Search
- **Sentence Transformers**
- **HuggingFace Embeddings**
- **LLM (Local / API)**

---

## 🔍 How It Works

### 1. Query Processing
- รับคำถามจากผู้ใช้
- ตรวจจับภาษา
- แปลเป็นภาษาอังกฤษ 

### 2. Embedding
- ใช้โมเดลเช่น:
  - `all-MiniLM-L6-v2`
- แปลงข้อความ → vector

### 3. Retrieval (Hybrid Search)
- FAISS → ค้นหาความหมาย (semantic)
- BM25 → ค้นหาคำสำคัญ (keyword)
- รวมผลลัพธ์ทั้งสองแบบ

### 4. Re-ranking
- ใช้ Cross-Encoder จัดอันดับใหม่
- เลือก Top-K ที่เกี่ยวข้องที่สุด

### 5. Answer Generation
- ส่ง context ให้ LLM
- สร้างคำตอบที่แม่นยำ

---

## 🧪 Example Queries

- "สรุปหนัง Titanic ให้หน่อย"
- "movie about revenge"
- "หนังที่เกี่ยวกับเรือจม"

---

## 📈 Improvements

- เปลี่ยน embedding เป็น `e5-base`
- เพิ่ม Re-ranking
- ใช้ Multilingual model
- ปรับ chunking ให้เหมาะสม

---


## 🛠 Installation

```bash
git clone https://github.com/chetodthief/llm-project.git
cd llm-project
pip install -r requirements.txt
