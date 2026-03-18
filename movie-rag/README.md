# Movie Plot Question Answering System (RAG)

This project builds a simple yet complete Retrieval-Augmented Generation (RAG) system utilizing AI to answer questions dynamically based strictly on a movie plot dataset.

## Architecture

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: `faiss-cpu` natively integrated through LangChain.
- **LLM Engine**: `mistralai/Mistral-7B-Instruct-v0.1` locally run using `transformers` pipelines.
- **Frontend App Environment**: `streamlit`

## Setup Instructions

**Important**: This setup employs local 7B models utilizing the native `transformers` pipeline. This will download a very large 14GB+ Mistral-7B weights checkpoint.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Make sure to also have bitsandbytes/accelerate installed if you encounter VRAM issues or simply want faster load times!)*

### 2. Pre-process and Chunk Data
The raw data is stored in `data/movie_plots.csv`. We will text clean it and split it into 300 token chunks retaining their metadata (Title, Year, Genre).

```bash
python preprocessing/chunk_data.py
```

### 3. Build Vector Index
Translate all created plot text chunks into high dimensional vector floats and store them into FAISS Index using HuggingFace sentence transformer representations.

```bash
python embeddings/build_index.py
```

### 4. Open the App UI
Boot up the Streamlit server and begin testing questions.

```bash
streamlit run app/streamlit_app.py
```

### Example Questions to Evaluate
- *Who is the main character in Interstellar?*
- *What happens at the end of Titanic?*
- *What is the story of The Matrix?*

### Outputs Shown
The Streamlit application will list the exact **movie contexts** used to derive the generated answers, showing its matching **Similarity Score**, **Title**, **Genre**, and **Year**.
