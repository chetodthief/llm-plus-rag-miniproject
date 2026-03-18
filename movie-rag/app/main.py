import os
import sys
import asyncio
import json
import re
import httpx
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Add the project root to the python path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.movie_qa import load_vectorstore, load_llm, answer_question

# Initialize RAG models on startup (only once)
print("Loading AI Models and Vector Database...")
index_dir = os.path.join(project_root, "data", "faiss_index")
vectorstore = load_vectorstore(index_dir)
llm = load_llm()
print("RAG System Ready!")

# --- Pre-load movie list for filter dropdowns (from chunked JSON) ---
_movie_list_cache = None

def get_movie_list():
    global _movie_list_cache
    if _movie_list_cache is not None:
        return _movie_list_cache
    try:
        json_path = os.path.join(project_root, "data", "chunked_plots.json")
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        seen = set()
        movies = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            title = meta.get("title", "")
            year = meta.get("year", "")
            key = f"{title}||{year}"
            if title and key not in seen:
                seen.add(key)
                movies.append({"title": title, "year": year})
        # Sort by title
        movies.sort(key=lambda x: x["title"].lower())
        _movie_list_cache = movies
        return movies
    except Exception as e:
        print(f"Warning: Could not load movie list: {e}")
        return []

app = FastAPI(title="CineRAG Movie API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# Helper: fetch movie poster from TMDB
# =============================================
# =============================================
# Helper: fetch TMDB info (Poster + Genre)
# =============================================
TMDB_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

async def fetch_tmdb_info(title: str, year: str):
    if not TMDB_API_KEY:
        return None
    
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": str(year)[:4] if year and year != "Unknown" else ""
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=5.0)
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                first_result = data["results"][0]
                poster_path = first_result.get("poster_path")
                genre_ids = first_result.get("genre_ids", [])
                genres = [TMDB_GENRES.get(gid) for gid in genre_ids if TMDB_GENRES.get(gid)]
                
                if poster_path:
                    return {
                        "tmdb_id": first_result.get("id"),
                        "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}",
                        "genre": genres[0] if genres else "Various"
                    }
        except Exception as e:
            print(f"Error fetching tmdb info for {title}: {e}")
    return None

# =============================================
# Helper: extract movie titles mentioned in LLM answer
# =============================================
def extract_mentioned_titles(answer_text: str, sources: list) -> list:
    """
    Match source movies that are explicitly mentioned in the LLM's answer text.
    Returns sources sorted by whether they appear in the answer.
    """
    answer_lower = answer_text.lower()
    mentioned = []
    not_mentioned = []
    for src in sources:
        title_lower = src["title"].lower()
        if title_lower in answer_lower:
            mentioned.append(src)
        else:
            not_mentioned.append(src)
    # Return mentioned-first, then non-mentioned
    return mentioned + not_mentioned

# =============================================
# API Models
# =============================================
class QuestionRequest(BaseModel):
    question: str
    filter_year: Optional[str] = None
    filter_title: Optional[str] = None

# =============================================
# Endpoints
# =============================================

@app.get("/movies")
def list_movies(search: str = ""):
    """Returns a searchable list of all movies for the filter dropdown."""
    movies = get_movie_list()
    if search:
        search_lower = search.lower()
        # No limit when searching — return ALL matching results
        movies = [m for m in movies if search_lower in m["title"].lower()]
        return movies[:500]  # generous limit for search results
    return movies[:200]  # limit initial load to 200 for performance

@app.get("/years")
def list_years():
    """Returns a sorted list of unique release years."""
    movies = get_movie_list()
    years = sorted(set(
        m["year"][:4] for m in movies 
        if m.get("year") and m["year"] != "Unknown" and m["year"][:4].isdigit()
    ), reverse=True)
    return years

@app.post("/ask")
async def ask_movie_question(request: QuestionRequest):
    try:
        # 1. RAG: retrieve and generate answer (with optional filters)
        result = answer_question(
            request.question, 
            vectorstore, 
            llm,
            filter_year=request.filter_year,
            filter_title=request.filter_title
        )
        raw_sources = result["sources"]
        answer_text = result["answer"]
        
        # 2. Deduplicate sources (keep first occurrence per title+year)
        unique_sources = []
        seen_movies = set()
        for src in raw_sources:
            movie_id = f"{src['title']}_{src.get('year', '')}"
            if movie_id not in seen_movies:
                seen_movies.add(movie_id)
                unique_sources.append(src)

        # 3. Re-order: put movies actually MENTIONED in the answer first
        ordered_sources = extract_mentioned_titles(answer_text, unique_sources)
        
        # 4. Fetch posters concurrently for all unique sources
        tasks = [fetch_tmdb_info(src["title"], src.get("year", "")) for src in ordered_sources]
        tmdb_infos = await asyncio.gather(*tasks)
        
        for src, info in zip(ordered_sources, tmdb_infos):
            src["poster_url"] = info["poster_url"] if info else "https://placehold.co/300x450/1e293b/94a3b8?text=No+Poster"
            src["mentioned_in_answer"] = src["title"].lower() in answer_text.lower()
            
        # 5. BONUS: Fetch posters for off-database movies (Internal LLM Knowledge)
        # Find capitalized phrases that might be movie titles (e.g., "John Wick")
        import re
        
        # Stopwords: common capitalized words that aren't usually movie titles by themselves
        ignore_words = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or", "With", "As", "By", "From", "About", "Into", "Through", "After", "Over", "Between", "Out", "Against", "During", "Without", "Before", "Under", "Around", "Among", "Rose", "Jack", "Heart", "Ocean", "Dawson", "Bukater", "Lovett", "Titanic", "Movie", "Film", "Director", "Actor", "Role"}
        
        potential_titles = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', answer_text)
        
        filtered_titles = []
        for t in set(potential_titles):
            words = t.split()
            # Ignore single words that are in our ignore list, or are single characters
            if len(words) == 1 and (t in ignore_words or len(t) <= 2):
                continue
            # Ignore if it's already in the source list
            if t.lower() in [s["title"].lower() for s in ordered_sources]:
                continue
            if 0 < len(words) <= 4:
                filtered_titles.append(t)
        
        if filtered_titles:
            # Ping TMDB to check if these are real movies
            tmdb_tasks = [fetch_tmdb_info(title, "") for title in filtered_titles]
            tmdb_results = await asyncio.gather(*tmdb_tasks)
            
            for title, info in zip(filtered_titles, tmdb_results):
                if info: # If TMDB confirms it's a real movie and has a poster
                    ordered_sources.append({
                        "title": title,
                        "year": "?",
                        "genre": info["genre"],
                        "director": "Unknown",
                        "cast": "Unknown",
                        "score": "N/A",
                        "poster_url": info["poster_url"],
                        "mentioned_in_answer": True,
                        "content": "Additional movie suggested by AI."
                    })

        # 5.5 RELATED MOVIES FROM RAG: ใช้หนังที่ RAG ค้นเจอในฐานข้อมูล แต่อาจจะไม่ถูกดึงขึ้นมาเป็น source หลัก
        related_movies = result.get("related_movies", [])
        existing_titles_lower = {s["title"].lower() for s in ordered_sources}
        
        added_similar = []
        for sim in related_movies:
            if len(added_similar) >= 3:
                break
            if sim["title"].lower() not in existing_titles_lower:
                new_src = {
                    "title": sim["title"],
                    "year": sim["year"],
                    "genre": sim["genre"],
                    "director": "Unknown",
                    "cast": "Unknown",
                    "score": "N/A",
                    "poster_url": None,
                    "mentioned_in_answer": False,
                    "is_similar": True,
                    "content": "Related movie found contextually in the RAG database."
                }
                ordered_sources.append(new_src)
                added_similar.append(new_src)
                existing_titles_lower.add(sim["title"].lower())
                print(f"[Similar RAG] Added: {sim['title']} ({sim['year']})")

        # Fetch TMDB posters only for the newly added similar RAG movies
        if added_similar:
            sim_tasks = [fetch_tmdb_info(s["title"], s.get("year", "")) for s in added_similar]
            sim_infos = await asyncio.gather(*sim_tasks)
            for s, info in zip(added_similar, sim_infos):
                s["poster_url"] = info["poster_url"] if info else "https://placehold.co/300x450/1e293b/94a3b8?text=No+Poster"

        # 6. Final Sorting: หนังที่ถูก filter → หนังที่ถูกพูดถึง → หนังอื่น → similar
        def sort_priority(src):
            title_lower = src["title"].lower()
            is_exact_filter = 1 if (request.filter_title and title_lower == request.filter_title.lower()) else 0
            is_mentioned = 1 if src.get("mentioned_in_answer") else 0
            is_similar = 1 if src.get("is_similar") else 0   # similar movies ไว้ท้ายสุด
            return (-is_exact_filter, -is_mentioned, is_similar)

        ordered_sources.sort(key=sort_priority)
            
        return {
            "answer": answer_text,
            "sources": ordered_sources
        }
    except Exception as e:
        print(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))