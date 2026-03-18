import streamlit as st
import os
import sys

# Add the project root to the python path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from rag.movie_qa import load_vectorstore, load_llm, answer_question

# Page Configuration
st.set_page_config(page_title="Movie RAG QA System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Plot Question Answering System")
st.markdown("Ask questions about movie plots, and get answers generated from a Retrieval-Augmented Generation (RAG) system.")

# Initialize session state for loading resources
if "resources_loaded" not in st.session_state:
    st.session_state.resources_loaded = False

@st.cache_resource
def get_resources():
    index_dir = os.path.join(project_root, "data", "faiss_index")
    vectorstore = load_vectorstore(index_dir)
    llm = load_llm()
    return vectorstore, llm

# Load resources with a spinner
if not st.session_state.resources_loaded:
    with st.spinner("Loading AI Models and Vector Database (this may take a few minutes if downloading)..."):
        try:
            st.session_state.vectorstore, st.session_state.llm = get_resources()
            st.session_state.resources_loaded = True
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.info("Make sure you have run `preprocessing/chunk_data.py` and `embeddings/build_index.py` first.")
            st.stop()

# Main UI
st.subheader("Ask a Question")
question = st.text_input("Enter your question about a movie:", placeholder="e.g., What happens at the end of Titanic?")

if st.button("Get Answer") and question:
    with st.spinner("Searching movie plots and generating answer..."):
        try:
            result = answer_question(question, st.session_state.vectorstore, st.session_state.llm)
            
            # Display generated answer
            st.success("### Generated Answer")
            st.write(result["answer"])
            
            st.markdown("---")
            
            # Display retrieved context
            st.subheader("📚 Retrieved Context")
            st.markdown("The answer was generated using the following movie plot chunks:")
            
            cols = st.columns(len(result["sources"]))
            
            for i, src in enumerate(result["sources"]):
                with cols[i]:
                    st.info(f"**Movie:** {src['title']} ({src['year']})  \n"
                            f"**Director:** {src.get('director', 'Unknown')}  \n"
                            f"**Cast:** {src.get('cast', 'Unknown')}  \n"
                            f"**Genre:** {src['genre']}  \n"
                            f"**Search Method:** {src['score']}")
                    with st.expander("View Plot Chunk Context"):
                        st.write(src['content'])
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
