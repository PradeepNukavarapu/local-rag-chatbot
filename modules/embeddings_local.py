#!/usr/bin/env python3
"""
Local Embeddings Handler using sentence-transformers
Generates vector embeddings for text chunks and queries
"""

from sentence_transformers import SentenceTransformer
import streamlit as st

# Load model once (cached)
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached for performance)"""
    try:
        # Using a small, fast model that works well for RAG
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

def generate_embeddings_for_chunk(chunk_text):
    """Generate embeddings for a text chunk"""
    try:
        model = load_embedding_model()
        if model is None:
            return None
        
        # Generate embedding (returns numpy array, convert to list)
        embedding = model.encode(chunk_text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embeddings for chunk: {e}")
        return None

def generate_embeddings_for_query(query_text):
    """Generate embeddings for a search query"""
    try:
        model = load_embedding_model()
        if model is None:
            return None
        
        # Generate embedding
        embedding = model.encode(query_text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embeddings for query: {e}")
        return None