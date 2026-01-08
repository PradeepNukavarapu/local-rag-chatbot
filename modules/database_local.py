#!/usr/bin/env python3
"""
Local Database Module using ChromaDB
Handles document storage, chunk storage, and vector search
"""

import chromadb
from chromadb.config import Settings
import streamlit as st
from datetime import datetime
import os
import re
import uuid

# Initialize ChromaDB client (persistent storage)
def get_chroma_client():
    """Get or create ChromaDB client"""
    try:
        # Store database in project folder
        db_path = os.path.join(os.getcwd(), "data", "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=db_path)
        return client
    except Exception as e:
        st.error(f"Error creating ChromaDB client: {e}")
        return None

def get_or_create_collection():
    """Get or create the documents collection"""
    try:
        client = get_chroma_client()
        if client is None:
            return None
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document chunks with embeddings"}
        )
        return collection
    except Exception as e:
        st.error(f"Error getting collection: {e}")
        return None

def store_document_chunks(doc_name, chunks_data):
    """
    Store document chunks in ChromaDB
    
    Args:
        doc_name: Name of the document
        chunks_data: List of dicts with keys: chunk_text, chunk_index, page_number, embeddings
    """
    try:
        collection = get_or_create_collection()
        if collection is None:
            return False
        
        # Sanitize document name
        safe_doc_name = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_name)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        # Generate unique IDs using UUID
        doc_uuid = str(uuid.uuid4())[:8]  # Short unique identifier
        
        for idx, chunk in enumerate(chunks_data):
            # Create unique ID
            chunk_id = f"{safe_doc_name}_{doc_uuid}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk['chunk_text'])
            embeddings.append(chunk['embeddings'])
            metadatas.append({
                'doc_name': doc_name,
                'chunk_index': chunk.get('chunk_index', idx),
                'page_number': chunk.get('page_number', 0),
                'created_at': datetime.now().isoformat()
            })
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return True
    except Exception as e:
        st.error(f"Error storing document chunks: {e}")
        return False

def search_similar_chunks(query_embedding, top_k=5, doc_name_filter=None):
    """
    Search for similar chunks using vector similarity
    
    Args:
        query_embedding: Embedding vector of the query
        top_k: Number of results to return
        doc_name_filter: Optional document name to filter by
    
    Returns:
        List of similar chunks with metadata
    """
    try:
        collection = get_or_create_collection()
        if collection is None:
            return []
        
        # Prepare where clause if filtering by document
        where_clause = None
        if doc_name_filter:
            where_clause = {"doc_name": doc_name_filter}
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'chunk_id': results['ids'][0][i],
                    'chunk_text': results['documents'][0][i],
                    'doc_name': results['metadatas'][0][i].get('doc_name', 'Unknown'),
                    'chunk_index': results['metadatas'][0][i].get('chunk_index', 0),
                    'page_number': results['metadatas'][0][i].get('page_number', 0),
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })
        
        return formatted_results
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

def get_all_documents():
    """Get list of all unique documents in the collection"""
    try:
        collection = get_or_create_collection()
        if collection is None:
            return []
        
        # Get all items
        all_data = collection.get()
        
        # Extract unique document names
        unique_docs = set()
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                if 'doc_name' in metadata:
                    unique_docs.add(metadata['doc_name'])
        
        return sorted(list(unique_docs))
    except Exception as e:
        st.error(f"Error getting documents: {e}")
        return []

def delete_document(doc_name):
    """Delete all chunks for a specific document"""
    try:
        collection = get_or_create_collection()
        if collection is None:
            return False
        
        # Get all chunks for this document
        all_data = collection.get(where={"doc_name": doc_name})
        
        if all_data['ids']:
            # Delete all chunks
            collection.delete(ids=all_data['ids'])
        
        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False