#!/usr/bin/env python3
"""
Local RAG Chatbot - Main Streamlit Application
Simple chatbot for document Q&A using local resources
"""

import streamlit as st
from modules.document_processor import (
    validate_document_requirements,
    extract_text_from_document,
    hybrid_chunking
)
from modules.embeddings_local import (
    generate_embeddings_for_chunk,
    generate_embeddings_for_query
)
from modules.database_local import (
    store_document_chunks,
    search_similar_chunks,
    get_all_documents,
    delete_document
)
from modules.llm_local import generate_rag_response, test_ollama_connection

# Page configuration
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🤖 Local RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Check Ollama connection
    if not test_ollama_connection():
        st.error("⚠️ Ollama is not running! Please start Ollama first.")
        st.info("To start Ollama, open a new terminal and run: `ollama serve`")
        st.stop()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["📄 Upload Documents", "💬 Chat Interface"])
    
    if page == "📄 Upload Documents":
        upload_page()
    elif page == "💬 Chat Interface":
        chat_page()

def upload_page():
    """Document upload and processing page"""
    st.markdown('<h2 class="sub-header">Upload Documents</h2>', unsafe_allow_html=True)
    
    st.info("Upload PDF or DOCX files to add them to the knowledge base. The documents will be processed and made searchable.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            process_documents(uploaded_files)
    
    # Show uploaded documents
    st.markdown("---")
    st.markdown('<h3 class="sub-header">Uploaded Documents</h3>', unsafe_allow_html=True)
    
    documents = get_all_documents()
    if documents:
        for doc_name in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc_name}")
            with col2:
                if st.button("Delete", key=f"delete_{doc_name}"):
                    if delete_document(doc_name):
                        st.success(f"Deleted {doc_name}")
                        st.rerun()
    else:
        st.info("No documents uploaded yet.")

def process_documents(uploaded_files):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((idx) / len(uploaded_files))
        
        try:
            # Step 1: Validate
            status_text.text(f"Validating {uploaded_file.name}...")
            is_valid, message = validate_document_requirements(uploaded_file)
            if not is_valid:
                st.error(f"❌ {uploaded_file.name}: {message}")
                continue
            
            st.success(f"✅ {uploaded_file.name}: {message}")
            
            # Step 2: Extract text
            status_text.text(f"Extracting text from {uploaded_file.name}...")
            pages_data = extract_text_from_document(uploaded_file)
            if not pages_data:
                st.error(f"❌ Failed to extract text from {uploaded_file.name}")
                continue
            
            # Step 3: Create chunks
            status_text.text(f"Creating chunks for {uploaded_file.name}...")
            all_chunks = []
            global_chunk_index = 0  # Keep track across all pages
            for page_data in pages_data:
                chunks = hybrid_chunking(page_data['text'])
                for chunk_text in chunks:
                    all_chunks.append({
                        'chunk_text': chunk_text,
                        'chunk_index': global_chunk_index,  # Unique across all pages
                        'page_number': page_data['page_number']
                    })
                    global_chunk_index += 1  # Increment for next chunk
            
            # Step 4: Generate embeddings
            status_text.text(f"Generating embeddings for {uploaded_file.name}...")
            chunks_with_embeddings = []
            total_chunks = len(all_chunks)
            
            # Show progress for embeddings
            embedding_progress = st.progress(0)
            for i, chunk in enumerate(all_chunks):
                embedding = generate_embeddings_for_chunk(chunk['chunk_text'])
                if embedding:
                    chunks_with_embeddings.append({
                        **chunk,
                        'embeddings': embedding
                    })
                embedding_progress.progress((i + 1) / total_chunks)
            
            embedding_progress.empty()
            
            # Step 5: Store in database
            status_text.text(f"Storing {uploaded_file.name} in database...")
            doc_name = uploaded_file.name.rsplit('.', 1)[0]  # Remove extension
            if store_document_chunks(doc_name, chunks_with_embeddings):
                st.success(f"✅ Successfully processed {uploaded_file.name} ({len(chunks_with_embeddings)} chunks)")
            else:
                st.error(f"❌ Failed to store {uploaded_file.name}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
    
    status_text.text("Processing complete!")
    progress_bar.empty()
    if len(uploaded_files) > 0:
        st.balloons()

def chat_page():
    """Chat interface page"""
    st.markdown('<h2 class="sub-header">Chat Interface</h2>', unsafe_allow_html=True)
    
    # Check if documents exist
    documents = get_all_documents()
    if not documents:
        st.warning("⚠️ No documents uploaded yet. Please upload documents first.")
        return
    
    st.info(f"📚 You have {len(documents)} document(s) in the knowledge base. Ask questions about them!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Validate the query
        prompt_lower = prompt.lower().strip()
        
        # Check if it's a valid question
        question_words = ['what', 'how', 'where', 'when', 'why', 'who', 'which', 'explain', 'describe', 'tell me', 'show me']
        is_question = any(prompt_lower.startswith(word) for word in question_words) or '?' in prompt
        
        # Check for feedback/comments (not questions)
        feedback_phrases = [
            'not complete', 'incomplete', 'wrong', 'incorrect', 'not right', 
            'doesn\'t work', 'not working', 'error', 'fix', 'help', 'answer is not'
        ]
        is_feedback = any(phrase in prompt_lower for phrase in feedback_phrases) and not is_question
        
        if is_feedback:
            st.warning("⚠️ It looks like you're providing feedback rather than asking a question.")
            st.info("💡 **Please ask a specific question instead:**")
            st.write("- 'What are the complete steps to create the user?'")
            st.write("- 'Can you provide the full SQL commands?'")
            st.write("- 'What are all the steps from the document?'")
            # Don't add this to chat history
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query expansion for better retrieval
                expanded_query = prompt
                query_hints = []
                
                # Expand query based on question type
                if any(word in prompt.lower() for word in ["where", "how to access", "navigate", "go to", "find"]):
                    query_hints.extend([
                        "menu navigation", "how to access", "where to find",
                        "location in menu", "screen navigation", "menu path",
                        "access path", "where to navigate"
                    ])
                
                # Database/user creation specific expansion
                if any(word in prompt.lower() for word in [
                    "create user", "create schema", "database user", "privileged user", 
                    "minimally privileged", "schema creation", "database schema",
                    "SQL user", "grant privileges", "create database user"
                ]):
                    query_hints.extend([
                        "database schema", "SQL create user", "grant privileges",
                        "database user creation", "SQL commands", "SYSDBA",
                        "create user identified by", "default tablespace",
                        "grant create view", "connect to database", "SQL*Plus", "database server"
                    ])
                
                # Connection/database access specific expansion
                if any(word in prompt.lower() for word in [
                    "connect to", "database connection", "database server", 
                    "how to connect", "database access", "SQL connection"
                ]):
                    query_hints.extend([
                        "SQL*Plus", "database connection", "connect as SYS",
                        "SYSDBA role", "connection string", "database server", "SQL connection"
                    ])
                
                # General "about" questions - add overview/summary keywords
                if any(phrase in prompt.lower() for phrase in ["what is this", "what is the", "what does this", "what is it about", "about this", "tell me about"]):
                    query_hints.extend([
                        "overview", "summary", "introduction", "executive summary", 
                        "document purpose", "what is", "description", "about",
                        "white paper", "document content", "main topic", "subject"
                    ])
                
                if any(word in prompt.lower() for word in ["what", "explain", "describe", "tell me about"]):
                    query_hints.append("information details explanation")
                
                if any(word in prompt.lower() for word in ["how", "steps", "process", "procedure"]):
                    query_hints.append("steps procedure process method")
                
                if query_hints:
                    expanded_query = f"{prompt} {' '.join(query_hints)}"
                
                # Step 1: Generate query embeddings
                query_embedding = generate_embeddings_for_query(expanded_query)
                original_embedding = generate_embeddings_for_query(prompt)
                
                if not query_embedding:
                    st.error("Failed to generate query embedding")
                    return
                
                # Step 2: Search similar chunks
                chunks_expanded = search_similar_chunks(query_embedding, top_k=30)
                chunks_original = search_similar_chunks(original_embedding, top_k=30)
                
                # Combine and deduplicate
                all_chunks = chunks_expanded + chunks_original
                seen_ids = set()
                unique_chunks = []
                for chunk in all_chunks:
                    chunk_id = chunk.get('chunk_id', '')
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        unique_chunks.append(chunk)
                
                # Sort by distance
                similar_chunks = sorted(unique_chunks, key=lambda x: x.get('distance', 1.0))
                
                # Quality check - filter out poor matches with adaptive threshold
                if similar_chunks:
                    # Use adaptive threshold: start with 1.2, but if no matches, use best available (up to 1.8)
                    primary_threshold = 1.2
                    fallback_threshold = 1.8
                    
                    good_chunks = [chunk for chunk in similar_chunks if chunk.get('distance', 1.0) < primary_threshold]
                    
                    # If no chunks meet primary threshold, use fallback for general questions
                    if not good_chunks:
                        # Check if this is a general "about" question - use best available chunks
                        is_general_question = any(phrase in prompt.lower() for phrase in [
                            "what is this", "what is the", "what does this", 
                            "what is it about", "about this", "tell me about",
                            "what is", "describe this", "explain this"
                        ])
                        
                        if is_general_question:
                            # For general questions, use best available chunks (up to fallback threshold)
                            good_chunks = [chunk for chunk in similar_chunks if chunk.get('distance', 1.0) < fallback_threshold]
                            if good_chunks:
                                st.info("ℹ️ Using best available matches for your general question.")
                        else:
                            # For specific questions, show warning but still use best chunks
                            good_chunks = [chunk for chunk in similar_chunks if chunk.get('distance', 1.0) < fallback_threshold]
                            if not good_chunks:
                                # All chunks have very poor similarity
                                st.warning("⚠️ I couldn't find highly relevant information for your question.")
                                st.info("💡 **Suggestions:**")
                                st.write("- Try rephrasing your question")
                                st.write("- Ask a more specific question")
                                st.write("- Use different keywords related to your topic")
                                
                                # Show debug info
                                with st.expander("🔍 Debug: Why no good matches?"):
                                    st.write(f"**All {len(similar_chunks)} retrieved chunks have high similarity scores (poor matches):**")
                                    for i, chunk in enumerate(similar_chunks[:5], 1):
                                        st.write(f"**Chunk {i}:** Score {chunk.get('distance', 0):.4f} - {chunk.get('doc_name', 'Unknown')} Page {chunk.get('page_number', 0)}")
                                        st.write(f"Preview: {chunk.get('chunk_text', '')[:200]}...")
                                return
                    
                    # Filter out chunks that are too short
                    filtered_chunks = [chunk for chunk in good_chunks if len(chunk.get('chunk_text', '')) > 50]
                    similar_chunks = filtered_chunks[:15]
                else:
                    similar_chunks = []
                
                # 🔍 DEBUG: Show retrieved chunks
                if similar_chunks:
                    with st.expander("🔍 Debug: Retrieved Chunks (click to see)"):
                        st.write(f"**Found {len(similar_chunks)} relevant chunk(s):**")
                        st.write(f"**Original query:** {prompt}")
                        if expanded_query != prompt:
                            st.write(f"**Expanded query:** {expanded_query}")
                        st.write(f"**Search strategy:** Combined original + expanded query results (top 30 each)")
                        st.write("---")
                        for i, chunk in enumerate(similar_chunks, 1):
                            st.write(f"**Chunk {i}:** {chunk.get('doc_name', 'Unknown')} - Page {chunk.get('page_number', 0)}")
                            st.write(f"**Text preview:** {chunk.get('chunk_text', '')[:400]}...")
                            if 'distance' in chunk:
                                st.write(f"**Similarity score:** {chunk.get('distance', 0):.4f} (lower is better)")
                            st.write("---")
                
                # Check if we have relevant chunks
                if not similar_chunks:
                    response = "I couldn't find relevant information in the documents for your question. Please try rephrasing or ask about a different topic."
                else:
                    # Ensure chunks are sorted by similarity (most relevant first)
                    similar_chunks = sorted(similar_chunks, key=lambda x: x.get('distance', 1.0))
                    
                    # Step 3: Generate AI response
                    response = generate_rag_response(prompt, similar_chunks)
                
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to generate response")

if __name__ == "__main__":
    main()