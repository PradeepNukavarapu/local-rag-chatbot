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
from modules.web_scraper import (
    validate_url,
    scrape_url_to_pages
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
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* User Message */
    [data-testid="stChatMessage"] [data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin-left: auto;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAssistant"] {
        background: #f0f2f6;
        color: #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: auto;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e5e7eb;
        z-index: 100;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Info/Warning Boxes */
    .stInfo {
        background: #e0f2fe;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stSuccess {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Code Blocks */
    pre {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        overflow-x: auto;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Document List Styling */
    .document-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .document-item:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-color: #667eea transparent #667eea transparent;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    
    # Sidebar for navigation with better styling
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["📄 Upload Documents", "💬 Chat Interface"],
        label_visibility="collapsed"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "**Local RAG Chatbot**\n\n"
        "Upload documents/URLs and ask questions using local AI models. "
        "All processing happens on your machine - your data stays private."
    )
    
    if page == "📄 Upload Documents":
        upload_page()
    elif page == "💬 Chat Interface":
        chat_page()

def upload_page():
    """Document upload and processing page"""
    st.markdown('<h2 class="sub-header">📄 Add Content to Knowledge Base</h2>', unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Files", "🌐 Add from URL"])
    
    with tab1:
        st.info("📤 **Upload PDF or DOCX files** to add them to the knowledge base. The documents will be processed, chunked, and made searchable.")
        
        # File uploader with better styling
        uploaded_files = st.file_uploader(
            "Choose PDF or DOCX files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="You can upload multiple files at once. Supported formats: PDF and DOCX."
        )
        
        if uploaded_files:
            st.success(f"✅ **{len(uploaded_files)} file(s)** selected and ready to process.")
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🚀 Process Documents", type="primary", use_container_width=True, key="process_files"):
                    process_documents(uploaded_files)
            with col2:
                st.caption("Click to start processing. This may take a few moments depending on file size.")
    
    with tab2:
        st.info("🌐 **Enter URL(s)** to scrape content from webpage(s) and add them to your knowledge base. You can enter multiple URLs (one per line).")
        
        # URL input - text area for multiple URLs
        url_input = st.text_area(
            "Enter URL(s) - one per line",
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3",
            help="Enter one or more URLs, each on a new line. URLs should start with http:// or https://",
            height=150
        )
        
        if url_input:
            # Parse URLs (split by newline and filter empty lines)
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            if urls:
                st.info(f"📋 **{len(urls)} URL(s)** entered")
                
                # Validate all URLs
                valid_urls = []
                invalid_urls = []
                
                with st.expander(f"🔍 Validate {len(urls)} URL(s)", expanded=False):
                    for idx, url in enumerate(urls, 1):
                        is_valid, message = validate_url(url)
                        if is_valid:
                            st.success(f"✅ URL {idx}: {url}")
                            valid_urls.append(url)
                        else:
                            st.warning(f"⚠️ URL {idx}: {url} - {message}")
                            invalid_urls.append((url, message))
                
                # Show summary
                if invalid_urls:
                    st.warning(f"⚠️ **{len(invalid_urls)} URL(s)** failed validation. Only valid URLs will be processed.")
                
                if valid_urls:
                    st.success(f"✅ **{len(valid_urls)} URL(s)** are valid and ready to process.")
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🚀 Scrape & Process URLs", type="primary", use_container_width=True, key="process_urls"):
                            process_urls(valid_urls)
                    with col2:
                        st.caption(f"Click to scrape and process {len(valid_urls)} URL(s).")
                elif urls:
                    st.error("❌ No valid URLs to process. Please fix the URLs above.")
            else:
                st.warning("⚠️ Please enter at least one URL.")
    
    # Show uploaded documents with better styling
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📚 Your Documents</h3>', unsafe_allow_html=True)
    
    documents = get_all_documents()
    if documents:
        st.caption(f"Total: {len(documents)} document(s) in your knowledge base")
        for doc_name in documents:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="document-item">
                        <strong>📄 {doc_name}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{doc_name}", use_container_width=True):
                        if delete_document(doc_name):
                            st.success(f"✅ Deleted **{doc_name}**")
                            st.rerun()
    else:
        st.info("📭 **No documents uploaded yet.** Upload your first document above to get started!")

def process_documents(uploaded_files):
    """Process uploaded documents with improved error handling and feedback"""
    if not uploaded_files:
        st.warning("⚠️ No files selected. Please select files to process.")
        return
    
    # Create a container for processing status
    status_container = st.container()
    
    with status_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        file_status = {}  # Track status of each file
    
    successful_files = 0
    failed_files = 0
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_key = f"{uploaded_file.name}_{idx}"
        status_text.text(f"📄 Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")
        progress_bar.progress((idx) / len(uploaded_files))
        
        try:
            # Step 1: Validate
            status_text.text(f"🔍 Validating {uploaded_file.name}...")
            is_valid, message = validate_document_requirements(uploaded_file)
            if not is_valid:
                st.error(f"❌ **{uploaded_file.name}**: {message}")
                file_status[file_key] = {"status": "failed", "message": message}
                failed_files += 1
                continue
            
            # Step 2: Extract text
            status_text.text(f"📖 Extracting text from {uploaded_file.name}...")
            pages_data = extract_text_from_document(uploaded_file)
            if not pages_data:
                st.error(f"❌ **{uploaded_file.name}**: Failed to extract text")
                file_status[file_key] = {"status": "failed", "message": "Text extraction failed"}
                failed_files += 1
                continue
            
            st.info(f"📄 Extracted {len(pages_data)} page(s) from {uploaded_file.name}")
            
            # Step 3: Create chunks
            status_text.text(f"✂️ Creating chunks for {uploaded_file.name}...")
            all_chunks = []
            global_chunk_index = 0
            for page_data in pages_data:
                chunks = hybrid_chunking(page_data['text'])
                for chunk_text in chunks:
                    all_chunks.append({
                        'chunk_text': chunk_text,
                        'chunk_index': global_chunk_index,
                        'page_number': page_data['page_number']
                    })
                    global_chunk_index += 1
            
            if not all_chunks:
                st.warning(f"⚠️ **{uploaded_file.name}**: No chunks created. File may be too short.")
                file_status[file_key] = {"status": "failed", "message": "No chunks created"}
                failed_files += 1
                continue
            
            st.info(f"📝 Created {len(all_chunks)} chunk(s) from {uploaded_file.name}")
            
            # Step 4: Generate embeddings
            status_text.text(f"🧮 Generating embeddings for {uploaded_file.name}...")
            chunks_with_embeddings = []
            total_chunks = len(all_chunks)
            
            # Show progress for embeddings with better feedback
            embedding_progress = st.progress(0)
            embedding_status = st.empty()
            for i, chunk in enumerate(all_chunks):
                embedding_status.text(f"Generating embedding {i + 1}/{total_chunks}...")
                embedding = generate_embeddings_for_chunk(chunk['chunk_text'])
                if embedding:
                    chunks_with_embeddings.append({
                        **chunk,
                        'embeddings': embedding
                    })
                embedding_progress.progress((i + 1) / total_chunks)
            
            embedding_progress.empty()
            embedding_status.empty()
            
            if not chunks_with_embeddings:
                st.error(f"❌ **{uploaded_file.name}**: Failed to generate embeddings")
                file_status[file_key] = {"status": "failed", "message": "Embedding generation failed"}
                failed_files += 1
                continue
            
            # Step 5: Store in database
            status_text.text(f"💾 Storing {uploaded_file.name} in database...")
            doc_name = uploaded_file.name.rsplit('.', 1)[0]  # Remove extension
            
            if store_document_chunks(doc_name, chunks_with_embeddings):
                st.success(f"✅ **{uploaded_file.name}**: Successfully processed ({len(chunks_with_embeddings)} chunks)")
                file_status[file_key] = {"status": "success", "chunks": len(chunks_with_embeddings)}
                successful_files += 1
            else:
                st.error(f"❌ **{uploaded_file.name}**: Failed to store in database")
                file_status[file_key] = {"status": "failed", "message": "Database storage failed"}
                failed_files += 1
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"❌ **{uploaded_file.name}**: Error - {str(e)}")
            with st.expander(f"Error Details for {uploaded_file.name}"):
                import traceback
                st.code(traceback.format_exc())
            file_status[file_key] = {"status": "failed", "message": str(e)}
            failed_files += 1
    
    # Final summary
    status_text.text("✅ Processing complete!")
    progress_bar.progress(1.0)
    
    # Show summary
    st.markdown("---")
    if successful_files > 0:
        st.success(f"🎉 **{successful_files} file(s)** processed successfully!")
    if failed_files > 0:
        st.warning(f"⚠️ **{failed_files} file(s)** failed to process")
    
    if successful_files > 0:
        st.balloons()
    
    # Keep progress indicators visible for user to see final status
    # They will be cleared when user interacts with the app

def process_urls(urls):
    """Process multiple URLs by scraping and adding to knowledge base"""
    if not urls:
        st.warning("⚠️ No URLs provided. Please enter URLs to process.")
        return
    
    # Create a container for processing status
    status_container = st.container()
    
    with status_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        url_status = {}  # Track status of each URL
    
    successful_urls = 0
    failed_urls = 0
    
    for idx, url in enumerate(urls):
        url_key = f"{url}_{idx}"
        status_text.text(f"🌐 Processing {idx + 1}/{len(urls)}: {url[:60]}...")
        progress_bar.progress((idx) / len(urls))
        
        try:
            # Step 1: Scrape content (validation already done)
            status_text.text(f"📥 Scraping content from {url[:60]}...")
            pages_data = scrape_url_to_pages(url)
            if not pages_data:
                st.error(f"❌ **{url[:60]}...**: Failed to scrape content")
                url_status[url_key] = {"status": "failed", "message": "Scraping failed"}
                failed_urls += 1
                continue
            
            st.info(f"📄 Scraped {len(pages_data)} section(s) from {url[:60]}...")
            
            # Step 2: Create chunks
            status_text.text(f"✂️ Creating chunks from {url[:60]}...")
            all_chunks = []
            global_chunk_index = 0
            for page_data in pages_data:
                chunks = hybrid_chunking(page_data['text'])
                for chunk_text in chunks:
                    all_chunks.append({
                        'chunk_text': chunk_text,
                        'chunk_index': global_chunk_index,
                        'page_number': page_data['page_number']
                    })
                    global_chunk_index += 1
            
            if not all_chunks:
                st.warning(f"⚠️ **{url[:60]}...**: No chunks created. Page may be too short.")
                url_status[url_key] = {"status": "failed", "message": "No chunks created"}
                failed_urls += 1
                continue
            
            st.info(f"📝 Created {len(all_chunks)} chunk(s) from {url[:60]}...")
            
            # Step 3: Generate embeddings
            status_text.text(f"🧮 Generating embeddings for {url[:60]}...")
            chunks_with_embeddings = []
            total_chunks = len(all_chunks)
            
            # Show progress for embeddings
            embedding_progress = st.progress(0)
            embedding_status = st.empty()
            for i, chunk in enumerate(all_chunks):
                embedding_status.text(f"Generating embedding {i + 1}/{total_chunks}...")
                embedding = generate_embeddings_for_chunk(chunk['chunk_text'])
                if embedding:
                    chunks_with_embeddings.append({
                        **chunk,
                        'embeddings': embedding
                    })
                embedding_progress.progress((i + 1) / total_chunks)
            
            embedding_progress.empty()
            embedding_status.empty()
            
            if not chunks_with_embeddings:
                st.error(f"❌ **{url[:60]}...**: Failed to generate embeddings")
                url_status[url_key] = {"status": "failed", "message": "Embedding generation failed"}
                failed_urls += 1
                continue
            
            # Step 4: Store in database
            status_text.text(f"💾 Storing {url[:60]}... in database...")
            
            # Create a clean document name from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            doc_name = f"{parsed_url.netloc.replace('www.', '')}_{parsed_url.path.replace('/', '_').replace('.', '_')[:50]}"
            if not doc_name or doc_name == '_':
                doc_name = f"webpage_{url[:50].replace('://', '_').replace('/', '_')}"
            
            if store_document_chunks(doc_name, chunks_with_embeddings):
                st.success(f"✅ **{url[:60]}...**: Successfully processed ({len(chunks_with_embeddings)} chunks)")
                url_status[url_key] = {"status": "success", "chunks": len(chunks_with_embeddings)}
                successful_urls += 1
            else:
                st.error(f"❌ **{url[:60]}...**: Failed to store in database")
                url_status[url_key] = {"status": "failed", "message": "Database storage failed"}
                failed_urls += 1
            
            progress_bar.progress((idx + 1) / len(urls))
            
        except Exception as e:
            st.error(f"❌ **{url[:60]}...**: Error - {str(e)}")
            with st.expander(f"Error Details for {url[:60]}..."):
                import traceback
                st.code(traceback.format_exc())
            url_status[url_key] = {"status": "failed", "message": str(e)}
            failed_urls += 1
    
    # Final summary
    status_text.text("✅ Processing complete!")
    progress_bar.progress(1.0)
    
    # Show summary
    st.markdown("---")
    if successful_urls > 0:
        st.success(f"🎉 **{successful_urls} URL(s)** processed successfully!")
    if failed_urls > 0:
        st.warning(f"⚠️ **{failed_urls} URL(s)** failed to process")
    
    if successful_urls > 0:
        st.balloons()
    
    # Keep progress indicators visible for user to see final status

def process_url(url):
    """Process single URL by scraping and adding to knowledge base (legacy function for backward compatibility)"""
    process_urls([url])

def chat_page():
    """Chat interface page"""
    st.markdown('<h2 class="sub-header">💬 Chat Interface</h2>', unsafe_allow_html=True)
    
    # Check if documents exist
    documents = get_all_documents()
    if not documents:
        st.warning("⚠️ **No documents uploaded yet.** Please upload documents first to start chatting.")
        st.info("💡 Go to the **Upload Documents** page in the sidebar to add your documents.")
        return
    
    # Show document count and info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📚 **{len(documents)} document(s)** in your knowledge base. Ask questions about them!")
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Welcome message if chat is empty
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("👋 **Hello! I'm your RAG assistant.**")
            st.markdown("I can help you find information from your uploaded documents. Try asking:")
            st.markdown("- What is this document about?")
            st.markdown("- What are the complete steps to...?")
            st.markdown("- Explain how to...")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with better placeholder
    if prompt := st.chat_input("💬 Ask a question about your documents..."):
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
                # Query expansion for better retrieval with context awareness
                expanded_query = prompt
                query_hints = []
                
                # Extract context from previous messages (last 2-3 messages)
                context_keywords = []
                if len(st.session_state.messages) > 0:
                    # Get recent messages for context
                    recent_messages = st.session_state.messages[-4:]  # Last 4 messages (2 Q&A pairs)
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            # Extract key terms from previous questions
                            words = msg["content"].lower().split()
                            # Filter out common words and keep important terms
                            important_words = [w for w in words if len(w) > 4 and w not in ['what', 'how', 'when', 'where', 'which', 'about', 'will', 'does', 'this', 'that', 'the', 'and', 'for', 'with']]
                            context_keywords.extend(important_words)
                        elif msg["role"] == "assistant":
                            # Extract key terms from previous answers (first 200 chars)
                            answer_text = msg["content"][:200].lower()
                            # Extract acronyms and capitalized terms
                            import re
                            acronyms = re.findall(r'\b[A-Z]{2,}\b', msg["content"][:200])
                            context_keywords.extend([a.lower() for a in acronyms])
                
                # Add context keywords to query if they're relevant
                if context_keywords:
                    # Check if current question mentions terms from context
                    prompt_lower = prompt.lower()
                    relevant_context = [kw for kw in context_keywords if kw in prompt_lower or any(kw in word for word in prompt_lower.split())]
                    if relevant_context:
                        query_hints.extend(relevant_context[:5])  # Add top 5 relevant context terms
                
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
                
                # AIDP/AI Data Platform specific expansion
                if any(term in prompt.lower() for term in ["aidp", "ai data platform", "oracle ai data platform"]):
                    query_hints.extend([
                        "AI Data Platform", "AIDP", "Oracle AI Data Platform", 
                        "AI Data Platform Workbench", "AIDP Units", "OCPU", "memory",
                        "compute cluster", "pricing", "cost", "help", "benefits", "features"
                    ])
                
                # APEX specific expansion (to differentiate from AIDP)
                if any(term in prompt.lower() for term in ["apex", "application express", "oracle apex"]):
                    query_hints.extend([
                        "Oracle APEX", "Application Express", "APEX", 
                        "workspace", "application builder", "page designer"
                    ])
                
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
                
                # Topic-based filtering: Ensure chunks are relevant to the question topic
                prompt_lower = prompt.lower()
                question_keywords = set()
                
                # Extract key terms from the question (excluding common words)
                import re
                stop_words = {'what', 'how', 'when', 'where', 'which', 'who', 'why', 'will', 'does', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about', 'this', 'that', 'these', 'those'}
                question_words = [w.lower() for w in re.findall(r'\b\w+\b', prompt) if len(w) > 3 and w.lower() not in stop_words]
                question_keywords.update(question_words)
                
                # Extract acronyms (like AIDP, APEX, OCI, etc.)
                acronyms = re.findall(r'\b[A-Z]{2,}\b', prompt)
                question_keywords.update([a.lower() for a in acronyms])
                
                # Filter chunks to ensure they contain at least one relevant keyword
                if question_keywords and len(question_keywords) > 0:
                    filtered_chunks = []
                    for chunk in similar_chunks:
                        chunk_text_lower = chunk.get('chunk_text', '').lower()
                        # Check if chunk contains any question keywords
                        keyword_matches = sum(1 for kw in question_keywords if kw in chunk_text_lower)
                        # If it's an acronym, require exact match
                        acronym_matches = sum(1 for ac in acronyms if ac.lower() in chunk_text_lower or ac in chunk.get('chunk_text', ''))
                        
                        # Keep chunk if it has keyword matches or if similarity is very good
                        if keyword_matches > 0 or acronym_matches > 0 or chunk.get('distance', 1.0) < 0.9:
                            filtered_chunks.append(chunk)
                        # Also keep if it's in top 5 and has decent similarity
                        elif len(filtered_chunks) < 5 and chunk.get('distance', 1.0) < 1.0:
                            filtered_chunks.append(chunk)
                    
                    # If filtering removed too many chunks, use original list but prioritize keyword matches
                    if len(filtered_chunks) < 3:
                        # Re-sort with keyword matches as priority
                        for chunk in similar_chunks[:15]:
                            if chunk not in filtered_chunks:
                                chunk_text_lower = chunk.get('chunk_text', '').lower()
                                keyword_matches = sum(1 for kw in question_keywords if kw in chunk_text_lower)
                                if keyword_matches > 0:
                                    filtered_chunks.append(chunk)
                        
                        # If still not enough, add best matches
                        if len(filtered_chunks) < 5:
                            for chunk in similar_chunks:
                                if chunk not in filtered_chunks:
                                    filtered_chunks.append(chunk)
                                    if len(filtered_chunks) >= 10:
                                        break
                    
                    similar_chunks = filtered_chunks[:30]  # Limit to top 30
                
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