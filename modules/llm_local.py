#!/usr/bin/env python3
"""
Local LLM Handler using Ollama
Generates AI responses based on context and user queries
"""

import requests
import streamlit as st

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"  # Using the smaller model

def generate_rag_response(question, context_chunks, max_tokens=1500):
    """
    Generate AI response using Ollama with RAG context
    
    Args:
        question: User's question
        context_chunks: List of relevant document chunks (should be sorted by relevance)
        max_tokens: Maximum tokens in response
    
    Returns:
        AI-generated response string
    """
    try:
        # Sort chunks by similarity score (distance) - lower is better
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('distance', 1.0))
        
        # Use ONLY the top 3 most relevant chunks
        top_chunks = sorted_chunks[:3]
        
        # Prepare context - emphasize the first (most relevant) chunk
        context_text = ""
        
        # First chunk gets special emphasis - this is the PRIMARY source
        if top_chunks:
            first_chunk = top_chunks[0]
            # Get full chunk text (not truncated)
            full_chunk_text = first_chunk.get('chunk_text', '')
            
            context_text += f"\n=== PRIMARY SOURCE - MOST RELEVANT (Page {first_chunk.get('page_number', 0)}) ===\n"
            context_text += f"{full_chunk_text}\n"
            context_text += "=== END PRIMARY SOURCE ===\n"
        
        # VERY STRICT system prompt - enforce using primary source
        system_prompt = f"""You are a documentation assistant. Answer the user's question using ONLY the PRIMARY SOURCE chunk below.

PRIMARY SOURCE:
{context_text}

CRITICAL INSTRUCTIONS:
1. Use ONLY the PRIMARY SOURCE chunk above - ignore all other information
2. Answer the question COMPLETELY using information from the PRIMARY SOURCE
3. PRESERVE SQL COMMANDS: If the source contains SQL commands or code, include them EXACTLY as shown, in a code block format
4. PRESERVE FORMATTING: Keep SQL statements together, don't break them into numbered list items
5. INCLUDE ALL STEPS: Include all steps from the PRIMARY SOURCE in the exact order shown
6. COMPLETE ANSWER: Make sure your answer includes all information needed to complete the task
7. DO NOT STOP MID-SENTENCE: Complete your answer fully
8. If you see "<<variable>>" in SQL, keep it as-is (it's a placeholder)

Answer format for SQL/commands:
- Use a code block () for SQL commands
- Keep SQL statements together as complete blocks
- Don't break SQL into numbered list items
- Include all commands exactly as shown in the source

Answer format for steps:
- Use numbered list (1., 2., 3.) for procedural steps
- Use code blocks for SQL commands and code
- Be clear and complete"""

        # Create the full prompt
        full_prompt = f"""{system_prompt}

USER QUESTION: {question}

ANSWER (use ONLY PRIMARY SOURCE, preserve SQL formatting, include all steps, complete your answer):"""

        # Call Ollama API with strict settings
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": [
                    "\n\nIf you are running",
                    "\nTo create the workspace",
                    "\nEnable Edition-Based Redefinition",
                    "\nCreate a workspace",
                    "\nAlter the APEX schema",
                    "\nRun the application"
                ]
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', 'No response generated').strip()
            
            # Post-process: Clean up the response
            if answer:
                # Remove content after stop phrases
                stop_phrases = [
                    "\nIf you are running Oracle E-Business Suite Release 12.2",
                    "\nTo create the workspace",
                    "\nEnable Edition-Based Redefinition",
                    "\nCreate a workspace",
                    "\nAlter the APEX schema",
                    "\nRun the application",
                    "\nClick Using Responsibilities"
                ]
                
                for phrase in stop_phrases:
                    if phrase in answer:
                        answer = answer.split(phrase)[0].strip()
                        break
                
                # Fix SQL formatting - ensure SQL commands are in code blocks
                if "create user" in answer.lower() or "grant" in answer.lower() or "identified by" in answer.lower():
                    lines = answer.split('\n')
                    formatted_lines = []
                    in_sql_block = False
                    sql_keywords = [
                        "create user", "identified by", "default tablespace", 
                        "grant", "alter user", "connect to", "sql*plus"
                    ]
                    
                    for line in lines:
                        line_lower = line.lower().strip()
                        # Detect SQL lines
                        if any(sql_keyword in line_lower for sql_keyword in sql_keywords):
                            if not in_sql_block:
                                formatted_lines.append("")
                                in_sql_block = True
                            # Remove numbered prefixes like "3)", "4)", etc.
                            cleaned_line = line
                            if line.strip() and line.strip()[0].isdigit() and ')' in line[:3]:
                                # Remove number and parenthesis at start
                                cleaned_line = line.split(')', 1)[-1].strip()
                            formatted_lines.append(cleaned_line)
                        elif in_sql_block:
                            # Check if this line ends the SQL block
                            if line.strip() and not any(sql_keyword in line.lower() for sql_keyword in sql_keywords):
                                # End SQL block if we hit a non-SQL line
                                if line.strip() and not line.strip().startswith(')'):
                                    formatted_lines.append("```")
                                    in_sql_block = False
                                    formatted_lines.append(line)
                                else:
                                    # Empty line or just closing paren - keep in SQL block
                                    formatted_lines.append(line)
                            else:
                                formatted_lines.append(line)
                        else:
                            formatted_lines.append(line)
                    
                    if in_sql_block:
                        formatted_lines.append("```")
                    
                    answer = '\n'.join(formatted_lines)
            
            return answer
        else:
            st.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
        return None
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False