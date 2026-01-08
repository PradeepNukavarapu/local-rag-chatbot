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
        
        # Detect if this is a "complete steps" question - need more context
        is_complete_steps_question = any(phrase in question.lower() for phrase in [
            "complete steps", "all steps", "full steps", "entire process", 
            "complete process", "all the steps", "step by step"
        ])
        
        # Detect if question asks for SQL/database commands
        is_sql_question = any(keyword in question.lower() for keyword in [
            "sql", "create user", "grant", "database user", "schema", 
            "privileges", "minimally privileged", "sql commands"
        ])
        
        # Use more chunks for complete steps or SQL questions
        if is_complete_steps_question or is_sql_question:
            # Use top 7 chunks for complete steps questions
            top_chunks = sorted_chunks[:7]
            # Prioritize chunks with SQL keywords
            sql_keywords = ["create user", "grant", "identified by", "tablespace", "alter user", "sql*plus"]
            chunks_with_sql = [c for c in top_chunks if any(kw in c.get('chunk_text', '').lower() for kw in sql_keywords)]
            other_chunks = [c for c in top_chunks if c not in chunks_with_sql]
            # Reorder: SQL chunks first, then others
            top_chunks = chunks_with_sql + other_chunks
        else:
            # Use top 3 for regular questions
            top_chunks = sorted_chunks[:3]
        
        # Prepare context - include multiple chunks for complete steps
        context_text = ""
        
        if is_complete_steps_question or is_sql_question:
            # For complete steps, include all relevant chunks
            for i, chunk in enumerate(top_chunks, 1):
                chunk_text = chunk.get('chunk_text', '')
                page_num = chunk.get('page_number', 0)
                doc_name = chunk.get('doc_name', 'Unknown')
                
                if i == 1:
                    context_text += f"\n=== PRIMARY SOURCE - MOST RELEVANT (Page {page_num}, {doc_name}) ===\n"
                else:
                    context_text += f"\n=== ADDITIONAL SOURCE {i} (Page {page_num}, {doc_name}) ===\n"
                context_text += f"{chunk_text}\n"
                context_text += f"=== END SOURCE {i} ===\n"
        else:
            # For regular questions, use only primary source
            if top_chunks:
                first_chunk = top_chunks[0]
                full_chunk_text = first_chunk.get('chunk_text', '')
                context_text += f"\n=== PRIMARY SOURCE - MOST RELEVANT (Page {first_chunk.get('page_number', 0)}) ===\n"
                context_text += f"{full_chunk_text}\n"
                context_text += "=== END PRIMARY SOURCE ===\n"
        
        # Adjust system prompt based on question type
        if is_complete_steps_question or is_sql_question:
            system_prompt = f"""You are a helpful documentation assistant. Answer the user's question using the source information provided below.

SOURCE INFORMATION:
{context_text}

INSTRUCTIONS:
- Use the source information above to provide a complete answer
- If the source contains SQL commands, include them in code blocks (```sql ... ```)
- If the source contains steps, list them in order
- Combine information from multiple sources if needed
- Do NOT repeat these instructions in your answer
- Start your answer directly without preamble"""
        else:
            # Regular question prompt
            system_prompt = f"""You are a helpful documentation assistant. Answer the user's question using the source information provided below.

SOURCE INFORMATION:
{context_text}

INSTRUCTIONS:
- Use the source information above to answer the question completely
- If the source contains SQL commands, include them in code blocks (```sql ... ```)
- If the source contains steps, list them in order
- Provide a clear, complete answer
- Do NOT repeat these instructions in your answer
- Start your answer directly without preamble"""

        # Create the full prompt - cleaner and more direct
        full_prompt = f"""{system_prompt}

USER QUESTION: {question}

Provide a clear, complete answer based on the source(s) above. Do NOT repeat these instructions. Start your answer directly:"""

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
                    "\nRun the application",
                    "\n<<variable>>",
                    "\nThe source does not provide",
                    "Do NOT repeat",
                    "CRITICAL INSTRUCTIONS",
                    "Answer format"
                ]
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', 'No response generated').strip()
            
            # Post-process: Clean up the response
            if answer:
                # Remove instruction echoes at the start
                instruction_patterns_start = [
                    "Use a code block",
                    "Keep SQL statements",
                    "Don't break SQL",
                    "Include all commands",
                    "Include ALL commands",
                    "Answer format for",
                    "CRITICAL INSTRUCTIONS:",
                    "PRIMARY SOURCE:",
                    "ADDITIONAL SOURCE",
                    "Do NOT repeat",
                    "Start your answer",
                    "Provide a clear, complete answer"
                ]
                
                lines = answer.split('\n')
                cleaned_lines = []
                skip_until_content = True
                found_real_content = False
                
                for line in lines:
                    line_stripped = line.strip()
                    # Check if this line is an instruction echo
                    is_instruction = any(pattern in line_stripped for pattern in instruction_patterns_start)
                    
                    # Also check if line is too short and looks like an instruction fragment
                    is_fragment = (len(line_stripped) < 50 and 
                                  any(word in line_stripped.lower() for word in ['format', 'command', 'instruction', 'source', 'answer']))
                    
                    if (is_instruction or is_fragment) and skip_until_content:
                        continue  # Skip instruction echoes
                    else:
                        # Found real content
                        if line_stripped and not is_instruction:
                            found_real_content = True
                            skip_until_content = False
                        if found_real_content or not skip_until_content:
                            cleaned_lines.append(line)
                
                answer = '\n'.join(cleaned_lines).strip()
                
                # Remove unwanted content at the end
                end_patterns = [
                    "<<variable>> The source does not provide",
                    "<<variable>>",
                    "The source does not provide",
                    "I will not include",
                    "Do NOT repeat",
                    "CRITICAL INSTRUCTIONS",
                    "Answer format",
                    "PRIMARY SOURCE",
                    "ADDITIONAL SOURCE",
                    "INSTRUCTIONS:",
                    "\n\nIf you are running Oracle E-Business Suite Release 12.2",
                    "\nTo create the workspace",
                    "\nEnable Edition-Based Redefinition",
                    "\nCreate a workspace",
                    "\nAlter the APEX schema",
                    "\nRun the application",
                    "\nClick Using Responsibilities"
                ]
                
                # Remove content after end patterns (but preserve <<variable>> if it's in SQL code blocks)
                # First, check if <<variable>> is in a code block (should be kept)
                in_code_block = False
                for pattern in end_patterns:
                    if pattern in answer and "<<variable>>" not in pattern:
                        # Find the last occurrence and remove everything after it
                        idx = answer.rfind(pattern)
                        if idx > len(answer) * 0.5:  # Only if it's in the latter half
                            answer = answer[:idx].strip()
                            break
                
                # Remove trailing instruction-like text line by line
                answer_lines = answer.split('\n')
                while answer_lines:
                    last_line = answer_lines[-1].strip()
                    # Check if last line matches end patterns
                    matches_end_pattern = any(pattern in last_line for pattern in end_patterns)
                    # Also check if it's a short line that looks like an instruction fragment
                    is_short_instruction = (len(last_line) < 60 and 
                                          any(word in last_line.lower() for word in ['format', 'command', 'instruction', 'source', 'answer', 'provide', 'include']))
                    
                    if matches_end_pattern or is_short_instruction:
                        answer_lines.pop()
                    else:
                        break
                answer = '\n'.join(answer_lines).strip()
                
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