#!/usr/bin/env python3
"""
Web Scraper Module
Handles web scraping from URLs and text extraction
"""

import requests
from bs4 import BeautifulSoup
import streamlit as st
from urllib.parse import urlparse, urljoin
import re

# User agent to avoid blocking
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def validate_url(url):
    """
    Validate if URL is accessible and valid
    
    Args:
        url: URL string to validate
    
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format. Please include http:// or https://"
        
        if parsed.scheme not in ['http', 'https']:
            return False, f"Unsupported URL scheme: {parsed.scheme}. Only http:// and https:// are supported."
        
        # Try to connect
        response = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            return False, f"URL returned error status: {response.status_code}"
        
        return True, f"URL is accessible (Status: {response.status_code})"
    except requests.exceptions.Timeout:
        return False, "Connection timeout. The URL may be unreachable."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check the URL and your internet connection."
    except requests.exceptions.RequestException as e:
        return False, f"Error accessing URL: {str(e)}"
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"

def scrape_url(url):
    """
    Scrape content from a URL
    
    Args:
        url: URL to scrape
    
    Returns:
        dict: Dictionary with 'title', 'text', 'url' keys, or None if error
    """
    try:
        # Fetch the page
        response = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            script.decompose()
        
        # Get title
        title = soup.find('title')
        title_text = title.string if title else urlparse(url).netloc
        
        # Get main content - try to find main/article/content sections first
        main_content = None
        for tag in ['main', 'article', '[role="main"]', '.content', '#content']:
            main_content = soup.select_one(tag)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = text.strip()
        
        if not text or len(text) < 100:
            return None
        
        return {
            'title': title_text.strip(),
            'text': text,
            'url': url,
            'char_count': len(text)
        }
    except requests.exceptions.Timeout:
        st.error(f"Timeout while scraping {url}. The page may be too slow.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

def scrape_url_to_pages(url):
    """
    Scrape URL and convert to pages format (similar to document extraction)
    
    Args:
        url: URL to scrape
    
    Returns:
        list: List of page dictionaries with 'page_number' and 'text' keys
    """
    scraped_data = scrape_url(url)
    if not scraped_data:
        return None
    
    # Split content into pages (similar to document processing)
    # For web content, we'll create sections based on length
    text = scraped_data['text']
    pages_data = []
    
    # Split into sections of ~3000 characters (similar to document pages)
    section_size = 3000
    sections = [text[i:i+section_size] for i in range(0, len(text), section_size)]
    
    for idx, section_text in enumerate(sections):
        if section_text.strip():
            pages_data.append({
                'page_number': idx + 1,
                'text': section_text.strip()
            })
    
    return pages_data

