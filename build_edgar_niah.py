"""
build_edgar_niah.py

A comprehensive script to build a Needle-In-A-Haystack (NIAH) dataset from SEC EDGAR filings.

Requirements: Python 3.9+, requests, pandas, beautifulsoup4, tqdm
Install: pip install requests pandas beautifulsoup4 tqdm
"""

import requests
import time
import json
import csv
import os
import uuid
import re
import random
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ---------- CONFIGURATION ----------
# Tickers or CIKs to process
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Can be extended
CIKS = []  # Optional: provide CIKs directly (e.g., ["0000320193"] for AAPL)

# Filing types to download
FORM_TYPES = ["10-K", "10-Q"]

# Limits
MAX_FILINGS_PER_TICKER = 5  # Maximum filings per ticker
MIN_HAYSTACK_LENGTH = 1000  # Minimum character length for haystack
MIN_NEEDLE_LENGTH = 20  # Minimum character length for needle

# Output configuration
OUT_DIR = "edgar_niah_dataset"
JSONL_PATH = os.path.join(OUT_DIR, "edgar_niah.jsonl")
CSV_PATH = os.path.join(OUT_DIR, "edgar_niah.csv")

# SEC API configuration
USER_AGENT = "ResearchProject/1.0 (contact@example.com)"  # SEC requires descriptive User-Agent
SLEEP_BETWEEN_CALLS = 0.75  # Rate limiting: 0.5-1.0 seconds between calls
REQUEST_TIMEOUT = 30

# NIAH generation parameters
NUM_EXTRACTIVE_PER_FILING = 3  # Number of extractive NIAH examples per filing
NUM_INSERTIVE_PER_FILING = 3  # Number of insertive NIAH examples per filing

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# -----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def get_company_cik(ticker: str) -> Optional[str]:
    """
    Get CIK for a given ticker symbol using SEC companytickers.json.
    Returns CIK as a string (zero-padded to 10 digits).
    """
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry.get("cik_str", ""))
                # Zero-pad CIK to 10 digits
                return cik.zfill(10)
        return None
    except Exception as e:
        print(f"Error fetching CIK for {ticker}: {e}")
        return None


def get_company_filings(cik: str, form_types: List[str], limit: int = 10) -> List[Dict]:
    """
    Retrieve filings for a company using SEC EDGAR submissions API.
    
    Args:
        cik: Company CIK (zero-padded to 10 digits)
        form_types: List of form types (e.g., ["10-K", "10-Q"])
        limit: Maximum number of filings to return
    
    Returns:
        List of filing metadata dictionaries
    """
    try:
        # SEC EDGAR submissions API endpoint
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        
        time.sleep(SLEEP_BETWEEN_CALLS)
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        filings = []
        recent_filings = data.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            return filings
        
        form_type_list = recent_filings.get("form", [])
        accession_list = recent_filings.get("accessionNumber", [])
        filing_date_list = recent_filings.get("filingDate", [])
        primary_doc_list = recent_filings.get("primaryDocument", [])
        
        # Collect filings matching the requested form types
        collected = 0
        for idx, form_type in enumerate(form_type_list):
            if collected >= limit:
                break
            
            if form_type in form_types:
                accession = accession_list[idx].replace("-", "")
                filing_date = filing_date_list[idx]
                primary_doc = primary_doc_list[idx]
                
                # Construct filing URL
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
                )
                
                filings.append({
                    "cik": cik,
                    "accession": accession_list[idx],  # Keep dashes for display
                    "accession_no_dash": accession,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "primary_document": primary_doc,
                    "url": filing_url
                })
                collected += 1
        
        return filings
    
    except Exception as e:
        print(f"Error fetching filings for CIK {cik}: {e}")
        return []


def fetch_filing_text(filing_meta: Dict) -> Optional[str]:
    """
    Fetch and extract clean text from an EDGAR filing HTML document.
    
    Args:
        filing_meta: Dictionary containing filing metadata including 'url'
    
    Returns:
        Cleaned text content of the filing, or None if extraction fails
    """
    url = filing_meta.get("url")
    if not url:
        return None
    
    try:
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        time.sleep(SLEEP_BETWEEN_CALLS)
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Try to find the main document content
        # EDGAR filings often use <document> tags or specific div classes
        text_content = None
        
        # Strategy 1: Look for <document> tags
        documents = soup.find_all("document")
        if documents:
            texts = []
            for doc in documents:
                text = doc.get_text(separator=" ", strip=True)
                if text and len(text) > 100:
                    texts.append(text)
            if texts:
                text_content = "\n\n".join(texts)
        
        # Strategy 2: Look for common content containers
        if not text_content:
            content_tags = soup.find_all(["div", "p", "span"], 
                                        class_=re.compile(r"document|content|text|body", re.I))
            if content_tags:
                texts = []
                for tag in content_tags:
                    text = tag.get_text(separator=" ", strip=True)
                    if text and len(text) > 100:
                        texts.append(text)
                if texts:
                    text_content = "\n\n".join(texts[:10])  # Limit to first 10 to avoid duplicates
        
        # Strategy 3: Fallback to body text
        if not text_content:
            body = soup.find("body")
            if body:
                text_content = body.get_text(separator=" ", strip=True)
            else:
                text_content = soup.get_text(separator=" ", strip=True)
        
        if not text_content:
            return None
        
        # Normalize whitespace
        text_content = re.sub(r'\s+', ' ', text_content)  # Collapse multiple spaces
        text_content = re.sub(r'\n\s*\n', '\n\n', text_content)  # Normalize line breaks
        text_content = text_content.strip()
        
        return text_content if len(text_content) >= MIN_HAYSTACK_LENGTH else None
    
    except Exception as e:
        print(f"Error fetching filing text from {url}: {e}")
        return None


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using heuristic rules.
    
    Args:
        text: Input text to split
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting: split on . ? ! followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
    return sentences


def generate_natural_query_from_needle(needle: str, ticker: str = "", filing_type: str = "") -> str:
    """
    Generate a natural query based on the content of the needle text.
    The query should be answerable by the needle but not hint at its location.
    
    Args:
        needle: The needle text (what we're looking for)
        ticker: Company ticker symbol
        filing_type: Type of filing (10-K, 10-Q)
    
    Returns:
        A natural question that can be answered by the needle
    """
    needle_lower = needle.lower()
    
    # Extract financial figures and dates
    date_pattern = r'(\w+\s+\d{1,2},?\s+\d{4})|(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, needle)
    
    # Common financial statement patterns
    if "balance sheet" in needle_lower or "total assets" in needle_lower:
        # Try to extract the date
        date_match = re.search(r'(September|December|March|June)\s+(\d{1,2}),?\s+(\d{4})', needle)
        if date_match:
            date_str = date_match.group(0)
            return f"What were {ticker}'s total assets as of {date_str}?"
        return f"What were {ticker}'s total assets?" if ticker else "What were the total assets?"
    
    elif "income" in needle_lower and "tax" in needle_lower:
        if "effective tax rate" in needle_lower:
            return f"What was {ticker}'s effective tax rate?" if ticker else "What was the effective tax rate?"
        return f"What was {ticker}'s provision for income taxes?" if ticker else "What was the provision for income taxes?"
    
    elif "revenue" in needle_lower or "net sales" in needle_lower:
        return f"What were {ticker}'s total net sales?" if ticker else "What were the total net sales?"
    
    elif "cash and cash equivalents" in needle_lower or "cash equivalents" in needle_lower:
        date_match = re.search(r'(September|December|March|June)\s+(\d{1,2}),?\s+(\d{4})', needle)
        if date_match:
            date_str = date_match.group(0)
            return f"What were {ticker}'s cash and cash equivalents as of {date_str}?" if ticker else f"What were cash and cash equivalents as of {date_str}?"
        return f"What were {ticker}'s cash and cash equivalents?" if ticker else "What were cash and cash equivalents?"
    
    elif "accounts receivable" in needle_lower:
        return f"What were {ticker}'s accounts receivable?" if ticker else "What were accounts receivable?"
    
    elif "inventory" in needle_lower or "inventories" in needle_lower:
        return f"What were {ticker}'s inventories?" if ticker else "What were inventories?"
    
    elif "debt" in needle_lower or "notes due" in needle_lower:
        return f"What was {ticker}'s total debt?" if ticker else "What was the total debt?"
    
    elif "shareholders' equity" in needle_lower or "stockholders' equity" in needle_lower:
        return f"What was {ticker}'s total shareholders' equity?" if ticker else "What was total shareholders' equity?"
    
    elif "operating income" in needle_lower:
        return f"What was {ticker}'s operating income?" if ticker else "What was operating income?"
    
    elif "net income" in needle_lower:
        return f"What was {ticker}'s net income?" if ticker else "What was net income?"
    
    # For insertive synthetic codes - create a more natural query
    elif needle.startswith("SECRET_CODE:"):
        # Extract the code part for the query
        code = needle.replace("SECRET_CODE:", "").strip()
        # Make it sound like a natural financial identifier
        if ticker:
            return f"What is the {ticker} internal reporting code mentioned in this {filing_type}?" if filing_type else f"What is the {ticker} internal reporting code?"
        return "What internal reporting identifier is mentioned in the financial statements?"
    
    # Generic fallback - try to extract key terms and form a question
    # Look for key financial terms
    key_terms = []
    if re.search(r'\$[\d,]+', needle):  # Contains dollar amounts
        key_terms.append("financial figure")
    if re.search(r'\d{4}', needle):  # Contains year
        years = re.findall(r'\d{4}', needle)
        if years:
            key_terms.append(f"for {years[0]}")
    
    if key_terms:
        return f"What {key_terms[0]} is reported in this {filing_type}?" if filing_type else f"What {key_terms[0]} is reported?"
    
    # Last resort: ask about a specific detail from the sentence
    # Extract first 50 chars and ask about it
    first_part = needle[:100].strip()
    # Remove common prefixes
    first_part = re.sub(r'^(CONSOLIDATED|NOTE|ITEM|PART)\s+', '', first_part, flags=re.I)
    if len(first_part) > 30:
        # Ask about the topic
        words = first_part.split()[:5]  # First few words
        topic = " ".join(words)
        return f"What information is provided about {topic}?"
    
    # Ultimate fallback
    return "What specific detail is mentioned in the financial statements?"


def create_extractive_niah(filing_text: str, num_examples: int = 3, ticker: str = "", filing_type: str = "") -> List[Dict]:
    """
    Create extractive NIAH examples by selecting real sentences from the filing.
    
    Args:
        filing_text: Full filing text (haystack)
        num_examples: Number of examples to generate
    
    Returns:
        List of NIAH example dictionaries
    """
    examples = []
    sentences = split_into_sentences(filing_text)
    
    if len(sentences) < num_examples:
        return examples
    
    # Prefer sentences that look like factual statements (contain numbers, financial terms, etc.)
    financial_keywords = [
        r'\d+', r'%', r'million', r'billion', r'revenue', r'income', r'profit', 
        r'loss', r'asset', r'liability', r'equity', r'earnings', r'share',
        r'company', r'corporation', r'fiscal', r'quarter', r'year'
    ]
    
    # Score sentences by how many keywords they contain
    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        score = sum(1 for pattern in financial_keywords if re.search(pattern, sentence, re.I))
        if score > 0:
            scored_sentences.append((idx, sentence, score))
    
    # Sort by score (descending) and take top candidates
    scored_sentences.sort(key=lambda x: x[2], reverse=True)
    
    # Select up to num_examples sentences
    selected_indices = set()
    for idx, sentence, _ in scored_sentences[:num_examples * 2]:  # Get more candidates
        if len(selected_indices) >= num_examples:
            break
        if len(sentence.strip()) >= MIN_NEEDLE_LENGTH:
            selected_indices.add(idx)
    
    # If we don't have enough scored sentences, randomly select from remaining
    if len(selected_indices) < num_examples:
        remaining = [i for i in range(len(sentences)) if i not in selected_indices]
        random.shuffle(remaining)
        for idx in remaining[:num_examples - len(selected_indices)]:
            if len(sentences[idx].strip()) >= MIN_NEEDLE_LENGTH:
                selected_indices.add(idx)
    
    # Create examples for selected sentences
    for idx in list(selected_indices)[:num_examples]:
        needle = sentences[idx].strip()
        
        # Find the exact character span in the original text
        # Try exact match first
        start = filing_text.find(needle)
        if start == -1:
            # Try to find approximate match (handle whitespace variations)
            needle_normalized = re.sub(r'\s+', ' ', needle)
            text_normalized = re.sub(r'\s+', ' ', filing_text)
            start_normalized = text_normalized.find(needle_normalized)
            if start_normalized != -1:
                # Map back to original positions (approximate)
                start = start_normalized
                end = start + len(needle)
            else:
                # Fallback: estimate position based on sentence index
                # This is approximate but should be close
                if idx > 0:
                    prior_text = " ".join(sentences[:idx])
                    start = len(prior_text) + 1  # +1 for separator
                else:
                    start = 0
                end = start + len(needle)
        else:
            end = start + len(needle)
        
        # Generate a natural query based on the needle content
        query = generate_natural_query_from_needle(needle, ticker, filing_type)
        
        examples.append({
            "haystack": filing_text,
            "needle": needle,
            "needle_span": [start, end],
            "query": query,
            "expected_answer": needle
        })
    
    return examples


def create_insertive_niah(filing_text: str, num_examples: int = 3, ticker: str = "", filing_type: str = "") -> List[Dict]:
    """
    Create insertive NIAH examples by inserting synthetic tokens into the filing.
    
    Args:
        filing_text: Full filing text (haystack)
        num_examples: Number of examples to generate
    
    Returns:
        List of NIAH example dictionaries
    """
    examples = []
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', filing_text)
    if not paragraphs:
        paragraphs = [filing_text]
    
    # Filter out very short paragraphs
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]
    
    if len(paragraphs) < num_examples:
        # If not enough paragraphs, split by sentences
        sentences = split_into_sentences(filing_text)
        paragraphs = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    for i in range(num_examples):
        # Generate a unique synthetic token
        secret_code = f"SECRET_CODE:{uuid.uuid4().hex[:8].upper()}"
        
        # Select a random paragraph to insert into
        target_idx = i % len(paragraphs)
        target_paragraph = paragraphs[target_idx]
        
        # Insert the secret code at a random position within the paragraph
        # Prefer inserting in the middle or end, not at the very start
        insert_pos = random.randint(len(target_paragraph) // 4, len(target_paragraph) - 10)
        
        # Find the insert position in the full text
        # Build the modified paragraph
        modified_paragraph = target_paragraph[:insert_pos] + " " + secret_code + " " + target_paragraph[insert_pos:]
        
        # Replace the paragraph in the full text
        paragraphs_copy = paragraphs.copy()
        paragraphs_copy[target_idx] = modified_paragraph
        haystack = "\n\n".join(paragraphs_copy)
        
        # Find the exact span of the inserted needle
        start = haystack.find(secret_code)
        if start == -1:
            # Should not happen, but handle gracefully
            continue
        end = start + len(secret_code)
        
        # Generate a natural query for the secret code
        query = generate_natural_query_from_needle(secret_code, ticker, filing_type)
        
        examples.append({
            "haystack": haystack,
            "needle": secret_code,
            "needle_span": [start, end],
            "query": query,
            "expected_answer": secret_code
        })
    
    return examples


def normalize_text(text: str) -> str:
    """
    Normalize text by removing excessive whitespace and ensuring UTF-8 compatibility.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def create_niah_examples(filing_text: str, filing_meta: Dict) -> List[Dict]:
    """
    Create all NIAH examples (both extractive and insertive) for a filing.
    
    Args:
        filing_text: Cleaned filing text
        filing_meta: Filing metadata dictionary
    
    Returns:
        List of complete NIAH example records
    """
    records = []
    
    # Normalize filing text
    filing_text = normalize_text(filing_text)
    
    if len(filing_text) < MIN_HAYSTACK_LENGTH:
        return records
    
    # Extract metadata for query generation
    ticker = filing_meta.get("ticker", "")
    filing_type = filing_meta.get("form_type", "")
    
    # Create extractive examples
    extractive_examples = create_extractive_niah(filing_text, NUM_EXTRACTIVE_PER_FILING, ticker, filing_type)
    for ex in extractive_examples:
        record = {
            "id": str(uuid.uuid4()),
            "source_ticker": filing_meta.get("ticker", ""),
            "source_cik": filing_meta.get("cik", ""),
            "filing_type": filing_meta.get("form_type", ""),
            "filing_date": filing_meta.get("filing_date", ""),
            "doc_id": filing_meta.get("accession", ""),
            "haystack": ex["haystack"],
            "needle": ex["needle"],
            "needle_span": ex["needle_span"],
            "query": ex["query"],
            "expected_answer": ex["expected_answer"],
            "metadata": {
                "method": "extract",
                "paragraph_index": None,  # Could be enhanced with paragraph tracking
                "section_title": None  # Could be enhanced with section extraction
            },
            "difficulty": "medium"
        }
        records.append(record)
    
    # Create insertive examples
    insertive_examples = create_insertive_niah(filing_text, NUM_INSERTIVE_PER_FILING, ticker, filing_type)
    for ex in insertive_examples:
        record = {
            "id": str(uuid.uuid4()),
            "source_ticker": filing_meta.get("ticker", ""),
            "source_cik": filing_meta.get("cik", ""),
            "filing_type": filing_meta.get("form_type", ""),
            "filing_date": filing_meta.get("filing_date", ""),
            "doc_id": filing_meta.get("accession", ""),
            "haystack": ex["haystack"],
            "needle": ex["needle"],
            "needle_span": ex["needle_span"],
            "query": ex["query"],
            "expected_answer": ex["expected_answer"],
            "metadata": {
                "method": "insert",
                "paragraph_index": None,  # Could be enhanced with paragraph tracking
                "section_title": None  # Could be enhanced with section extraction
            },
            "difficulty": "medium"
        }
        records.append(record)
    
    return records


def write_jsonl(records: List[Dict], filepath: str):
    """
    Write records to JSONL file.
    
    Args:
        records: List of record dictionaries
        filepath: Output file path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def write_csv(records: List[Dict], filepath: str):
    """
    Write records to CSV file with haystack truncated to 1000 chars.
    
    Args:
        records: List of record dictionaries
        filepath: Output file path
    """
    if not records:
        return
    
    # Prepare CSV records
    csv_records = []
    for record in records:
        csv_record = record.copy()
        # Truncate haystack for CSV
        haystack = csv_record.get("haystack", "")
        if len(haystack) > 1000:
            csv_record["haystack"] = haystack[:1000] + "..."
        
        # Convert needle_span to string representation
        needle_span = csv_record.get("needle_span")
        if needle_span is None:
            csv_record["needle_span"] = ""
        elif isinstance(needle_span, list):
            csv_record["needle_span"] = str(needle_span)
        else:
            csv_record["needle_span"] = str(needle_span)
        
        # Convert metadata to JSON string
        metadata = csv_record.get("metadata")
        if metadata is None:
            csv_record["metadata"] = "{}"
        elif isinstance(metadata, dict):
            csv_record["metadata"] = json.dumps(metadata)
        else:
            csv_record["metadata"] = str(metadata)
        
        csv_records.append(csv_record)
    
    # Write CSV
    fieldnames = [
        "id", "source_ticker", "source_cik", "filing_type", "filing_date",
        "doc_id", "haystack", "needle", "needle_span", "query",
        "expected_answer", "metadata", "difficulty"
    ]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_records)


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("EDGAR NIAH Dataset Builder")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Form types: {FORM_TYPES}")
    print(f"Max filings per ticker: {MAX_FILINGS_PER_TICKER}")
    print(f"Examples per filing: {NUM_EXTRACTIVE_PER_FILING} extractive + {NUM_INSERTIVE_PER_FILING} insertive")
    print("=" * 60)
    print()
    
    all_records = []
    
    # Process tickers
    tickers_to_process = TICKERS.copy()
    
    # If CIKs are provided directly, use them
    if CIKS:
        for cik in CIKS:
            # Try to find corresponding ticker (optional)
            ticker = None
            tickers_to_process.append({"ticker": ticker, "cik": cik})
    
    # Process each ticker
    for ticker in tqdm(tickers_to_process, desc="Processing tickers"):
        if isinstance(ticker, dict):
            # Direct CIK provided
            cik = ticker["cik"]
            ticker_symbol = ticker.get("ticker", f"CIK{cik}")
        else:
            # Get CIK for ticker
            print(f"\nProcessing {ticker}...")
            cik = get_company_cik(ticker)
            if not cik:
                print(f"  Could not find CIK for {ticker}, skipping...")
                continue
            ticker_symbol = ticker
            print(f"  Found CIK: {cik}")
        
        # Get filings for this company
        filings = get_company_filings(cik, FORM_TYPES, limit=MAX_FILINGS_PER_TICKER)
        if not filings:
            print(f"  No filings found for {ticker_symbol}")
            continue
        
        print(f"  Found {len(filings)} filings")
        
        # Process each filing
        for filing_meta in tqdm(filings, desc=f"  Processing {ticker_symbol} filings", leave=False):
            filing_meta["ticker"] = ticker_symbol
            
            # Fetch filing text
            filing_text = fetch_filing_text(filing_meta)
            if not filing_text:
                print(f"    Could not extract text from {filing_meta.get('url', 'unknown')}")
                continue
            
            print(f"    Extracted {len(filing_text)} characters from {filing_meta.get('form_type')} filed on {filing_meta.get('filing_date')}")
            
            # Create NIAH examples
            examples = create_niah_examples(filing_text, filing_meta)
            all_records.extend(examples)
            print(f"    Created {len(examples)} NIAH examples")
    
    # Write output files
    print(f"\nWriting output files...")
    print(f"  Total records: {len(all_records)}")
    
    write_jsonl(all_records, JSONL_PATH)
    print(f"  ✓ Written JSONL: {JSONL_PATH}")
    
    write_csv(all_records, CSV_PATH)
    print(f"  ✓ Written CSV: {CSV_PATH}")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"JSONL file: {JSONL_PATH}")
    print(f"CSV file: {CSV_PATH}")
    print(f"Total examples: {len(all_records)}")
    print(f"  - Extractive: {sum(1 for r in all_records if r.get('metadata', {}).get('method') == 'extract')}")
    print(f"  - Insertive: {sum(1 for r in all_records if r.get('metadata', {}).get('method') == 'insert')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

