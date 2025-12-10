"""
view_dataset.py

A human-readable viewer for the EDGAR NIAH dataset.
Makes it easy to see what's actually in the JSONL/CSV files!
"""

import json
import sys
from pathlib import Path

def pretty_print_example(example, example_num=1, max_haystack_preview=200, max_needle_preview=300):
    """Print a single example in a nice, readable format."""
    
    print("=" * 80)
    print(f"EXAMPLE #{example_num}")
    print("=" * 80)
    
    # Basic Info
    print(f"\nBasic Information:")
    print(f"   ID: {example.get('id', 'N/A')}")
    print(f"   Company: {example.get('source_ticker', 'N/A')} (CIK: {example.get('source_cik', 'N/A')})")
    print(f"   Filing Type: {example.get('filing_type', 'N/A')}")
    print(f"   Filing Date: {example.get('filing_date', 'N/A')}")
    
    # Method
    metadata = example.get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}
    
    method = metadata.get('method', 'unknown') if isinstance(metadata, dict) else 'unknown'
    method_name = "Extractive" if method == "extract" else "Insertive"
    print(f"   Method: {method_name} (real sentence from filing)" if method == "extract" else f"   Method: {method_name} (synthetic code inserted)")
    
    # Query
    print(f"\nTHE QUESTION:")
    print(f"   \"{example.get('query', 'N/A')}\"")
    
    # Haystack
    haystack = str(example.get('haystack', ''))
    haystack_len = len(haystack)
    print(f"\nTHE HAYSTACK (The Full Document):")
    print(f"   Total Length: {haystack_len:,} characters")
    print(f"   (That's like {haystack_len/1000:.1f} pages of text!)")
    print(f"\n   Preview (first {max_haystack_preview} characters):")
    print(f"   {'-' * 76}")
    preview = haystack[:max_haystack_preview].replace('\n', ' ')
    # Wrap text nicely
    words = preview.split()
    line = ""
    for word in words:
        if len(line + word) > 76:
            print(f"   {line}")
            line = word + " "
        else:
            line += word + " "
    if line:
        print(f"   {line}...")
    print(f"   {'-' * 76}")
    
    # Needle
    needle = str(example.get('needle', ''))
    needle_len = len(needle)
    print(f"\nTHE NEEDLE (What We're Looking For):")
    print(f"   Length: {needle_len} characters")
    
    needle_span = example.get('needle_span', [])
    if needle_span and len(needle_span) >= 2:
        start, end = needle_span[0], needle_span[1]
        position_pct = (start / haystack_len * 100) if haystack_len > 0 else 0
        print(f"   Location: Characters {start:,} to {end:,} in the haystack")
        print(f"   Position: {position_pct:.1f}% into the document")
    
    print(f"\n   The needle text:")
    print(f"   {'-' * 76}")
    
    # Show needle - wrap if too long
    if needle_len > max_needle_preview:
        needle_preview = needle[:max_needle_preview]
        words = needle_preview.split()
        line = ""
        for word in words:
            if len(line + word) > 76:
                print(f"   {line}")
                line = word + " "
            else:
                line += word + " "
        if line:
            print(f"   {line}...")
        print(f"   {'-' * 76}")
        print(f"   (... {needle_len - max_needle_preview} more characters)")
    else:
        words = needle.split()
        line = ""
        for word in words:
            if len(line + word) > 76:
                print(f"   {line}")
                line = word + " "
            else:
                line += word + " "
        if line:
            print(f"   {line}")
        print(f"   {'-' * 76}")
    
    # Expected Answer
    expected = str(example.get('expected_answer', ''))
    if expected != needle:
        print(f"\nEXPECTED ANSWER:")
        print(f"   {expected[:200]}..." if len(expected) > 200 else f"   {expected}")
    
    # Challenge
    print(f"\nTHE CHALLENGE:")
    if method == "extract":
        print(f"   Can a language model find this REAL sentence (from a financial report)")
        print(f"   when it's hidden inside a {haystack_len:,}-character document?")
    else:
        print(f"   Can a language model find this SYNTHETIC SECRET CODE")
        print(f"   when it's hidden inside a {haystack_len:,}-character document?")
    
    print()


def view_dataset(filepath, num_examples=3, method_filter=None):
    """View examples from the dataset in a readable format."""
    
    print("\n" + "=" * 80)
    print("EDGAR NIAH DATASET VIEWER")
    print("=" * 80)
    
    examples = []
    
    # Load JSONL
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    
    # Load CSV would need different handling, but let's stick with JSONL
    
    print(f"\nDataset Info:")
    print(f"   Total examples in file: {len(examples)}")
    
    # Filter by method if requested
    if method_filter:
        filtered = []
        for ex in examples:
            metadata = ex.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            method = metadata.get('method', '') if isinstance(metadata, dict) else ''
            if method == method_filter:
                filtered.append(ex)
        examples = filtered
        print(f"   Showing only '{method_filter}' examples: {len(examples)} found")
    
    # Show requested number of examples
    num_to_show = min(num_examples, len(examples))
    print(f"   Displaying {num_to_show} examples below\n")
    
    for i in range(num_to_show):
        pretty_print_example(examples[i], example_num=i+1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='View EDGAR NIAH dataset in readable format')
    parser.add_argument('--file', '-f', 
                       default='edgar_niah_dataset/edgar_niah.jsonl',
                       help='Path to JSONL file')
    parser.add_argument('--num', '-n', type=int, default=3,
                       help='Number of examples to show')
    parser.add_argument('--method', '-m', 
                       choices=['extract', 'insert', None],
                       help='Filter by method type')
    
    args = parser.parse_args()
    
    view_dataset(args.file, args.num, args.method)

