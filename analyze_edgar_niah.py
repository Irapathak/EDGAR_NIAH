"""
analyze_edgar_niah.py

Analysis and visualization tools for the EDGAR NIAH dataset.

Requirements: pandas, matplotlib, seaborn, json
Install: pip install pandas matplotlib seaborn
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import List, Dict
import os

# Configuration
DATA_DIR = "edgar_niah_dataset"
JSONL_PATH = os.path.join(DATA_DIR, "edgar_niah.jsonl")
CSV_PATH = os.path.join(DATA_DIR, "edgar_niah.csv")


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_dataset(jsonl_path: str = JSONL_PATH, csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the dataset. Prefer JSONL for full data, but can use CSV for quick checks.
    
    Returns:
        DataFrame with all dataset records
    """
    if os.path.exists(jsonl_path):
        print(f"Loading from JSONL: {jsonl_path}")
        records = load_jsonl(jsonl_path)
        df = pd.DataFrame(records)
        
        # Parse metadata if it's a string (from CSV) or keep as dict
        if 'metadata' in df.columns and isinstance(df['metadata'].iloc[0], str):
            df['metadata'] = df['metadata'].apply(json.loads)
        
        # Parse needle_span if it's a string
        if 'needle_span' in df.columns:
            def parse_span(span):
                if isinstance(span, str):
                    try:
                        return eval(span)  # Convert "[start, end]" string to list
                    except:
                        return None
                return span
            df['needle_span'] = df['needle_span'].apply(parse_span)
        
        return df
    elif os.path.exists(csv_path):
        print(f"Loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    else:
        raise FileNotFoundError(f"Neither {jsonl_path} nor {csv_path} found. Run build_edgar_niah.py first.")


def basic_statistics(df: pd.DataFrame):
    """Print basic statistics about the dataset."""
    print("=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(f"Total examples: {len(df)}")
    print(f"Unique filings: {df['doc_id'].nunique()}")
    print(f"Unique companies (tickers): {df['source_ticker'].nunique()}")
    print(f"Unique CIKs: {df['source_cik'].nunique()}")
    print()
    
    # Method breakdown
    if 'metadata' in df.columns:
        df['method'] = df['metadata'].apply(lambda x: x.get('method', 'unknown') if isinstance(x, dict) else 'unknown')
        method_counts = df['method'].value_counts()
        print("Method distribution:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} ({count/len(df)*100:.1f}%)")
        print()
    
    # Filing type breakdown
    if 'filing_type' in df.columns:
        filing_counts = df['filing_type'].value_counts()
        print("Filing type distribution:")
        for ftype, count in filing_counts.items():
            print(f"  {ftype}: {count} ({count/len(df)*100:.1f}%)")
        print()
    
    # Company breakdown
    if 'source_ticker' in df.columns:
        company_counts = df['source_ticker'].value_counts().head(10)
        print("Top companies by example count:")
        for ticker, count in company_counts.items():
            print(f"  {ticker}: {count}")
        print()
    
    # Date range
    if 'filing_date' in df.columns:
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        dates = df['filing_date'].dropna()
        if len(dates) > 0:
            print(f"Filing date range: {dates.min()} to {dates.max()}")
            print()


def analyze_haystack_needle_lengths(df: pd.DataFrame):
    """Analyze haystack and needle text lengths."""
    print("=" * 60)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 60)
    
    df['haystack_length'] = df['haystack'].astype(str).apply(len)
    df['needle_length'] = df['needle'].astype(str).apply(len)
    df['needle_ratio'] = df['needle_length'] / df['haystack_length'] * 100
    
    print(f"Haystack length statistics (characters):")
    print(f"  Mean: {df['haystack_length'].mean():.0f}")
    print(f"  Median: {df['haystack_length'].median():.0f}")
    print(f"  Min: {df['haystack_length'].min()}")
    print(f"  Max: {df['haystack_length'].max()}")
    print(f"  Std: {df['haystack_length'].std():.0f}")
    print()
    
    print(f"Needle length statistics (characters):")
    print(f"  Mean: {df['needle_length'].mean():.1f}")
    print(f"  Median: {df['needle_length'].median():.1f}")
    print(f"  Min: {df['needle_length'].min()}")
    print(f"  Max: {df['needle_length'].max()}")
    print(f"  Std: {df['needle_length'].std():.1f}")
    print()
    
    print(f"Needle/Haystack ratio (%):")
    print(f"  Mean: {df['needle_ratio'].mean():.3f}%")
    print(f"  Median: {df['needle_ratio'].median():.3f}%")
    print(f"  Min: {df['needle_ratio'].min():.3f}%")
    print(f"  Max: {df['needle_ratio'].max():.3f}%")
    print()


def analyze_needle_positions(df: pd.DataFrame):
    """Analyze where needles are positioned in haystacks."""
    print("=" * 60)
    print("NEEDLE POSITION ANALYSIS")
    print("=" * 60)
    
    def get_relative_position(span, haystack_len):
        if not span or not isinstance(span, list) or len(span) < 2:
            return None
        start = span[0]
        if haystack_len > 0:
            return start / haystack_len * 100
        return None
    
    df['haystack_length'] = df['haystack'].astype(str).apply(len)
    df['relative_position'] = df.apply(
        lambda row: get_relative_position(row.get('needle_span'), row['haystack_length']),
        axis=1
    )
    
    positions = df['relative_position'].dropna()
    if len(positions) > 0:
        print(f"Needle position statistics (% from start of haystack):")
        print(f"  Mean: {positions.mean():.1f}%")
        print(f"  Median: {positions.median():.1f}%")
        print(f"  Min: {positions.min():.1f}%")
        print(f"  Max: {positions.max():.1f}%")
        print()
        
        # Position distribution by quartiles
        print("Position distribution (quartiles):")
        q1 = positions.quantile(0.25)
        q2 = positions.quantile(0.50)
        q3 = positions.quantile(0.75)
        print(f"  0-25% (beginning): {len(positions[positions <= q1])} examples")
        print(f"  25-50% (early): {len(positions[(positions > q1) & (positions <= q2)])} examples")
        print(f"  50-75% (middle): {len(positions[(positions > q2) & (positions <= q3)])} examples")
        print(f"  75-100% (end): {len(positions[positions > q3])} examples")
        print()


def show_sample_examples(df: pd.DataFrame, n: int = 3, method: str = None):
    """Display sample examples from the dataset."""
    print("=" * 60)
    print(f"SAMPLE EXAMPLES{f' ({method})' if method else ''}")
    print("=" * 60)
    
    sample_df = df.copy()
    if method:
        if 'metadata' in sample_df.columns:
            sample_df['method'] = sample_df['metadata'].apply(
                lambda x: x.get('method', '') if isinstance(x, dict) else ''
            )
            sample_df = sample_df[sample_df['method'] == method]
    
    if len(sample_df) == 0:
        print(f"No examples found for method: {method}")
        return
    
    samples = sample_df.sample(min(n, len(sample_df)))
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n--- Example {idx} ---")
        print(f"ID: {row.get('id', 'N/A')}")
        print(f"Ticker: {row.get('source_ticker', 'N/A')}")
        print(f"Filing: {row.get('filing_type', 'N/A')} ({row.get('filing_date', 'N/A')})")
        if 'metadata' in row and isinstance(row['metadata'], dict):
            print(f"Method: {row['metadata'].get('method', 'N/A')}")
        print(f"\nQuery: {row.get('query', 'N/A')}")
        print(f"\nHaystack (first 500 chars):")
        haystack = str(row.get('haystack', ''))
        print(f"  {haystack[:500]}...")
        print(f"\nNeedle:")
        print(f"  {row.get('needle', 'N/A')}")
        if row.get('needle_span'):
            span = row['needle_span']
            if isinstance(span, list) and len(span) >= 2:
                print(f"\nNeedle Position: characters {span[0]} to {span[1]}")
        print(f"\nExpected Answer:")
        print(f"  {row.get('expected_answer', 'N/A')}")
        print("-" * 60)


def create_visualizations(df: pd.DataFrame, output_dir: str = "edgar_niah_dataset/plots"):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    df['haystack_length'] = df['haystack'].astype(str).apply(len)
    df['needle_length'] = df['needle'].astype(str).apply(len)
    
    if 'metadata' in df.columns:
        df['method'] = df['metadata'].apply(
            lambda x: x.get('method', 'unknown') if isinstance(x, dict) else 'unknown'
        )
    
    # Set style
    sns.set_style("whitegrid")
    fig_size = (12, 8)
    
    # 1. Haystack length distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['haystack_length'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Haystack Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Haystack Lengths', fontsize=14, fontweight='bold')
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'haystack_length_dist.png'), dpi=150)
    plt.close()
    print(f"Saved: Saved: {output_dir}/haystack_length_dist.png")
    
    # 2. Needle length distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['needle_length'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Needle Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Needle Lengths', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'needle_length_dist.png'), dpi=150)
    plt.close()
    print(f"Saved: Saved: {output_dir}/needle_length_dist.png")
    
    # 3. Method comparison
    if 'method' in df.columns:
        plt.figure(figsize=fig_size)
        method_counts = df['method'].value_counts()
        plt.bar(method_counts.index, method_counts.values, color=['#3498db', '#e74c3c'])
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Examples by Method Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: Saved: {output_dir}/method_distribution.png")
        
        # Haystack length by method
        plt.figure(figsize=fig_size)
        df.boxplot(column='haystack_length', by='method', ax=plt.gca())
        plt.suptitle('')  # Remove default title
        plt.title('Haystack Length by Method', fontsize=14, fontweight='bold')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Haystack Length (characters)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'haystack_length_by_method.png'), dpi=150)
        plt.close()
        print(f"Saved: Saved: {output_dir}/haystack_length_by_method.png")
    
    # 4. Filing type distribution
    if 'filing_type' in df.columns:
        plt.figure(figsize=fig_size)
        filing_counts = df['filing_type'].value_counts()
        plt.bar(filing_counts.index, filing_counts.values, color='#9b59b6')
        plt.xlabel('Filing Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Examples by Filing Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'filing_type_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: Saved: {output_dir}/filing_type_distribution.png")
    
    # 5. Top companies
    if 'source_ticker' in df.columns:
        plt.figure(figsize=fig_size)
        company_counts = df['source_ticker'].value_counts().head(10)
        plt.barh(company_counts.index, company_counts.values, color='#16a085')
        plt.xlabel('Number of Examples', fontsize=12)
        plt.ylabel('Company Ticker', fontsize=12)
        plt.title('Top 10 Companies by Example Count', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_companies.png'), dpi=150)
        plt.close()
        print(f"Saved: Saved: {output_dir}/top_companies.png")
    
    # 6. Needle position distribution
    def get_relative_position(span, haystack_len):
        if not span or not isinstance(span, list) or len(span) < 2:
            return None
        start = span[0]
        if haystack_len > 0:
            return start / haystack_len * 100
        return None
    
    df['relative_position'] = df.apply(
        lambda row: get_relative_position(row.get('needle_span'), row['haystack_length']),
        axis=1
    )
    positions = df['relative_position'].dropna()
    
    if len(positions) > 0:
        plt.figure(figsize=fig_size)
        plt.hist(positions, bins=20, edgecolor='black', alpha=0.7, color='#f39c12')
        plt.xlabel('Needle Position (% from start)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Needle Positions in Haystacks', fontsize=14, fontweight='bold')
        plt.axvline(positions.mean(), color='red', linestyle='--', label=f'Mean: {positions.mean():.1f}%')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'needle_position_distribution.png'), dpi=150)
        plt.close()
        print(f"Saved: Saved: {output_dir}/needle_position_distribution.png")
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def export_summary_report(df: pd.DataFrame, output_path: str = "edgar_niah_dataset/dataset_summary.txt"):
    """Export a text summary report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("EDGAR NIAH Dataset Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Examples: {len(df)}\n")
        f.write(f"Unique Filings: {df['doc_id'].nunique()}\n")
        f.write(f"Unique Companies: {df['source_ticker'].nunique()}\n\n")
        
        if 'metadata' in df.columns:
            df['method'] = df['metadata'].apply(
                lambda x: x.get('method', '') if isinstance(x, dict) else ''
            )
            f.write("Method Distribution:\n")
            method_counts = df['method'].value_counts()
            for method, count in method_counts.items():
                f.write(f"  {method}: {count}\n")
            f.write("\n")
        
        f.write("Filing Type Distribution:\n")
        filing_counts = df['filing_type'].value_counts()
        for ftype, count in filing_counts.items():
            f.write(f"  {ftype}: {count}\n")
        f.write("\n")
        
        df['haystack_length'] = df['haystack'].astype(str).apply(len)
        df['needle_length'] = df['needle'].astype(str).apply(len)
        
        f.write("Haystack Length Statistics:\n")
        f.write(f"  Mean: {df['haystack_length'].mean():.0f}\n")
        f.write(f"  Median: {df['haystack_length'].median():.0f}\n")
        f.write(f"  Min: {df['haystack_length'].min()}\n")
        f.write(f"  Max: {df['haystack_length'].max()}\n\n")
        
        f.write("Needle Length Statistics:\n")
        f.write(f"  Mean: {df['needle_length'].mean():.1f}\n")
        f.write(f"  Median: {df['needle_length'].median():.1f}\n")
        f.write(f"  Min: {df['needle_length'].min()}\n")
        f.write(f"  Max: {df['needle_length'].max()}\n")
    
    print(f"Saved: Summary report saved to: {output_path}")


def main():
    """Main analysis function."""
    print("Loading dataset...")
    try:
        df = load_dataset()
        print(f"Saved: Loaded {len(df)} examples\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run build_edgar_niah.py first to generate the dataset.")
        return
    
    # Run analyses
    basic_statistics(df)
    analyze_haystack_needle_lengths(df)
    analyze_needle_positions(df)
    
    # Show samples
    print("\n")
    show_sample_examples(df, n=2, method="extract")
    print("\n")
    show_sample_examples(df, n=2, method="insert")
    
    # Create visualizations
    print("\n")
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    create_visualizations(df)
    
    # Export summary
    export_summary_report(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nYou can also use pandas directly:")
    print("  import pandas as pd")
    print("  df = pd.read_json('edgar_niah_dataset/edgar_niah.jsonl', lines=True)")


if __name__ == "__main__":
    main()

