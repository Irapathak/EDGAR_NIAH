# EDGAR NIAH Dataset

## What Is This?

This is a Needle-In-A-Haystack (NIAH) evaluation dataset built from real SEC EDGAR financial filings. The dataset tests whether language models can find specific information hidden inside very long documents.

### The Basic Concept

Imagine you have a 220-page financial report (the "haystack") and you ask a question like "What were Apple's total assets as of September 27, 2025?" The answer (the "needle") is somewhere in those 220 pages. Can a language model find it?

That's what this dataset tests.

## What's In The Dataset?

### Dataset Structure

- **150 test examples** total
- Each example contains:
  - A **haystack**: Full SEC filing document (~220,000 characters = ~220 pages)
  - A **needle**: Specific fact or information to find (~1,500 characters)
  - A **query**: Natural question about the document (e.g., "What were AAPL's total assets?")
  - **Needle location**: Exact character positions where the answer appears
  - **Metadata**: Company, filing type, method, etc.

### Two Types of Examples

1. **Extractive (75 examples)**: Real sentences from the actual financial filings
   - Example: Balance sheet information, income statements, tax rates
   - Query: "What were AAPL's total assets as of September 27, 2025?"
   - Answer: The actual balance sheet section from the filing

2. **Insertive (75 examples)**: Synthetic codes inserted into the documents
   - Example: A unique identifier code placed somewhere in the text
   - Query: "What is the AAPL internal reporting code mentioned in this 10-K?"
   - Answer: The inserted code (e.g., "SECRET_CODE:ABC12345")

### Companies and Filings

- **5 companies**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **25 filings total**: Mix of 10-K (annual) and 10-Q (quarterly) reports
- **Date range**: 2024-2025 filings

## Files in This Project

### Essential Files

1. **build_edgar_niah.py** - Main script to generate the dataset
   - Downloads filings from SEC EDGAR
   - Creates NIAH examples
   - Outputs JSONL and CSV files

2. **analyze_edgar_niah.py** - Analysis and visualization script
   - Generates statistics about the dataset
   - Creates plots and charts
   - Exports summary reports

3. **view_dataset.py** - Human-readable viewer
   - Shows examples in readable format
   - Useful for understanding what's in the dataset

4. **requirements.txt** - Python dependencies

5. **edgar_niah_dataset/** - Output directory
   - **edgar_niah.jsonl** - Full dataset (for experiments)
   - **edgar_niah.csv** - Truncated dataset (for quick viewing)
   - **plots/** - Visualization charts
   - **dataset_summary.txt** - Summary statistics

## How The Dataset Was Created

1. **Download filings**: Script queries SEC EDGAR API for company filings
2. **Extract text**: Removes HTML and extracts clean text from filings
3. **Generate examples**:
   - **Extractive**: Selects real sentences that contain financial facts
   - **Insertive**: Inserts synthetic codes into random paragraphs
4. **Create queries**: Generates natural questions based on the needle content
5. **Track locations**: Records exact character positions for evaluation

## How To Use This Dataset To Test Language Models

### Step 1: Load The Dataset

```python
import json

# Load the JSONL file
examples = []
with open('edgar_niah_dataset/edgar_niah.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples")
```

### Step 2: Test A Language Model

For each example, you need to:

1. **Give the model the haystack + query**
2. **Get the model's answer**
3. **Compare to expected answer**
4. **Check if the answer contains the needle**

Here's a simple example:

```python
def test_model_on_example(model, example):
    """
    Test a language model on one NIAH example.
    
    Args:
        model: Your language model (OpenAI, Anthropic, etc.)
        example: One example from the dataset
    
    Returns:
        bool: True if model found the needle correctly
    """
    # Get the haystack and query
    haystack = example['haystack']
    query = example['query']
    expected_answer = example['expected_answer']
    
    # Create the prompt for your model
    prompt = f"""You are analyzing a financial document. Answer the following question based on the document.

Document:
{haystack}

Question: {query}

Answer:"""
    
    # Get model's response
    response = model.generate(prompt)  # Adjust to your model's API
    
    # Check if the response contains the expected answer
    # Simple check: see if key parts of the needle are in the response
    needle_key_parts = expected_answer[:100]  # First 100 chars of needle
    is_correct = needle_key_parts.lower() in response.lower()
    
    return is_correct, response
```

### Step 3: Evaluate Accuracy

Run all examples and calculate accuracy:

```python
def evaluate_model(model, examples):
    """Evaluate a model on all examples."""
    results = {
        'total': len(examples),
        'correct': 0,
        'extractive_correct': 0,
        'insertive_correct': 0,
        'extractive_total': 0,
        'insertive_total': 0
    }
    
    for example in examples:
        method = example['metadata']['method']
        
        if method == 'extract':
            results['extractive_total'] += 1
        else:
            results['insertive_total'] += 1
        
        is_correct, response = test_model_on_example(model, example)
        
        if is_correct:
            results['correct'] += 1
            if method == 'extract':
                results['extractive_correct'] += 1
            else:
                results['insertive_correct'] += 1
    
    # Calculate accuracies
    overall_acc = results['correct'] / results['total']
    extractive_acc = results['extractive_correct'] / results['extractive_total']
    insertive_acc = results['insertive_correct'] / results['insertive_total']
    
    print(f"Overall Accuracy: {overall_acc:.2%}")
    print(f"Extractive Accuracy: {extractive_acc:.2%}")
    print(f"Insertive Accuracy: {insertive_acc:.2%}")
    
    return results
```

### Step 4: Test With Different Model APIs

Here are examples for popular APIs:

#### OpenAI GPT-4

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def test_openai(example):
    haystack = example['haystack']
    query = example['query']
    
    prompt = f"""Analyze this financial document and answer the question.

Document:
{haystack[:100000]}...  # May need to truncate for context limits

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

#### Anthropic Claude

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-key")

def test_claude(example):
    haystack = example['haystack']
    query = example['query']
    
    prompt = f"""Analyze this financial document and answer the question.

Document:
{haystack[:100000]}...

Question: {query}

Answer:"""
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

#### Local Models (Llama, Mistral, etc.)

```python
from transformers import pipeline

generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

def test_local_model(example):
    haystack = example['haystack']
    query = example['query']
    
    prompt = f"""<s>[INST] Analyze this financial document and answer the question.

Document:
{haystack[:50000]}...

Question: {query}

Answer: [/INST]"""
    
    response = generator(prompt, max_length=1000, return_full_text=False)
    return response[0]['generated_text']
```

### Step 5: Advanced Evaluation Metrics

Beyond simple accuracy, you can measure:

1. **Exact Match**: Does the model's answer exactly match the expected answer?
2. **Contains Needle**: Does the answer contain the key information from the needle?
3. **Position Accuracy**: For extractive examples, did the model reference information from the correct location?
4. **Partial Credit**: Give credit if the model gets the main fact right even if formatting differs

```python
def evaluate_advanced(example, model_response):
    """More sophisticated evaluation."""
    expected = example['expected_answer']
    needle_span = example['needle_span']
    
    metrics = {
        'exact_match': model_response.strip() == expected.strip(),
        'contains_needle': expected[:50].lower() in model_response.lower(),
        'contains_key_numbers': extract_numbers(expected) == extract_numbers(model_response),
        'position_hint': check_if_model_references_correct_section(model_response, needle_span)
    }
    
    return metrics
```

## Context Window Considerations

Most language models have context limits:
- GPT-4: ~128K tokens
- Claude 3: ~200K tokens  
- Llama 2: ~4K tokens
- Mistral: ~32K tokens

**Important**: The haystacks in this dataset are ~220,000 characters, which is roughly:
- ~55,000 tokens (using 4 chars/token estimate)
- May exceed some model limits

**Solutions**:
1. **Truncate haystack**: Test with first N characters
2. **Chunk and retrieve**: Split haystack into chunks, retrieve relevant ones, then answer
3. **Test on longer-context models**: Use models with large context windows
4. **Subset analysis**: Test on examples with shorter haystacks

## What Makes This Dataset Useful?

1. **Real-world data**: Uses actual financial documents, not synthetic text
2. **Realistic length**: Documents are genuinely long (hundreds of pages)
3. **Natural queries**: Questions sound like real business questions
4. **Diverse content**: Mix of balance sheets, income statements, tax info, etc.
5. **Precise evaluation**: Exact character positions recorded for accurate evaluation

## Example Evaluation Workflow

```python
# 1. Load dataset
examples = load_dataset('edgar_niah_dataset/edgar_niah.jsonl')

# 2. Filter if needed (e.g., only extractive, only certain companies)
extractive = [e for e in examples if e['metadata']['method'] == 'extract']

# 3. Test on subset (start small)
test_set = extractive[:10]

# 4. Run evaluation
results = []
for example in test_set:
    model_answer = test_your_model(example)
    is_correct = evaluate_answer(example, model_answer)
    results.append({
        'id': example['id'],
        'correct': is_correct,
        'model_answer': model_answer
    })

# 5. Calculate metrics
accuracy = sum(r['correct'] for r in results) / len(results)
print(f"Accuracy: {accuracy:.2%}")
```

## Tips For Good Results

1. **Start small**: Test on 5-10 examples first
2. **Check context limits**: Make sure your model can handle the haystack length
3. **Use retrieval**: For very long documents, consider RAG (Retrieval Augmented Generation)
4. **Analyze failures**: Look at what the model gets wrong - is it finding the right section but misreading numbers?
5. **Compare methods**: Test both extractive and insertive examples separately
6. **Position analysis**: Check if failures correlate with needle position (beginning vs middle vs end)

## Understanding The Results

**Good performance indicators**:
- High accuracy on extractive examples = model can find real financial facts
- High accuracy on insertive examples = model can find arbitrary tokens (harder)
- Consistent performance across companies = generalizes well
- Better on shorter haystacks = context length matters

**Common failure modes**:
- Model answers from wrong section (found similar info elsewhere)
- Model makes up numbers (hallucination)
- Model finds right section but extracts wrong values
- Model doesn't search long enough (early stopping)

## Summary

This dataset provides 150 real-world test cases to evaluate if language models can:
1. Search through very long documents (220K+ characters)
2. Find specific financial information
3. Answer natural business questions accurately

Use it to benchmark your models, compare different approaches (full context vs. RAG), and understand model limitations with long documents.

