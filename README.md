# 📄 Reviewer Recommendation System

An intelligent system that recommends suitable reviewers for research papers using semantic similarity matching powered by LLMs and sentence transformers.

---

## 🎯 Overview

This system automatically extracts metadata from research papers and matches them with the most suitable reviewers based on their research profiles. It uses advanced NLP techniques including:

- **Groq LLM API** for intelligent metadata extraction
- **Sentence Transformers** for semantic embeddings
- **Cosine Similarity** for reviewer matching
- **Streamlit** for interactive web interface

---

## ✨ Features

- 📥 **PDF Upload & Processing**: Extract text from research papers automatically
- 🤖 **LLM-Powered Extraction**: Extract title, authors, abstract, keywords, and summary
- 🧠 **Semantic Matching**: Use embeddings to find similar research profiles
- 🎯 **Weighted Scoring**: Customizable weights for different fields (abstract, title, keywords, summary)
- 📊 **Interactive Dashboard**: User-friendly Streamlit interface
- 💾 **Efficient Storage**: Pre-computed embeddings for fast recommendations

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Research Papers                    │
│                         (PDF files)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: Feature Extraction                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Script: extract_papers.py                           │   │
│  │  • Extract text from PDFs (PyMuPDF)                  │   │
│  │  • LLM extraction (Groq API)                         │   │
│  │  • Output: Individual CSV files per paper            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: Data Consolidation                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Script: merge_csvs.py                               │   │
│  │  • Merge all paper CSVs                              │   │
│  │  • Group by author                                   │   │
│  │  • Combine multiple papers per author                │   │
│  │  • Output: merged_authors.csv                        │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: Data Cleaning                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Script: clean_authors.py                            │   │
│  │  • Remove noise patterns                             │   │
│  │  • Filter invalid authors                            │   │
│  │                                                      │   │
│  │  • Output: merged_authors_cleaned.csv                │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 4: Embedding Generation                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Script: generate_embeddings.py                      │   │
│  │  • Load sentence transformer model                   │   │
│  │  • Generate embeddings for:                          │   │
│  │    - Titles                                          │   │
│  │    - Abstracts                                       │   │
│  │    - Keywords                                        │   │
│  │    - Summaries                                       │   │
│  │  • Output: reviewer_embeddings.npz                   │   │
│  │           reviewer_metadata.csv                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 5: Reviewer Recommendation                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Script: app.py (Streamlit)                          │   │
│  │  • Upload new paper PDF                              │   │
│  │  • Extract metadata with LLM                         │   │
│  │  • Generate paper embeddings                         │   │
│  │  • Calculate similarity scores                       │   │
│  │  • Return top-K reviewers                            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   OUTPUT:    │
                    │ Top Reviewers│
                    └──────────────┘
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11 or higher
- Groq API Key ([Get it here](https://console.groq.com))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/reviewer-recommendation-system.git
cd reviewer-recommendation-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
sentence-transformers
scikit-learn>=1.3.0
PyMuPDF>=1.23.0
groq>=0.4.0
tqdm>=4.65.0
```

### Step 3: Set Up API Keys

Create a `.env` file or set environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Or on Windows:
```cmd
set GROQ_API_KEY=your_groq_api_key_here
```

---

## 🔍 Feature Extraction Pipeline

The feature extraction pipeline consists of 4 sequential scripts that transform raw PDFs into embeddings ready for matching.

### **Script 1: `llm_extraction.py` - Paper Metadata Extraction**

**Purpose**: Extract metadata from individual research papers using Groq LLM.

**What it does:**
- Reads PDF files from specified directory
- Extracts text using PyMuPDF
- Sends text to Groq LLM (llama-3.1-8b-instant)
- Extracts structured metadata:
  - Title
  - Authors (list)
  - Abstract
  - Keywords (list)
  - Summary (LLM-generated)
- Saves each paper as a separate CSV file

**Key Features:**
- ✅ **Automatic author detection** from folder structure
- ✅ **Chunked processing** for large papers (10,000 chars per chunk)
- ✅ **Multi-chunk merging** using voting and length-based selection
- ✅ **Main author validation** - ensures folder author is included in author list

**Configuration:**
```python
RAW_DATA_DIR = "path/to/pdfs"           # Input: PDF folder
OUTPUT_DIR = "path/to/output"           # Output: CSV folder
MODEL_NAME = "llama-3.1-8b-instant"     # Groq model
TEMPERATURE = 0.1                        # Low temp for consistency
MAX_TOKENS = 2048                        # Response length
```

**Output Format (CSV):**
```csv
Title,Author,Abstract,Keywords,Summary
"Paper Title","Author Name","Abstract text","keyword1, keyword2","Summary text"
```

**Usage:**
```bash
# Set your API key
export GROQ_API_KEY="gsk_..."

# Run extraction
python llm-extraction.py
```

**Expected Output:**
```
🚀 Starting Research Paper CSV Extraction (Groq API)
📌 Main author will be extracted from folder name
📌 If not found in extracted authors, it will be added automatically

📂 Processing folder: C:\...\Rudresh dwivedi
   👤 Main Author (from folder): Rudresh dwivedi
   Found 5 PDFs

   🔍 Processing: paper1.pdf
      📄 Processing chunk 1/2
      📄 Processing chunk 2/2
      ✅ Extracted metadata:
         Title: Deep Learning for NLP Applications...
         Authors: Rudresh Dwivedi, John Doe, Jane Smith
         Keywords: 8 found
         Abstract length: 1245
         Summary length: 523
      💾 Saved: Deep_Learning_for_NLP_Applications.csv

✅ PROCESSING COMPLETE!
   📊 Total PDFs processed: 5
   ❌ Failed: 0
   📁 CSV files saved to: C:\...\AuthorProfiles
```

---

### **Script 2: `combine_all_csv.py` - Data Consolidation**

**Purpose**: Merge individual paper CSVs and group by author.

**What it does:**
- Scans output directory for all CSV files
- Loads and merges into single DataFrame
- Groups papers by author name
- Combines multiple papers per author using separators
- Creates comprehensive author profiles

**Combination Strategy:**
- Titles: Separated by ` ||| `
- Abstracts: Separated by ` ||| `
- Keywords: Deduplicated and comma-separated
- Summaries: Separated by ` ||| `

**Configuration:**
```python
INPUT_DIR = "path/to/individual/csvs"   # From extract_papers.py
OUTPUT_FILE = "merged_authors.csv"       # Combined output
```

**Output Format:**
```csv
Author,Title,Abstract,Keywords,Summary
"Author Name","Title1 ||| Title2","Abstract1 ||| Abstract2","kw1, kw2, kw3","Summary1 ||| Summary2"
```

**Usage:**
```bash
python combine_all_csv.py
```

**Expected Output:**
```
🚀 STARTING CSV MERGE & AUTHOR COMBINATION PIPELINE

STEP 1: Merging CSV files...
📂 Reading CSV files from: C:\...\Papers_CSV
✅ Found 47 CSV files

   ✓ Loaded: paper1.csv (1 rows)
   ✓ Loaded: paper2.csv (3 rows)
   ...

✅ Merged 47 files into DataFrame
   Total rows: 156

STEP 2: Combining by author...
   Total rows before filtering: 156
   Total rows after filtering empty authors: 156
✅ Combined into 89 unique authors

STEP 3: Saving combined data...
💾 Saved combined data to: merged_authors.csv
   File size: 2.45 MB
   ✅ File created successfully!

✅ PIPELINE COMPLETE!
   Total unique authors: 89
   Total papers processed: 156
```

---

### **Script 3: `data_clean.py` - Data Cleaning**

**Purpose**: Remove noise and invalid entries from author data.

**What it does:**
- Detects and removes noisy author entries:
  - ❌ Generic terms: "unknown", "et al", "author"
  - ❌ Bracketed numbers: [10], [11]
  - ❌ Section headers: "ABSTRACT", "INTRODUCTION"
  - ❌ URLs, emails, DOIs
  - ❌ Copyright statements
  - ❌ Single letters or initials only
  - ❌ Sentences/phrases (contains action words)
  - ❌ Names with special patterns (e.g., "BACZY / NSKI")
- Normalizes remaining author names
- Logs all removed entries for review

**Noise Detection Categories:**

1. **Exact Matches** (70+ patterns)
   - "unknown", "et al", "corresponding author"
   
2. **Pattern-Based Removal**
   - Regex patterns for brackets, URLs, emails
   - Special character density checks
   
3. **Sentence Detection**
   - Action words: "estimating", "training", "using"
   - Technical terms: "algorithm", "model", "learning"
   - Multiple common English words

**Configuration:**
```python
INPUT_CSV = "merged_authors.csv"            # From merge_csvs.py
OUTPUT_CSV = "merged_authors_cleaned.csv"   # Cleaned output
REMOVED_AUTHORS_CSV = "removed_authors_log.csv"  # Audit log
```

**Usage:**
```bash
python data_clean.py
```

**Expected Output:**
```
🚀 CSV Author Data Cleaner

STEP 1: PREVIEW MODE
=====================================================
BEFORE CLEANING:
Total rows: 89
Unique authors: 89

🔍 Identifying noisy authors...

⚠️  Found 12 noisy author entries (13.5%)

📋 ALL 12 UNIQUE AUTHORS THAT WILL BE REMOVED:
  1. [10]
  2. [UNKNOWN AUTHOR]
  3. BACZY / NSKI
  4. et al
  5. FAOHall, L. O.
  6. Estimating values using sample data
  7. L. O.
  8. MECSAuthor
  9. N/A
  10. Unknown
  11. Van der Heijden
  12. corresponding author

AFTER CLEANING:
Total rows: 77
Unique authors: 77
Rows removed: 12 (13.5%)
Rows retained: 77 (86.5%)

⚠️  PREVIEW MODE - No file saved

💡 Review the removed authors list above carefully!

❓ Do you want to save the cleaned CSV? (yes/no): yes

STEP 2: SAVING CLEANED CSV
✅ Cleaned CSV saved to: merged_authors_cleaned.csv
✅ Removed authors log saved to: removed_authors_log.csv

✅ Process complete!
```

---

### **Script 4: `embeddings.py` - Embedding Generation**

**Purpose**: Convert text data into semantic embeddings for similarity matching.

**What it does:**
- Loads cleaned author data
- Initializes sentence transformer model
- Generates embeddings for each field:
  - Title embeddings (384-dim vectors)
  - Abstract embeddings
  - Keywords embeddings
  - Summary embeddings
- Saves embeddings in compressed format (.npz)
- Creates metadata file with author mappings

**Model Options:**
- `all-MiniLM-L6-v2` (default): 384 dims, fast, good quality
- `all-mpnet-base-v2`: 768 dims, slower, better quality
- `all-MiniLM-L12-v2`: 384 dims, balanced

**Configuration:**
```python
INPUT_CSV = "merged_authors_cleaned.csv"    # From clean_authors.py
OUTPUT_DIR = "embeddings/"                   # Output folder
MODEL_NAME = "all-MiniLM-L6-v2"             # Embedding model
```

**Output Files:**

1. **`reviewer_embeddings.npz`** (compressed numpy arrays)
   - `title_embeddings`: (N, 384) array
   - `abstract_embeddings`: (N, 384) array
   - `keywords_embeddings`: (N, 384) array
   - `summary_embeddings`: (N, 384) array
   - `reviewer_ids`: (N,) array
   - `model_name`: string
   - `embedding_dim`: int

2. **`reviewer_metadata.csv`**
   ```csv
   reviewer_id,Author
   0,"Author Name 1"
   1,"Author Name 2"
   ```

**Usage:**
```bash
python embeddings.py
```

**Expected Output:**
```
🚀 REVIEWER EMBEDDINGS GENERATION PIPELINE

STEP 1: Loading reviewer data...
✅ Loaded 77 reviewers
   Columns: ['Author', 'Title', 'Abstract', 'Keywords', 'Summary']

STEP 2: Loading embedding model...
📦 Loading embedding model: all-MiniLM-L6-v2
✅ Model loaded! Embedding dimension: 384

STEP 3: Generating embeddings...
🔄 Generating embeddings for 77 reviewers...
Processing reviewers: 100%|████████████| 77/77 [00:15<00:00, 5.13it/s]
✅ Embeddings generated!
   Shape: (77, 384)
   Dimension: 384

STEP 4: Saving embeddings...
💾 Saving embeddings to: embeddings/reviewer_embeddings.npz
✅ Embeddings saved!
   File size: 0.12 MB

STEP 5: Saving metadata...
💾 Saving metadata to: embeddings/reviewer_metadata.csv
✅ Metadata saved!

STEP 6: Verifying saved embeddings...
📂 Loading embeddings from: embeddings/reviewer_embeddings.npz
✅ Embeddings loaded!
   Number of reviewers: 77
   Embedding dimension: 384
   Model used: all-MiniLM-L6-v2

✅ PIPELINE COMPLETE!
   Embeddings file: embeddings/reviewer_embeddings.npz
   Metadata file: embeddings/reviewer_metadata.csv
   Total reviewers: 77
   Embedding dimension: 384
   Model: all-MiniLM-L6-v2
```

---

## 📊 Pipeline Summary

| Step | Script | Input | Output | Time |
|------|--------|-------|--------|------|
| 1 | `extract_papers.py` | PDFs | Individual CSVs | ~30s/paper |
| 2 | `merge_csvs.py` | Individual CSVs | `merged_authors.csv` | ~5s |
| 3 | `clean_authors.py` | `merged_authors.csv` | `merged_authors_cleaned.csv` | ~2s |
| 4 | `generate_embeddings.py` | `merged_authors_cleaned.csv` | `.npz` + metadata | ~20s for 100 authors |

**Total Pipeline Time**: ~1-2 hours for 100 papers (depends on Groq API speed)

---



### Using the Web Interface

1. **Upload a PDF**: Click "Upload PDF" and select a research paper
2. **Set Top-K**: Choose how many reviewers to recommend (1-10)
3. **Click "Run Recommendation"**: System will:
   - Extract paper metadata
   - Generate embeddings
   - Calculate similarity scores
   - Display top matches
4. **Download Results**: Export recommendations as CSV

---

## ⚙️ Configuration

### Embedding Weights

Customize field importance in `app.py`:

```python
WEIGHTS = {
    "abstract": 0.40,   # Most important
    "title": 0.30,      # Second most important
    "keywords": 0.20,   # Moderate importance
    "summary": 0.10     # Least important
}
```

### Model Selection

Change embedding model in `generate_embeddings.py`:

```python
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast (default)
# MODEL_NAME = 'all-mpnet-base-v2'  # Better quality
# MODEL_NAME = 'all-MiniLM-L12-v2'  # Balanced
```




