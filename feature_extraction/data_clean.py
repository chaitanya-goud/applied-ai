import pandas as pd
import re
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
INPUT_CSV = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\merged_authors.csv"
OUTPUT_CSV = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\merged_authors_cleaned.csv"
REMOVED_AUTHORS_CSV = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\removed_authors_log.csv"

# =========================
# NOISE DETECTION PATTERNS
# =========================
def is_noisy_author(author_name: str) -> bool:
    """
    Detect if an author name is likely noise/garbage data.
    Returns True if the name should be removed.
    """
    if not author_name or not isinstance(author_name, str):
        return True
    
    author_name = author_name.strip()
    
    # Empty or too short
    if len(author_name) < 2:
        return True
    
    # Convert to lowercase for pattern matching
    author_lower = author_name.lower()
    
    # ========================================
    # EXACT MATCHES - Specific noise entries
    # ========================================
    
    exact_noise = [
        '[unknown]',
        'unknown',
        'unknown author',
        'et al',
        'et al.',
        'et.al',
        'etal',
        'author',
        'authors',
        'n/a',
        'na',
        'none',
        'null',
        'mecsauthor',
        'corresponding author',
        'na li',
        'van der heijden',
        'van der horst',
        'van der linden',
        'van der velde',
        'van der weijden',
        'faohall',
        'l. o.',
        'jena, d.',
        'd. jena',
        'baczy',
        'nski',
        'baczy / nski',
        'baczy/nski',
        'a. author',
        'b. author',
        'c. author'
    ]
    
    # Check exact match
    if author_lower in exact_noise:
        return True
    
    # Remove if it contains "author" as a standalone word (but not in middle of name)
    author_words = author_lower.split()
    if 'author' in author_words or 'authors' in author_words:
        return True
    
    # ========================================
    # PATTERN-BASED REMOVAL
    # ========================================
    
    # Remove if it's JUST numbers in brackets like [10], [11]
    if re.match(r'^\[\d+\]$', author_name.strip()):
        return True
    
    # Remove if it starts with "et al" (including variations)
    if author_lower.startswith('et al'):
        return True
    
    # Remove if ends with "et al"
    if author_lower.endswith('et al') or author_lower.endswith('et al.'):
        return True
    
    # Remove entries like "FAOHall, L. O." - contains abbreviation + comma pattern
    if re.search(r'[A-Z]{3,}.*,\s*[A-Z]\.\s*[A-Z]\.', author_name):
        return True
    
    # Remove entries with "/" in the middle (like "BACZY / NSKI")
    if '/' in author_name:
        return True
    
    # Remove obvious section headers (ALL CAPS and common words)
    section_headers = ['abstract', 'introduction', 'conclusion', 'references', 
                       'acknowledgment', 'acknowledgments', 'bibliography', 'appendix', 'contents',
                       'keywords', 'summary']
    if author_name.isupper() and len(author_name) > 5 and author_lower in section_headers:
        return True
    
    # Remove if it's clearly an email
    if '@' in author_name and '.' in author_name:
        return True
    
    # Remove if it's a URL
    if author_lower.startswith(('http://', 'https://', 'www.')):
        return True
    
    # Remove if it contains "department of" or "university of"
    if 'department of' in author_lower or 'university of' in author_lower:
        return True
    
    # Remove DOI patterns
    if author_lower.startswith('doi:') or 'doi.org' in author_lower:
        return True
    
    # Remove copyright statements
    if 'copyright' in author_lower or '©' in author_name:
        return True
    
    # Remove if it's ONLY special characters or numbers
    if not any(c.isalpha() for c in author_name):
        return True
    
    # Remove extremely long entries (likely abstracts or paragraphs)
    if len(author_name) > 150:
        return True
    
    # Remove if it has more than 70% special characters
    special_chars = sum(1 for c in author_name if not c.isalnum() and not c.isspace())
    if len(author_name) > 5 and special_chars > len(author_name) * 0.7:
        return True
    
    # Remove entries that are JUST initials with periods (like "L. O." alone)
    if re.match(r'^[A-Z]\.\s*[A-Z]\.$', author_name.strip()):
        return True
    
    # Remove single letter entries
    if len(author_name.strip()) == 1:
        return True
    
    # ========================================
    # SENTENCE/PHRASE DETECTION
    # ========================================
    
    # Remove entries that are clearly sentences or phrases (not names)
    # These contain action words or technical terms
    sentence_keywords = [
        'estimating', 'training', 'values', 'sample', 'data',
        'using', 'based', 'approach', 'method', 'analysis',
        'system', 'model', 'algorithm', 'framework', 'application',
        'study', 'research', 'evaluation', 'comparison', 'review',
        'detection', 'classification', 'prediction', 'optimization',
        'learning', 'neural', 'network', 'deep', 'machine',
        'performance', 'efficient', 'improved', 'novel', 'proposed'
    ]
    
    # If name is longer than 40 characters and contains any of these keywords, it's likely a title/sentence
    if len(author_name) > 40:
        for keyword in sentence_keywords:
            if keyword in author_lower:
                return True
    
    # Remove if it contains multiple common English words (likely a phrase/sentence)
    common_words = ['the', 'and', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'is', 'are', 'was', 'were']
    word_count = sum(1 for word in author_lower.split() if word in common_words)
    if word_count >= 2:  # If it has 2+ common words, it's probably not a name
        return True
    
    # Remove if name has more than 8 words (likely a sentence)
    if len(author_name.split()) > 8:
        return True
    
    return False

# =========================
# CLEANING FUNCTIONS
# =========================
def clean_author_name(author_name: str) -> str:
    """
    Clean up author name by removing extra whitespace and normalizing.
    """
    if not isinstance(author_name, str):
        return ""
    
    # Remove extra whitespace
    author_name = re.sub(r'\s+', ' ', author_name.strip())
    
    # Remove leading/trailing special characters (but keep periods for initials)
    author_name = re.sub(r'^[^\w\s.]+|[^\w\s.]+$', '', author_name)
    
    return author_name.strip()

def clean_csv(input_path: str, output_path: str, removed_log_path: str, preview_only: bool = False):
    """
    Clean the CSV by removing noisy author entries.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save cleaned CSV
        removed_log_path: Path to save removed authors log
        preview_only: If True, only show statistics without saving
    """
    try:
        print(f"📂 Reading CSV from: {input_path}")
        df = pd.read_csv(input_path, encoding='utf-8')
        
        print(f"✅ Loaded {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}\n")
        
        # Check if 'Author' column exists
        if 'Author' not in df.columns:
            print("❌ 'Author' column not found in CSV!")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Statistics before cleaning
        original_count = len(df)
        unique_authors_before = df['Author'].nunique()
        
        print("=" * 60)
        print("BEFORE CLEANING:")
        print("=" * 60)
        print(f"Total rows: {original_count}")
        print(f"Unique authors: {unique_authors_before}")
        
        # Identify noisy rows
        print("\n🔍 Identifying noisy authors...")
        df['is_noise'] = df['Author'].apply(is_noisy_author)
        noisy_count = df['is_noise'].sum()
        
        print(f"\n⚠️  Found {noisy_count} noisy author entries ({noisy_count/len(df)*100:.1f}%)")
        
        # Get all removed authors (unique)
        removed_authors = df[df['is_noise']]['Author'].unique()
        removed_df = pd.DataFrame({'Removed_Author': sorted(removed_authors)})
        
        # Show ALL removed authors
        if noisy_count > 0:
            print("\n" + "=" * 60)
            print(f"📋 ALL {len(removed_authors)} UNIQUE AUTHORS THAT WILL BE REMOVED:")
            print("=" * 60)
            for i, author in enumerate(sorted(removed_authors), 1):
                print(f"  {i}. {author}")
        
        # Clean the dataframe
        df_cleaned = df[~df['is_noise']].copy()
        df_cleaned = df_cleaned.drop(columns=['is_noise'])
        
        # Clean remaining author names
        df_cleaned['Author'] = df_cleaned['Author'].apply(clean_author_name)
        
        # Remove any empty authors after cleaning
        df_cleaned = df_cleaned[df_cleaned['Author'].str.len() > 0]
        
        # Statistics after cleaning
        cleaned_count = len(df_cleaned)
        unique_authors_after = df_cleaned['Author'].nunique()
        removed_count = original_count - cleaned_count
        
        print("\n" + "=" * 60)
        print("AFTER CLEANING:")
        print("=" * 60)
        print(f"Total rows: {cleaned_count}")
        print(f"Unique authors: {unique_authors_after}")
        print(f"Rows removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        print(f"Rows retained: {cleaned_count} ({cleaned_count/original_count*100:.1f}%)")
        
        print(f"\nSample clean authors (first 20):")
        for i, author in enumerate(df_cleaned['Author'].head(20), 1):
            print(f"  {i}. {author}")
        
        if preview_only:
            print("\n" + "=" * 60)
            print("⚠️  PREVIEW MODE - No file saved")
            print("=" * 60)
            print("\n💡 Review the removed authors list above carefully!")
            print("💡 If any valid authors were removed, let me know and I'll adjust.")
        else:
            # Save cleaned CSV
            df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\n✅ Cleaned CSV saved to: {output_path}")
            
            # Save removed authors log
            removed_df.to_csv(removed_log_path, index=False, encoding='utf-8')
            print(f"✅ Removed authors log saved to: {removed_log_path}")
        
        print("=" * 60)
        
        return df_cleaned, removed_df
        
    except FileNotFoundError:
        print(f"❌ File not found: {input_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("🚀 CSV Author Data Cleaner\n")
    print("📌 This will remove all specified noise patterns")
    print("📌 All removed authors will be logged and displayed\n")
    
    # First, preview the cleaning (no file saved)
    print("=" * 60)
    print("STEP 1: PREVIEW MODE")
    print("=" * 60)
    clean_csv(INPUT_CSV, OUTPUT_CSV, REMOVED_AUTHORS_CSV, preview_only=True)
    
    # Ask user for confirmation
    print("\n" + "=" * 60)
    response = input("\n❓ Do you want to save the cleaned CSV? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n" + "=" * 60)
        print("STEP 2: SAVING CLEANED CSV")
        print("=" * 60)
        clean_csv(INPUT_CSV, OUTPUT_CSV, REMOVED_AUTHORS_CSV, preview_only=False)
        print("\n✅ Process complete!")
        print(f"\n📄 Check '{REMOVED_AUTHORS_CSV}' for the full list of removed authors")
    else:
        print("\n❌ Cleaning cancelled. No file saved.")
        print("\n💡 If you need to adjust which authors to remove, let me know!")