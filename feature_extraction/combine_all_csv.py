import os
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# =========================
# CONFIGURATION
# =========================
INPUT_DIR = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\Papers_CSV"
OUTPUT_FILE = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\merged_authors.csv"

# =========================
# MERGE ALL CSVs
# =========================
def merge_all_csvs(input_dir: str) -> pd.DataFrame:
    """
    Read all CSV files from the directory and merge them into one DataFrame.
    """
    print(f"📂 Reading CSV files from: {input_dir}\n")
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        print(f"❌ ERROR: Directory does not exist: {input_dir}")
        return pd.DataFrame()
    
    print(f"✅ Directory exists")
    
    # List all files in directory
    all_items = os.listdir(input_dir)
    print(f"   Total items in directory: {len(all_items)}")
    
    # Find CSV files
    all_files = list(Path(input_dir).glob("*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {input_dir}")
        print(f"   Files in directory: {all_items[:10]}")  # Show first 10 items
        return pd.DataFrame()
    
    print(f"✅ Found {len(all_files)} CSV files\n")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            print(f"   ✓ Loaded: {file.name} ({len(df)} rows)")
            print(f"      Columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            print(f"   ✗ Failed to load {file.name}: {e}")
    
    if not dfs:
        print("❌ No valid CSV files could be loaded")
        return pd.DataFrame()
    
    # Combine all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Merged {len(dfs)} files into DataFrame")
    print(f"   Total rows: {len(merged_df)}")
    print(f"   Columns: {list(merged_df.columns)}")
    print(f"   First few rows:\n{merged_df.head()}\n")
    
    return merged_df

# =========================
# COMBINE BY AUTHOR
# =========================
def combine_by_author(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by author and combine all their titles, abstracts, keywords, and summaries.
    """
    print("🔄 Combining research papers by author...\n")
    
    # Check for missing columns
    required_cols = ['Author', 'Title', 'Abstract', 'Keywords', 'Summary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Remove rows with missing authors
    print(f"   Total rows before filtering: {len(df)}")
    df = df[df['Author'].notna() & (df['Author'] != '')]
    print(f"   Total rows after filtering empty authors: {len(df)}")
    
    if df.empty:
        print("❌ No valid data after filtering")
        return pd.DataFrame()
    
    # Fill NaN values with empty strings
    df['Title'] = df['Title'].fillna('')
    df['Abstract'] = df['Abstract'].fillna('')
    df['Keywords'] = df['Keywords'].fillna('')
    df['Summary'] = df['Summary'].fillna('')
    
    # Group by author
    grouped = df.groupby('Author', as_index=False).agg({
        'Title': lambda x: ' ||| '.join(filter(None, x.unique())),
        'Abstract': lambda x: ' ||| '.join(filter(None, x.unique())),
        'Keywords': lambda x: ', '.join(filter(None, set(', '.join(str(k) for k in x).split(', ')))),
        'Summary': lambda x: ' ||| '.join(filter(None, x.unique()))
    })
    
    # Reorder columns (removed Paper_Count)
    grouped = grouped[['Author', 'Title', 'Abstract', 'Keywords', 'Summary']]
    
    print(f"✅ Combined into {len(grouped)} unique authors")
    print(f"\n📊 Author Statistics:")
    print(f"   Total unique authors: {len(grouped)}\n")
    
    return grouped

# =========================
# SAVE COMBINED DATA
# =========================
def save_combined_data(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the combined author data to a CSV file.
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"   Created directory: {output_dir}")
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 Saved combined data to: {output_file}")
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024*1024)
            print(f"   File size: {file_size:.2f} MB")
            print(f"   ✅ File created successfully!")
        else:
            print(f"   ⚠️ Warning: File may not have been created")
        
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        import traceback
        traceback.print_exc()

# =========================
# DISPLAY SAMPLE
# =========================
def display_sample(df: pd.DataFrame, n: int = 3) -> None:
    """
    Display sample of the combined data.
    """
    print(f"\n{'='*80}")
    print(f"📋 SAMPLE DATA (First {min(n, len(df))} authors)")
    print(f"{'='*80}\n")
    
    for idx, row in df.head(n).iterrows():
        print(f"Author #{idx+1}: {row['Author']}")
        
        titles = str(row['Title'])
        title_count = len(titles.split('|||'))
        print(f"   Titles: {title_count} combined")
        if len(titles) > 100:
            print(f"      Preview: {titles[:100]}...")
        else:
            print(f"      Preview: {titles}")
        
        keywords = str(row['Keywords'])
        if len(keywords) > 100:
            print(f"   Keywords: {keywords[:100]}...")
        else:
            print(f"   Keywords: {keywords}")
        
        abstracts = str(row['Abstract'])
        abstract_count = len(abstracts.split('|||'))
        print(f"   Abstracts: {abstract_count} combined (total {len(abstracts)} chars)")
        
        summaries = str(row['Summary'])
        summary_count = len(summaries.split('|||'))
        print(f"   Summaries: {summary_count} combined (total {len(summaries)} chars)")
        print()

# =========================
# MAIN PIPELINE
# =========================
def main():
    """
    Main pipeline to merge CSVs and combine by author.
    """
    print("="*80)
    print("🚀 STARTING CSV MERGE & AUTHOR COMBINATION PIPELINE")
    print("="*80)
    print()
    
    try:
        # Step 1: Merge all CSV files
        print("STEP 1: Merging CSV files...")
        merged_df = merge_all_csvs(INPUT_DIR)
        
        if merged_df.empty:
            print("❌ No data to process. Exiting.")
            return
        
        print(f"\n{'='*80}\n")
        
        # Step 2: Combine by author
        print("STEP 2: Combining by author...")
        combined_df = combine_by_author(merged_df)
        
        if combined_df.empty:
            print("❌ No combined data. Exiting.")
            return
        
        print(f"\n{'='*80}\n")
        
        # Step 3: Save combined data
        print("STEP 3: Saving combined data...")
        save_combined_data(combined_df, OUTPUT_FILE)
        
        print(f"\n{'='*80}\n")
        
        # Step 4: Display sample
        print("STEP 4: Displaying sample...")
        display_sample(combined_df)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        print(f"   Input: {INPUT_DIR}")
        print(f"   Output: {OUTPUT_FILE}")
        print(f"   Total unique authors: {len(combined_df)}")
        print(f"   Total papers processed: {merged_df.shape[0]}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()