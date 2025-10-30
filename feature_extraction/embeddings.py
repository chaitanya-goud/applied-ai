import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
INPUT_CSV = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\merged_authors_cleaned.csv"
OUTPUT_DIR = r"C:\Users\admin\Documents\4.1\applied ai\Papers_CSV\embeddings"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "reviewer_embeddings.npz")
METADATA_FILE = os.path.join(OUTPUT_DIR, "reviewer_metadata.csv")

# Embedding model (choose one):
# - 'all-MiniLM-L6-v2': Fast, 384 dims, good quality
# - 'all-mpnet-base-v2': Slower, 768 dims, better quality
# - 'all-MiniLM-L12-v2': Balance, 384 dims
MODEL_NAME = 'all-MiniLM-L6-v2'

# =========================
# LOAD EMBEDDING MODEL
# =========================
def load_model():
    """Load sentence transformer model"""
    print(f"📦 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model

# =========================
# GENERATE EMBEDDINGS
# =========================
def generate_embeddings(df, model):
    """
    Generate embeddings for all reviewers
    Returns: Dictionary with embedding arrays
    """
    print(f"\n🔄 Generating embeddings for {len(df)} reviewers...")
    
    # Initialize lists to store embeddings
    title_embeddings = []
    abstract_embeddings = []
    keywords_embeddings = []
    summary_embeddings = []
    reviewer_ids = []
    
    # Process each reviewer
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviewers"):
        reviewer_ids.append(idx)
        
        # Generate embeddings for each field
        # Handle empty strings gracefully
        title_text = str(row['Title']) if pd.notna(row['Title']) and row['Title'] else "No title"
        abstract_text = str(row['Abstract']) if pd.notna(row['Abstract']) and row['Abstract'] else "No abstract"
        keywords_text = str(row['Keywords']) if pd.notna(row['Keywords']) and row['Keywords'] else "No keywords"
        summary_text = str(row['Summary']) if pd.notna(row['Summary']) and row['Summary'] else "No summary"
        
        # Encode texts to embeddings
        title_emb = model.encode(title_text, convert_to_numpy=True)
        abstract_emb = model.encode(abstract_text, convert_to_numpy=True)
        keywords_emb = model.encode(keywords_text, convert_to_numpy=True)
        summary_emb = model.encode(summary_text, convert_to_numpy=True)
        
        # Append to lists
        title_embeddings.append(title_emb)
        abstract_embeddings.append(abstract_emb)
        keywords_embeddings.append(keywords_emb)
        summary_embeddings.append(summary_emb)
    
    # Convert lists to numpy arrays
    embeddings = {
        'title': np.array(title_embeddings),
        'abstract': np.array(abstract_embeddings),
        'keywords': np.array(keywords_embeddings),
        'summary': np.array(summary_embeddings),
        'reviewer_ids': np.array(reviewer_ids)
    }
    
    print(f"✅ Embeddings generated!")
    print(f"   Shape: {embeddings['title'].shape}")
    print(f"   Dimension: {embeddings['title'].shape[1]}")
    
    return embeddings

# =========================
# SAVE EMBEDDINGS TO NPZ
# =========================
def save_embeddings(embeddings, output_file):
    """
    Save embeddings to compressed .npz file
    """
    print(f"\n💾 Saving embeddings to: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save as compressed npz file
    np.savez_compressed(
        output_file,
        title_embeddings=embeddings['title'],
        abstract_embeddings=embeddings['abstract'],
        keywords_embeddings=embeddings['keywords'],
        summary_embeddings=embeddings['summary'],
        reviewer_ids=embeddings['reviewer_ids'],
        model_name=MODEL_NAME,
        embedding_dim=embeddings['title'].shape[1]
    )
    
    # Check file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Embeddings saved!")
    print(f"   File size: {file_size:.2f} MB")

# =========================
# SAVE METADATA
# =========================
def save_metadata(df, output_file):
    """
    Save reviewer metadata (names, etc.) separately
    """
    print(f"\n💾 Saving metadata to: {output_file}")
    
    # Create metadata DataFrame with index as reviewer_id
    metadata = df[['Author']].copy()
    metadata.insert(0, 'reviewer_id', range(len(metadata)))
    
    # Save to CSV
    metadata.to_csv(output_file, index=False)
    print(f"✅ Metadata saved!")

# =========================
# LOAD EMBEDDINGS (for testing)
# =========================
def load_embeddings(embeddings_file):
    """
    Load embeddings from .npz file
    """
    print(f"\n📂 Loading embeddings from: {embeddings_file}")
    
    data = np.load(embeddings_file, allow_pickle=True)
    
    embeddings = {
        'title': data['title_embeddings'],
        'abstract': data['abstract_embeddings'],
        'keywords': data['keywords_embeddings'],
        'summary': data['summary_embeddings'],
        'reviewer_ids': data['reviewer_ids'],
        'model_name': str(data['model_name']),
        'embedding_dim': int(data['embedding_dim'])
    }
    
    print(f"✅ Embeddings loaded!")
    print(f"   Number of reviewers: {len(embeddings['reviewer_ids'])}")
    print(f"   Embedding dimension: {embeddings['embedding_dim']}")
    print(f"   Model used: {embeddings['model_name']}")
    
    return embeddings

# =========================
# MAIN PIPELINE
# =========================
def main():
    """
    Main pipeline to generate and store embeddings
    """
    print("="*80)
    print("🚀 REVIEWER EMBEDDINGS GENERATION PIPELINE")
    print("="*80)
    
    try:
        # Step 1: Load merged CSV
        print("\nSTEP 1: Loading reviewer data...")
        if not os.path.exists(INPUT_CSV):
            print(f"❌ ERROR: File not found: {INPUT_CSV}")
            return
        
        df = pd.read_csv(INPUT_CSV)
        print(f"✅ Loaded {len(df)} reviewers")
        print(f"   Columns: {list(df.columns)}")
        
        # Step 2: Load embedding model
        print("\nSTEP 2: Loading embedding model...")
        model = load_model()
        
        # Step 3: Generate embeddings
        print("\nSTEP 3: Generating embeddings...")
        embeddings = generate_embeddings(df, model)
        
        # Step 4: Save embeddings
        print("\nSTEP 4: Saving embeddings...")
        save_embeddings(embeddings, EMBEDDINGS_FILE)
        
        # Step 5: Save metadata
        print("\nSTEP 5: Saving metadata...")
        save_metadata(df, METADATA_FILE)
        
        # Step 6: Test loading (verification)
        print("\nSTEP 6: Verifying saved embeddings...")
        loaded = load_embeddings(EMBEDDINGS_FILE)
        
        # Final summary
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print("="*80)
        print(f"   Embeddings file: {EMBEDDINGS_FILE}")
        print(f"   Metadata file: {METADATA_FILE}")
        print(f"   Total reviewers: {len(df)}")
        print(f"   Embedding dimension: {loaded['embedding_dim']}")
        print(f"   Model: {loaded['model_name']}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()