import os
import pandas as pd
import kagglehub
import glob
import shutil
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    """Download and cache the Email Spam Detection dataset from Kaggle."""
    existing = list(RAW_DIR.glob("*.csv"))
    if existing:
        print(f"✅ Dataset already exists: {existing[0].name}")
        return existing[0]

    print("⬇️ Downloading Email Spam Detection dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("zeeshanyounas001/email-spam-detection")
        files = [f for f in glob.glob(f"{path}/*.csv") if os.path.isfile(f)]
        
        if not files:
            raise FileNotFoundError("No CSV files found in Kaggle download.")
        
        raw_path = RAW_DIR / "spam_mail.csv"
        shutil.copy(files[0], raw_path)
        print(f"✅ Downloaded & saved to {raw_path}")
        return raw_path
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("💡 Ensure KAGGLE_USERNAME & KAGGLE_KEY are set in your environment.")
        raise

def clean_and_split():
    """Clean, preprocess, and split the dataset."""
    existing = list(RAW_DIR.glob("*.csv"))
    if not existing:
        raise FileNotFoundError("No CSV dataset found. Run download_dataset() first.")
    
    raw_path = existing[0]
    print(f"📖 Reading {raw_path.name}...")

    # 🔑 Read CSV with correct columns & encoding
    df = pd.read_csv(raw_path, encoding='utf-8')
    
    # 🔑 Robust column mapping (handles typos & variations)
    col_map = {col.lower().strip(): col for col in df.columns}
    
    # Map Category column
    if 'Category' not in df.columns:
        if 'category' in col_map:
            df = df.rename(columns={col_map['category']: 'Category'})
        elif 'label' in col_map or 'class' in col_map:
            rename_col = col_map.get('label') or col_map.get('class')
            df = df.rename(columns={rename_col: 'Category'})
        else:
            print(f"❌ Could not find 'Category' column. Found: {list(df.columns)}")
            raise ValueError("Dataset missing target column.")
    
    # Map Message column (handles "Masseges" typo + other variants)
    if 'Message' not in df.columns:
        message_candidates = ['message', 'messages', 'masseges', 'text', 'content', 'body', 'email']
        found = next((c for c in message_candidates if c in col_map), None)
        if found:
            df = df.rename(columns={col_map[found]: 'Message'})
            print(f"⚠️ Renamed column '{col_map[found]}' → 'Message'")
        else:
            print(f"❌ Could not find 'Message' column. Found: {list(df.columns)}")
            raise ValueError("Dataset missing text column.")

    print(f"🔍 Loaded {len(df)} rows with columns: {list(df.columns)}")
    print(f"🔍 Sample Category values: {df['Category'].dropna().unique().tolist()[:5]}")

    # 🔑 Clean Category labels
    df['Category'] = df['Category'].astype(str).str.strip().str.lower()
    df = df[df['Category'].isin(['ham', 'spam'])].copy()
    df['label'] = (df['Category'] == 'spam').astype(int)  # 1=spam, 0=ham

    # 🔑 Clean Message text
    df['Message'] = df['Message'].fillna("").astype(str)
    # Lowercase + remove non-alphanumeric (keep spaces for NLP)
    df['Message'] = df['Message'].str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)
    # Collapse multiple spaces + strip
    df['Message'] = df['Message'].str.replace(r"\s+", " ", regex=True).str.strip()
    
    # Drop empty messages
    df = df[df['Message'].str.len() > 2].reset_index(drop=True)  # min 3 chars

    if len(df) == 0:
        raise ValueError("All rows were filtered out. Check dataset format.")

    print(f"🔍 After cleaning: {len(df)} rows | Spam: {(df['label']==1).sum()} | Ham: {(df['label']==0).sum()}")

    # 🔑 Stratified train/test split
    df_train, df_test = train_test_split(
        df[['Message', 'label']], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Save to parquet (fast, preserves types)
    df_train.to_parquet(PROC_DIR / "train.parquet", index=False)
    df_test.to_parquet(PROC_DIR / "test.parquet", index=False)
    
    print(f"✅ Data saved. Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"📊 Train distribution: {df_train['label'].value_counts().to_dict()}")
    print(f"📊 Test distribution: {df_test['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    download_dataset()
    clean_and_split()