import os
import numpy as np
import pandas as pd
from typing import List, Tuple

def load_texts(csv_path: str, text_col: str = "message") -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()
    return df, texts

def save_rowmap(df: pd.DataFrame, out_path: str, cols=("message_id", "date")) -> None:
    keep = [c for c in cols if c in df.columns]
    df[keep].to_csv(out_path, index=False)

def save_embeddings(path: str, emb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, emb.astype(np.float16))

def batch_iter(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i+batch_size]
