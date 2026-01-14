import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embed_common import load_texts, save_rowmap, save_embeddings, batch_iter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="cleaned_news_exp.csv")
    p.add_argument("--model", required=True, help="SentenceTransformer model name")
    p.add_argument("--out", required=True, help="Output .npy (float16)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--prefix", default="", help="Prefix for docs (e.g. 'passage: ' for E5)")
    args = p.parse_args()

    df, texts = load_texts(args.input)

    if args.prefix:
        texts = [args.prefix + t for t in texts]

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_len

    embs = []
    for i, batch in tqdm(list(batch_iter(texts, args.batch_size)), desc=f"Encoding {args.model}"):
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(e.astype(np.float16))

    emb = np.vstack(embs)
    save_embeddings(args.out, emb)

    print("Saved:", args.out, emb.shape, emb.dtype)

if __name__ == "__main__":
    main()
