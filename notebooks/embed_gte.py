#!/usr/bin/env python3
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from embed_common import load_texts, save_rowmap, save_embeddings, batch_iter

MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

@torch.inference_mode()
def encode(tokenizer, model, texts, batch_size, max_len, normalize=True):
    out = []
    for i, batch in tqdm(list(batch_iter(texts, batch_size)), desc="Encoding GTE"):
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy().astype(np.float16))
    return np.vstack(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="cleaned_news_exp.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=256)
    args = p.parse_args()

    df, texts = load_texts(args.input)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    emb = encode(tok, model, texts, args.batch_size, args.max_len)
    save_embeddings(args.out, emb)

    print("Saved:", args.out, emb.shape, emb.dtype)

if __name__ == "__main__":
    main()
