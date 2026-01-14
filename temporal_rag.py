# temporal_rag.py
# ------------------------------------------------------------
# 1) TemporalRAGIndexBuilder — готовит BM25 + E5 embeddings + FAISS, умеет save/load.
# 2) TemporalRAGPipeline — retrieval (+time RRF) -> dedup/cluster -> judge -> summarize
#    + отдельный метод retrieve_only() (без LLM, или без summarize).
# ------------------------------------------------------------

from __future__ import annotations

import re
import json
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from vllm import SamplingParams


# =========================
# Utils: tokenize/snippet/slug
# =========================

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def tokenize_ru(text: str):
    text = str(text).lower()
    text = re.sub(r"[^0-9a-zа-яё\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def snippet(t: str, n: int = 1000) -> str:
    return str(t)[:n]


def slugify_encoder_name(name: str) -> str:
    s = str(name).strip().lower().replace("/", "_")
    s = re.sub(r"[^0-9a-z._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# =========================
# Index builder
# =========================

@dataclass
class IndexArtifacts:
    df: pd.DataFrame
    encoder_name: str
    device: str
    encoder_tag: str
    corpus_tok: list
    bm25: BM25Okapi
    encoder: SentenceTransformer
    E_docs: np.ndarray
    index: Any  # faiss.Index
    rowmap: pd.DataFrame


class TemporalRAGIndexBuilder:
    """
    Делает ровно то, что у тебя в пайплайне:
      - corpus_tok + BM25Okapi
      - E_docs по SentenceTransformer (E5: passage:)
      - FAISS IndexFlatIP
      - save(rowmap, E_docs, faiss index, bm25 tokens)

    Имена файлов строятся из encoder_name (encoder_tag).
    """

    def __init__(self, save_dir: str | Path, encoder_name: str, device: str = "cuda"):
        if faiss is None:
            raise ImportError("faiss is not available in this environment")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.encoder_name = encoder_name
        self.device = device
        self.encoder_tag = slugify_encoder_name(encoder_name)

    # ---- file paths (derived from encoder_name) ----
    def path_rowmap(self) -> Path:
        return self.save_dir / "rowmap.parquet"

    def path_embeddings(self) -> Path:
        return self.save_dir / f"E_docs__{self.encoder_tag}.npy"

    def path_faiss(self) -> Path:
        return self.save_dir / f"faiss__{self.encoder_tag}.index"

    def path_bm25_tokens(self) -> Path:
        return self.save_dir / f"bm25_corpus_tok__{self.encoder_tag}.pkl"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "message" not in df.columns:
            raise KeyError("df must contain column 'message'")
        df["message"] = df["message"].fillna("").astype(str)

        # ensure date_day exists (UTC normalized)
        if "date_day" not in df.columns:
            if "date" not in df.columns:
                raise KeyError("df must contain 'date_day' or 'date'")
            df["date_day"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.normalize()
        else:
            df["date_day"] = pd.to_datetime(df["date_day"], errors="coerce", utc=True).dt.normalize()

        # optional columns normalize
        for col in ["message_id", "id_channel", "channel_name"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    def build(self, df: pd.DataFrame, batch_size: int = 64) -> IndexArtifacts:
        df = self._prepare_df(df)

        # BM25
        corpus_tok = [tokenize_ru(t) for t in df["message"].tolist()]
        bm25 = BM25Okapi(corpus_tok)

        # Encoder + embeddings (E5 style: passage:)
        encoder = SentenceTransformer(self.encoder_name, device=self.device)
        doc_inputs = ["passage: " + t for t in df["message"].tolist()]
        E_docs = encoder.encode(
            doc_inputs,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # FAISS
        
        try:
            import torch
            if isinstance(E_docs, torch.Tensor):
                # если это CUDA tensor — обязательно на CPU
                E_docs = E_docs.detach().to("cpu").numpy()
        except Exception:
            pass
        
        # если это список батчей и т.п.
        E_docs = np.asarray(E_docs, dtype=np.float32)
        E_docs = np.ascontiguousarray(E_docs, dtype=np.float32)
        
        # отладка на 1 запуск (можешь потом убрать)
        print("E_docs:", type(E_docs), E_docs.dtype, E_docs.shape, "is ndarray:", isinstance(E_docs, np.ndarray))
        
        dim = int(E_docs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(E_docs)


        # Rowmap (как у тебя)
        want_cols = ["message_id", "date", "date_day", "id_channel", "channel_name"]
        have_cols = [c for c in want_cols if c in df.columns]
        rowmap = df[have_cols].copy() if have_cols else pd.DataFrame()

        return IndexArtifacts(
            df=df,
            encoder_name=self.encoder_name,
            device=self.device,
            encoder_tag=self.encoder_tag,
            corpus_tok=corpus_tok,
            bm25=bm25,
            encoder=encoder,
            E_docs=E_docs,
            index=index,
            rowmap=rowmap,
        )

    def save(self, art: IndexArtifacts) -> None:
        if not art.rowmap.empty:
            art.rowmap.to_parquet(self.path_rowmap(), index=False)

        np.save(self.path_embeddings(), art.E_docs)
        faiss.write_index(art.index, str(self.path_faiss()))

        with open(self.path_bm25_tokens(), "wb") as f:
            pickle.dump(art.corpus_tok, f)

    def load(
        self,
        df: pd.DataFrame,
        *,
        build_encoder: bool = True,
    ) -> tuple[pd.DataFrame, Optional[SentenceTransformer], Any, BM25Okapi]:
        """
        load() = df (prepared), encoder (optional), faiss index, bm25
        """
        if faiss is None:
            raise ImportError("faiss is not available in this environment")

        df = self._prepare_df(df)

        if not self.path_faiss().exists():
            raise FileNotFoundError(f"FAISS index not found: {self.path_faiss()}")
        if not self.path_bm25_tokens().exists():
            raise FileNotFoundError(f"BM25 tokens not found: {self.path_bm25_tokens()}")

        index = faiss.read_index(str(self.path_faiss()))

        with open(self.path_bm25_tokens(), "rb") as f:
            corpus_tok = pickle.load(f)
        bm25 = BM25Okapi(corpus_tok)

        encoder = None
        if build_encoder:
            encoder = SentenceTransformer(self.encoder_name, device=self.device)

        return df, encoder, index, bm25


# =========================
# Dedup/cluster (твоя функция, чуть упакована)
# =========================

def _normalize_for_dedup(text: str, mask_numbers: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _HANDLE_RE.sub(" ", t)
    t = re.sub(r"[^\w\s%.,\-]+", " ", t, flags=re.UNICODE)
    if mask_numbers:
        t = re.sub(r"\d+(?:[.,]\d+)?", "<num>", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def _union_find(n: int):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return find, union


def dedup_cluster_candidates_time(
    cand: pd.DataFrame,
    encoder,
    text_col: str = "message",
    date_col: str = "date_day",
    channel_col: str = "channel_name",
    score_col: str = "score_rrf",
    sim_threshold: float = 0.95,
    knn: int = 20,
    keep_per_cluster: int = 1,
    mask_numbers: bool = True,
    max_day_diff: int = 1,
    overwrite_channel: bool = True,
    channel_join: str = "; ",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if cand is None or len(cand) == 0:
        return cand, pd.DataFrame(), pd.DataFrame()

    cand = cand.copy().reset_index(drop=True)

    dts = pd.to_datetime(cand[date_col], errors="coerce", utc=True).dt.normalize()
    cand["_dt"] = dts
    cand["_dt_str"] = cand["_dt"].dt.strftime("%Y-%m-%d").fillna("")

    norm = cand[text_col].fillna("").map(lambda s: _normalize_for_dedup(s, mask_numbers=mask_numbers))
    cand["_h"] = norm.map(_stable_hash)
    cand["_hk"] = cand["_h"].astype(str) + "|" + cand["_dt_str"].astype(str)

    if score_col in cand.columns:
        rep_idx = (
            cand.sort_values(score_col, ascending=False)
                .groupby("_hk", as_index=False)
                .head(1)
                .index.to_numpy()
        )
    else:
        rep_idx = cand.groupby("_hk", as_index=False).head(1).index.to_numpy()

    rep = cand.loc[rep_idx].copy().reset_index(drop=True)

    texts = rep[text_col].fillna("").tolist()
    X = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    rep_dt = pd.to_datetime(rep["_dt"], errors="coerce", utc=True)
    rep_dt = rep_dt.dt.tz_convert(None).dt.normalize().to_numpy(dtype="datetime64[D]")

    m = len(rep)
    find, union = _union_find(m)

    if m > 1:
        if faiss is None:
            S = X @ X.T
            for i in range(m):
                js = np.where(S[i, i + 1:] >= sim_threshold)[0] + (i + 1)
                for j in js:
                    if np.isnat(rep_dt[i]) or np.isnat(rep_dt[j]):
                        continue
                    day_diff = abs(int((rep_dt[i] - rep_dt[j]).astype("timedelta64[D]").astype(int)))
                    if day_diff <= max_day_diff:
                        union(i, int(j))
        else:
            idx = faiss.IndexFlatIP(X.shape[1])
            idx.add(X)
            D, I = idx.search(X, min(knn, m))
            for i in range(m):
                for score, j in zip(D[i], I[i]):
                    if j < 0 or j == i:
                        continue
                    if float(score) < sim_threshold:
                        continue
                    if np.isnat(rep_dt[i]) or np.isnat(rep_dt[j]):
                        continue
                    day_diff = abs(int((rep_dt[i] - rep_dt[j]).astype("timedelta64[D]").astype(int)))
                    if day_diff <= max_day_diff:
                        union(i, int(j))

    rep_cluster = np.array([find(i) for i in range(m)], dtype=np.int32)
    _, rep_cluster = np.unique(rep_cluster, return_inverse=True)
    rep["_rep_cluster"] = rep_cluster

    hk_to_cluster = dict(zip(rep["_hk"].tolist(), rep["_rep_cluster"].tolist()))
    cand["_cluster_id"] = cand["_hk"].map(hk_to_cluster).fillna(-1).astype(np.int32)

    cluster_sizes = cand.groupby("_cluster_id").size()

    if channel_col in cand.columns:
        ch_joined = (
            cand.groupby("_cluster_id")[channel_col]
                .apply(lambda s: channel_join.join(sorted({str(x) for x in s.dropna().tolist()})))
        )
    else:
        ch_joined = pd.Series(dtype=str)

    if score_col in cand.columns:
        cand_dedup = (
            cand.sort_values(score_col, ascending=False)
                .groupby("_cluster_id", group_keys=False)
                .head(keep_per_cluster)
                .reset_index(drop=True)
        )
    else:
        cand_dedup = (
            cand.groupby("_cluster_id", group_keys=False)
                .head(keep_per_cluster)
                .reset_index(drop=True)
        )

    cand_dedup["cluster_size"] = cand_dedup["_cluster_id"].map(cluster_sizes).astype(int)

    if channel_col in cand.columns:
        cand_dedup["channel_all"] = cand_dedup["_cluster_id"].map(ch_joined).fillna("")
        cand_dedup["channel_primary"] = cand_dedup[channel_col].astype(str)
        if overwrite_channel and channel_col in cand_dedup.columns:
            cand_dedup[channel_col] = cand_dedup["channel_all"]

    cand_dedup = cand_dedup.drop(columns=["_h", "_hk", "_dt", "_dt_str"], errors="ignore")

    clusters = cand[["_cluster_id"]].copy()
    clusters["cluster_size"] = clusters["_cluster_id"].map(cluster_sizes).astype(int)

    members = cand[["_cluster_id"]].copy()
    for c in ["date_day", "date", "channel_name", "channel", "message_id", "score_rrf"]:
        if c in cand.columns:
            members[c] = cand[c]
    members["text_snip"] = cand[text_col].fillna("").map(lambda s: s[:250])

    return cand_dedup, clusters, members


# =========================
# Retrieval: dense + bm25 + RRF + time bonus
# =========================

def _topk_indices_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    if k == len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int)


def dense_candidates_faiss(index, encoder, query: str, topN: int = 500):
    qv = encoder.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    scores, idx = index.search(qv, int(topN))
    return idx[0].astype(int), scores[0].astype(np.float32)


def _compute_time_arrays(df: pd.DataFrame, rowpos: np.ndarray, anchor_date, date_col: str):
    ad = pd.to_datetime(anchor_date, utc=True).normalize()
    dts = pd.to_datetime(df.loc[rowpos, date_col], errors="coerce", utc=True).dt.normalize()
    age = (ad - dts).dt.days.to_numpy(dtype=np.float32)
    age = np.where(np.isfinite(age), age, 1e9).astype(np.float32)
    age = np.where(age < 0, 1e9, age).astype(np.float32)
    return dts, age


def _time_rank_from_age(age_days: np.ndarray) -> np.ndarray:
    order = np.argsort(age_days, kind="stable")
    rank = np.empty_like(order, dtype=np.int32)
    rank[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return rank


def hybrid_retrieve_rrf(
    df: pd.DataFrame,
    index,
    encoder,
    bm25,
    tokenize_fn: Callable[[str], list],
    query: str,
    *,
    k: int = 50,
    topN_each: int = 500,
    k_rrf: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    anchor_date: str | pd.Timestamp | None = None,
    date_col: str = "date_day",
    max_window_days: int | None = 365,
    w_time: float = 0.5,
    w_channel: float | None = None,
    channel_w_col: str = "channel_w",
) -> pd.DataFrame:
    # allowed mask (<= anchor_date, optionally within window)
    if anchor_date is not None:
        ad = pd.to_datetime(anchor_date, utc=True).normalize()
        if date_col not in df.columns:
            raise KeyError(f"date_col='{date_col}' not found in df.columns")
        dts_all = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.normalize()
        allowed = (dts_all <= ad)
        if max_window_days is not None:
            age_all = (ad - dts_all).dt.days
            allowed &= (age_all >= 0) & (age_all <= int(max_window_days))
        allowed_np = allowed.to_numpy(dtype=bool)
    else:
        allowed_np = None

    # dense
    d_idx, _ = dense_candidates_faiss(index, encoder, query, topN=int(topN_each))
    if allowed_np is not None and len(d_idx) > 0:
        d_idx = d_idx[allowed_np[d_idx]]
    dense_rank = {int(rowpos): r for r, rowpos in enumerate(d_idx, start=1)}

    # dense-only path
    if bm25 is None:
        union = d_idx.astype(int)
        if len(union) == 0:
            return df.iloc[[]].copy().reset_index(drop=True)

        rrf = w_dense / (k_rrf + np.arange(1, len(union) + 1, dtype=np.float32))

        rank_time = None
        if anchor_date is not None and w_time and len(union) > 0:
            _, age = _compute_time_arrays(df, union, anchor_date, date_col)
            rank_time = _time_rank_from_age(age)
            rrf = rrf + (w_time / (k_rrf + rank_time.astype(np.float32)))

        order = np.argsort(-rrf)
        union, rrf = union[order], rrf[order]
        if rank_time is not None:
            rank_time = rank_time[order]

        out = df.iloc[union].copy()
        out["_rowpos"] = union
        out["score_rrf"] = rrf
        out["rank_dense"] = out["_rowpos"].map(lambda rp: dense_rank.get(int(rp), np.nan))
        out["rank_bm25"] = np.nan

        if anchor_date is not None:
            doc_day, age = _compute_time_arrays(df, union, anchor_date, date_col)
            out["doc_day"] = doc_day.dt.tz_localize(None)
            out["age_days"] = age
            if rank_time is not None:
                out["rank_time"] = rank_time

        if channel_w_col in out.columns:
            if w_channel is None:
                w_channel = 0.10 * float(np.std(out["score_rrf"].to_numpy(dtype=np.float32)) or 1.0)
            out["score_rrf"] = out["score_rrf"] + float(w_channel) * out[channel_w_col].astype(np.float32)
            out = out.sort_values("score_rrf", ascending=False)

        return out.head(int(k)).reset_index(drop=True)

    # bm25
    bm_scores = bm25.get_scores(tokenize_fn(query)).astype(np.float32)
    if allowed_np is not None:
        bm_scores[~allowed_np] = -np.inf
    b_idx = _topk_indices_from_scores(bm_scores, int(topN_each))
    bm_rank = {int(rowpos): r for r, rowpos in enumerate(b_idx, start=1)}

    # rrf union
    union = np.array(sorted(set(dense_rank) | set(bm_rank)), dtype=int)
    if len(union) == 0:
        return df.iloc[[]].copy().reset_index(drop=True)

    rrf = np.zeros(len(union), dtype=np.float32)
    for j, rowpos in enumerate(union):
        if rowpos in dense_rank:
            rrf[j] += w_dense / (k_rrf + dense_rank[rowpos])
        if rowpos in bm_rank:
            rrf[j] += w_bm25 / (k_rrf + bm_rank[rowpos])

    rank_time = None
    if anchor_date is not None and w_time and len(union) > 0:
        _, age = _compute_time_arrays(df, union, anchor_date, date_col)
        rank_time = _time_rank_from_age(age)
        rrf = rrf + (w_time / (k_rrf + rank_time.astype(np.float32)))

    order = np.argsort(-rrf)
    union, rrf = union[order], rrf[order]
    if rank_time is not None:
        rank_time = rank_time[order]

    out = df.iloc[union].copy()
    out["_rowpos"] = union
    out["score_rrf"] = rrf
    out["rank_dense"] = out["_rowpos"].map(lambda rp: dense_rank.get(int(rp), np.nan))
    out["rank_bm25"] = out["_rowpos"].map(lambda rp: bm_rank.get(int(rp), np.nan))

    if anchor_date is not None:
        doc_day, age = _compute_time_arrays(df, union, anchor_date, date_col)
        out["doc_day"] = doc_day.dt.tz_localize(None)
        out["age_days"] = age
        if rank_time is not None:
            out["rank_time"] = rank_time

    if channel_w_col in out.columns:
        if w_channel is None:
            w_channel = 0.10 * float(np.std(out["score_rrf"].to_numpy(dtype=np.float32)) or 1.0)
        out["score_rrf"] = out["score_rrf"] + float(w_channel) * out[channel_w_col].astype(np.float32)
        out = out.sort_values("score_rrf", ascending=False)

    return out.head(int(k)).reset_index(drop=True)


# =========================
# Judge + Summarize helpers
# =========================

JUDGE_SYSTEM_DEFAULT = """Ты — строгий эксперт по информационному поиску по новостям (в т.ч. экономическим).

Твоя задача: оценить релевантность кандидатной новости запросу. Запрос может быть:
- коротким топиком (например "курс рубля к доллару"),
- или текстом другой новости (тогда запрос описывает конкретный инфоповод).

Используй ТОЛЬКО текст кандидатного документа. Ничего не додумывай.

Шкала релевантности:
2 — документ явно про то же самое: отвечает топику ИЛИ описывает тот же инфоповод/факт/событие, что и запрос.
1 — документ связан по теме/контексту, но это немного другой инфоповод, или про то же, но без прямого соответствия.
0 — нерелевантно совсем.

Правило строгости:
ставь 2 только если связь очевидна по тексту документа; если информации недостаточно — ставь 0 или 1.

Верни строго валидный JSON и ничего больше:
{"relevance": 0|1|2}
"""


def _parse_relevance(text: str) -> int:
    text = str(text).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        try:
            obj = json.loads(blob)
            val = int(obj.get("relevance", 0))
            return val if val in (0, 1, 2) else 0
        except Exception:
            pass
    m2 = re.search(r"relevance\"\s*:\s*([012])", text)
    if m2:
        return int(m2.group(1))
    return 0


def judge_filter_candidates(
    cand: pd.DataFrame,
    query: str,
    judge_llm,
    judge_tokenizer,
    *,
    judge_system: str = JUDGE_SYSTEM_DEFAULT,
    keep_threshold: int = 1,
    text_col: str = "message",
    channel_col: str = "channel_name",
    date_col: str = "date_day",
    doc_max_chars: int = 1200,
    batch_size: int = 32,
    max_out_tokens: int = 40,
) -> pd.DataFrame:
    if cand is None or len(cand) == 0:
        return cand

    prompts = []
    for _, row in cand.iterrows():
        doc = str(row.get(text_col, ""))[:doc_max_chars]
        ch = str(row.get(channel_col, ""))
        dt = str(row.get(date_col, ""))

        user_msg = (
            f"ЗАПРОС:\n{query}\n\n"
            f"КАНДИДАТ:\n"
            f"channel={ch}\n"
            f"date={dt}\n"
            f"text:\n{doc}\n"
        )
        messages = [
            {"role": "system", "content": judge_system},
            {"role": "user", "content": user_msg},
        ]
        prompt = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_out_tokens))

    relevances: list[int] = []
    for i in range(0, len(prompts), int(batch_size)):
        batch_prompts = prompts[i:i + int(batch_size)]
        outs = judge_llm.generate(batch_prompts, sampling)
        for o in outs:
            relevances.append(_parse_relevance(o.outputs[0].text))

    out_df = cand.copy()
    out_df["judge_relevance"] = relevances
    return out_df[out_df["judge_relevance"] >= int(keep_threshold)].reset_index(drop=True)


def build_rag_context(
    query: str,
    cand: pd.DataFrame,
    anchor_date: str,
    *,
    k_docs: int = 30,
    snip_chars: int = 850,
    hot_window_days: int = 30,
    hot_ratio: float = 0.8,
) -> str:
    if cand is None or len(cand) == 0:
        return (
            f"АКТУАЛЬНАЯ ДАТА ОБЗОРА: {anchor_date}\n"
            f"ВОПРОС/ЗАПРОС:\n{query}\n\n"
            f"ИСТОЧНИКИ:\n(нет документов)\n"
        )

    c = cand.copy()
    date_col = "date_day" if "date_day" in c.columns else "date"
    score_col = "score_temporal" if "score_temporal" in c.columns else "score_rrf"

    if "age_days" not in c.columns:
        ad = pd.to_datetime(anchor_date, utc=True).normalize()
        dts = pd.to_datetime(c[date_col], errors="coerce", utc=True).dt.normalize()
        c["age_days"] = (ad - dts).dt.days.astype("float32")

    age = c["age_days"].to_numpy(dtype=np.float32)
    hot_mask = (age >= 0) & (age <= float(hot_window_days))

    c = c.sort_values(score_col, ascending=False)

    n_hot = int(round(int(k_docs) * float(hot_ratio)))
    n_hot = max(0, min(n_hot, int(k_docs)))

    hot_part = c[hot_mask].head(n_hot)
    rest_part = c[~hot_mask].head(int(k_docs) - len(hot_part))
    picked = pd.concat([hot_part, rest_part], axis=0)

    dd = pd.to_datetime(picked[date_col], errors="coerce", utc=True).dt.normalize()
    picked = picked.assign(_doc_day=dd).sort_values(["_doc_day", score_col], ascending=[False, False]).head(int(k_docs))

    blocks = []
    for i, row in enumerate(picked.itertuples(index=False), start=1):
        date_day = getattr(row, "date_day", getattr(row, "date", ""))
        if isinstance(date_day, pd.Timestamp):
            date_day = date_day.strftime("%Y-%m-%d")
        date_day = str(date_day)[:10]
        channel = getattr(row, "channel_name", "")
        text = getattr(row, "message", "")
        blocks.append(f"[{i}] date={date_day} channel(s)={channel}\n document=" + snippet(text, snip_chars))

    return (
        f"АКТУАЛЬНАЯ ДАТА ОБЗОРА: {anchor_date}\n"
        f"ВОПРОС/ЗАПРОС:\n{query}\n\n"
        f"ИСТОЧНИКИ:\n" + "\n\n".join(blocks)
    )


def rag_summarize(
    sum_llm,
    sum_tokenizer,
    system_prompt: str,
    query: str,
    cand: pd.DataFrame,
    anchor_date: str,
    *,
    k_docs: int = 25,
    snip_chars: int = 900,
    max_new_tokens: int = 2000,
    hot_window_days: int = 30,
    hot_ratio: float = 0.8,
) -> tuple[str, str]:
    user = build_rag_context(
        query=query,
        cand=cand,
        anchor_date=str(anchor_date),
        k_docs=int(k_docs),
        snip_chars=int(snip_chars),
        hot_window_days=int(hot_window_days),
        hot_ratio=float(hot_ratio),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    prompt = sum_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens))
    result = sum_llm.generate([prompt], sampling)[0]
    text = result.outputs[0].text.strip()
    return text, user


# =========================
# Main pipeline class
# =========================

@dataclass
class RetrievalResult:
    candidates: pd.DataFrame
    candidates_dedup: Optional[pd.DataFrame] = None
    candidates_filtered: Optional[pd.DataFrame] = None
    clusters: Optional[pd.DataFrame] = None
    members: Optional[pd.DataFrame] = None


@dataclass
class RAGResult(RetrievalResult):
    context: Optional[str] = None
    summary: Optional[str] = None


class TemporalRAGPipeline:
    """
    Реализует твой end-to-end пайплайн:

      retrieval = hybrid_retrieve_rrf (dense+bm25+RRF + time bonus)
      -> dedup_cluster_candidates_time
      -> judge_filter_candidates
      -> rag_summarize (LLM)

    Плюс:
      - retrieve_only(...)  : только retrieval (+ optional dedup/judge)
      - answer(...)         : полный пайплайн с summary
    """

    def __init__(
        self,
        df: pd.DataFrame,
        index,
        encoder,
        bm25: Optional[BM25Okapi],
        *,
        tokenize_fn: Callable[[str], list] = tokenize_ru,
        # LLMs (optional)
        sum_llm=None,
        sum_tokenizer=None,
        judge_llm=None,
        judge_tokenizer=None,
        # prompts
        system_prompt: str = "",
        judge_system: str = JUDGE_SYSTEM_DEFAULT,
    ):
        self.df = df
        self.index = index
        self.encoder = encoder
        self.bm25 = bm25
        self.tokenize_fn = tokenize_fn

        self.sum_llm = sum_llm
        self.sum_tokenizer = sum_tokenizer
        self.judge_llm = judge_llm
        self.judge_tokenizer = judge_tokenizer

        self.system_prompt = system_prompt
        self.judge_system = judge_system

    # -------- retrieval only --------
    def retrieve_only(
        self,
        query: str,
        *,
        anchor_date: str,
        # retrieval params
        k_retrieve: int = 50,
        topN_each: int = 500,
        k_rrf: int = 60,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
        max_window_days: int | None = 365,
        w_time: float = 0.5,
        w_channel: float | None = None,
        # post-processing toggles
        do_dedup: bool = True,
        do_judge: bool = True,
        # dedup params
        dedup_sim_threshold: float = 0.95,
        dedup_knn: int = 30,
        dedup_keep_per_cluster: int = 1,
        dedup_mask_numbers: bool = False,
        dedup_max_day_diff: int = 1,
        dedup_overwrite_channel: bool = True,
        # judge params
        judge_keep_threshold: int = 1,
        judge_batch_size: int = 32,
        judge_doc_max_chars: int = 1200,
        judge_max_out_tokens: int = 40,
    ) -> RetrievalResult:
        # 1) retrieve
        cand = hybrid_retrieve_rrf(
            df=self.df,
            index=self.index,
            encoder=self.encoder,
            bm25=self.bm25,
            tokenize_fn=self.tokenize_fn,
            query=query,
            k=k_retrieve,
            topN_each=topN_each,
            k_rrf=k_rrf,
            w_dense=w_dense,
            w_bm25=w_bm25,
            anchor_date=anchor_date,
            max_window_days=max_window_days,
            w_time=w_time,
            w_channel=w_channel,
        )

        cand_before = cand

        # 2) dedup cluster
        cand_clusters = None
        members = None
        cand_after = None
        if do_dedup and cand is not None and len(cand) > 0 and self.encoder is not None:
            cand_after, cand_clusters, members = dedup_cluster_candidates_time(
                cand=cand,
                encoder=self.encoder,
                text_col="message",
                score_col="score_rrf",
                sim_threshold=float(dedup_sim_threshold),
                knn=int(dedup_knn),
                keep_per_cluster=int(dedup_keep_per_cluster),
                mask_numbers=bool(dedup_mask_numbers),
                max_day_diff=int(dedup_max_day_diff),
                overwrite_channel=bool(dedup_overwrite_channel),
            )
            cand = cand_after

        # 3) judge filter
        cand_filtered = None
        if do_judge and self.judge_llm is not None and self.judge_tokenizer is not None and cand is not None and len(cand) > 0:
            cand_filtered = judge_filter_candidates(
                cand=cand,
                query=query,
                judge_llm=self.judge_llm,
                judge_tokenizer=self.judge_tokenizer,
                judge_system=self.judge_system,
                keep_threshold=int(judge_keep_threshold),
                doc_max_chars=int(judge_doc_max_chars),
                batch_size=int(judge_batch_size),
                max_out_tokens=int(judge_max_out_tokens),
            )
        else:
            cand_filtered = cand

        return RetrievalResult(
            candidates=cand_before,
            candidates_dedup=cand_after,
            candidates_filtered=cand_filtered,
            clusters=cand_clusters,
            members=members,
        )

    # -------- full answer --------
    def answer(
        self,
        query: str,
        *,
        anchor_date: str,
        # retrieval params
        k_retrieve: int = 50,
        topN_each: int = 500,
        k_rrf: int = 60,
        w_dense: float = 1.0,
        w_bm25: float = 1.0,
        max_window_days: int | None = 365,
        w_time: float = 0.5,
        w_channel: float | None = None,
        # post-processing toggles
        do_dedup: bool = True,
        do_judge: bool = True,
        # dedup params
        dedup_sim_threshold: float = 0.95,
        dedup_knn: int = 30,
        dedup_keep_per_cluster: int = 1,
        dedup_mask_numbers: bool = False,
        dedup_max_day_diff: int = 1,
        dedup_overwrite_channel: bool = True,
        # judge params
        judge_keep_threshold: int = 1,
        judge_batch_size: int = 32,
        judge_doc_max_chars: int = 1200,
        judge_max_out_tokens: int = 40,
        # summarization params
        k_docs: int = 50,
        snip_chars: int = 1000,
        max_new_tokens: int = 5000,
        hot_window_days: int = 30,
        hot_ratio: float = 0.7,
    ) -> RAGResult:
        rr = self.retrieve_only(
            query=query,
            anchor_date=anchor_date,
            k_retrieve=k_retrieve,
            topN_each=topN_each,
            k_rrf=k_rrf,
            w_dense=w_dense,
            w_bm25=w_bm25,
            max_window_days=max_window_days,
            w_time=w_time,
            w_channel=w_channel,
            do_dedup=do_dedup,
            do_judge=do_judge,
            dedup_sim_threshold=dedup_sim_threshold,
            dedup_knn=dedup_knn,
            dedup_keep_per_cluster=dedup_keep_per_cluster,
            dedup_mask_numbers=dedup_mask_numbers,
            dedup_max_day_diff=dedup_max_day_diff,
            dedup_overwrite_channel=dedup_overwrite_channel,
            judge_keep_threshold=judge_keep_threshold,
            judge_batch_size=judge_batch_size,
            judge_doc_max_chars=judge_doc_max_chars,
            judge_max_out_tokens=judge_max_out_tokens,
        )

        cand_final = rr.candidates_filtered
        ctx = build_rag_context(
            query=query,
            cand=cand_final,
            anchor_date=anchor_date,
            k_docs=min(int(k_docs), len(cand_final)) if cand_final is not None else 0,
            snip_chars=int(snip_chars),
            hot_window_days=int(hot_window_days),
            hot_ratio=float(hot_ratio),
        )

        # If no summarizer model provided, return context + candidates.
        if self.sum_llm is None or self.sum_tokenizer is None:
            return RAGResult(
                candidates=rr.candidates,
                candidates_dedup=rr.candidates_dedup,
                candidates_filtered=rr.candidates_filtered,
                clusters=rr.clusters,
                members=rr.members,
                context=ctx,
                summary=None,
            )

        summary, _ctx_used = rag_summarize(
            sum_llm=self.sum_llm,
            sum_tokenizer=self.sum_tokenizer,
            system_prompt=self.system_prompt,
            query=query,
            cand=cand_final,
            anchor_date=anchor_date,
            k_docs=min(int(k_docs), len(cand_final)) if cand_final is not None else 0,
            snip_chars=int(snip_chars),
            max_new_tokens=int(max_new_tokens),
            hot_window_days=int(hot_window_days),
            hot_ratio=float(hot_ratio),
        )

        return RAGResult(
            candidates=rr.candidates,
            candidates_dedup=rr.candidates_dedup,
            candidates_filtered=rr.candidates_filtered,
            clusters=rr.clusters,
            members=rr.members,
            context=_ctx_used,
            summary=summary,
        )


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset parquet/csv")
    parser.add_argument("--save_dir", type=str, default="indexes")
    parser.add_argument("--encoder_name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--anchor_date", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["build", "retrieve", "answer"], default="answer")

    # rag params
    parser.add_argument("--k_retrieve", type=int, default=150)
    parser.add_argument("--topN_each", type=int, default=2000)
    parser.add_argument("--k_docs", type=int, default=50)
    parser.add_argument("--snip_chars", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=5000)

    args = parser.parse_args()

    # load df
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)

    builder = TemporalRAGIndexBuilder(
        save_dir=args.save_dir,
        encoder_name=args.encoder_name,
        device=args.device,
    )

    if args.mode == "build":
        art = builder.build(df)
        builder.save(art)
        print("OK: built & saved indexes")
        raise SystemExit(0)

    # build in-memory (или можно load, если ты добавишь отдельный режим load)
    art = builder.build(df)

    # llm
    tok = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    llm = LLM(model=args.llm_model, dtype="bfloat16", max_model_len=19200, gpu_memory_utilization=0.88)

    pipe = TemporalRAGPipeline(
        df=art.df,
        index=art.index,
        encoder=art.encoder,
        bm25=art.bm25,
        tokenize_fn=tokenize_ru,
        sum_llm=llm,
        sum_tokenizer=tok,
        judge_llm=llm,
        judge_tokenizer=tok,
        system_prompt=SYSTEM_PROMPT,  # должен быть определён в файле
    )

    if args.mode == "retrieve":
        rr = pipe.retrieve_only(
            query=args.query,
            anchor_date=args.anchor_date,
            k_retrieve=args.k_retrieve,
            topN_each=args.topN_each,
        )
        print(rr.candidates_filtered.head(20).to_string(index=False))
        raise SystemExit(0)

    out = pipe.answer(
        query=args.query,
        anchor_date=args.anchor_date,
        k_retrieve=args.k_retrieve,
        topN_each=args.topN_each,
        k_docs=args.k_docs,
        snip_chars=args.snip_chars,
        max_new_tokens=args.max_new_tokens,
    )

    print(out.summary or out.context or "")
