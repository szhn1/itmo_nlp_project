import argparse
import pandas as pd

USECOLS = [
    "message_id","post_uuid","id_channel","subscribers","types_reactions",
    "date","message","message_vector","dayofweek","hour", "viral_final", "is_economic","topic",
    "confidence","reason","economic_signals","noise_signals"
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../dataset_tg.csv", help="Path to raw CSV")
    p.add_argument("--output", default="cleaned_news_exp.csv", help="Output CSV")
    p.add_argument("--only_economic", action="store_true", help="Keep only is_economic==True")
    p.add_argument("--min_len", type=int, default=0, help="Min message length")
    args = p.parse_args()

    df = pd.read_csv(args.input, usecols=USECOLS)

    if args.only_economic:
        df = df[df["is_economic"] == True].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["message", "date", "topic"]).copy()

    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() >= args.min_len].copy()

    df["topic"] = df["topic"].replace({
        "Сырьевые рын рынки": "Сырьевые рынки",
        "Сырьевые рынры": "Сырьевые рынки",
    })

    df = df.reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("Topics:", df["topic"].nunique())

if __name__ == "__main__":
    main()
