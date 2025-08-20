#!/usr/bin/env python3
"""
prep_data.py — one-time preprocessing for the anime recommender.

INPUT  : a CSV from Kaggle (columns you listed are supported)
OUTPUT : artifacts + a clean parquet table for the Streamlit app

What we do:
1) Parse/clean columns, derive a numeric year from `aired_from`
2) Build numeric features (scaled): score_norm, year_norm, members_log_norm, popularity_inv_norm
3) Build multi-hot tag features from genres/themes/demographics (top-N tags)
4) Fit K-Means (descriptive), compute PCA(2) for the cluster plot
5) Save artifacts for fast loading at runtime

Run:
    python prep_data.py --csv data/anime.csv --clusters 12
"""

import argparse, os, json, re
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------- helpers ----------

def parse_year_from_any(x):
    """Try to parse a 4-digit year from 'aired_from' which may be a date or text."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    # 1) try pandas datetime
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return float(dt.year)
    # 2) fallback: grab first 4-digit year
    m = re.search(r'(19|20)\d{2}', s)
    if m:
        return float(m.group(0))
    return np.nan

def split_tags(cell):
    """Split comma/semicolon separated tags; return list[str]."""
    if pd.isna(cell):
        return []
    parts = re.split(r'[;,]', str(cell))
    return [p.strip() for p in parts if p.strip()]

def top_k_tags(series_of_lists, k=60):
    """Find top-k most frequent tags across a Series of lists."""
    from collections import Counter
    c = Counter()
    for tags in series_of_lists:
        c.update(tags)
    return [t for t,_ in c.most_common(k)]

# ---------- main pipeline ----------

def build_artifacts(
    csv_path: str,
    top_genres: int = 60,
    top_themes: int = 40,
    top_demo: int = 10,
    n_clusters: int = 12,
    random_state: int = 42
):
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1) Core columns
    # keep originals for display
    keep_cols = ["title","image","synopsis","status","episodes","rating",
                 "rank","popularity","members","favorites","score","scored_by",
                 "genres","themes","demographics","aired_from","aired_to"]
    for c in keep_cols:
        if c not in df.columns:
            # keep missing ones as NA if not present
            df[c] = pd.NA

    # 2) Coerce numerics
    num_cols_raw = ["score","members","popularity","episodes","rank","scored_by","favorites"]
    for c in num_cols_raw:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Year from aired_from
    df["year"] = df["aired_from"].apply(parse_year_from_any)

    # 4) Fill numeric missings with medians (simple, stable choice)
    for c in ["score","members","popularity","episodes","rank","scored_by","favorites","year"]:
        med = df[c].median() if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    # 5) Build tag lists
    genres_list = df["genres"].apply(split_tags)
    themes_list = df["themes"].apply(split_tags)
    demo_list   = df["demographics"].apply(split_tags)

    # choose which tags to keep for one-hot
    topG  = top_k_tags(genres_list, k=top_genres)
    topT  = top_k_tags(themes_list, k=top_themes)
    topD  = top_k_tags(demo_list,   k=top_demo)

    # 6) Start clean frame for display
    out = pd.DataFrame({
        "title": df["title"].astype(str),
        "image": df["image"].astype(str),
        "synopsis": df["synopsis"].astype(str),
        "status": df["status"].astype(str),
        "episodes": df["episodes"].astype(float),
        "rating_label": df["rating"].astype(str),  # e.g., PG-13 (content rating)
        "rank": df["rank"].astype(float),
        "popularity": df["popularity"].astype(float),
        "members": df["members"].astype(float),
        "favorites": df["favorites"].astype(float),
        "score": df["score"].astype(float),
        "scored_by": df["scored_by"].astype(float),
        "genres_raw": df["genres"].astype(str),
        "themes_raw": df["themes"].astype(str),
        "demographics_raw": df["demographics"].astype(str),
        "year": df["year"].astype(float),
    })

    # 7) Numeric features (scaled 0..1)
    #    - members_log_norm: log1p(members) to compress large ranges
    #    - popularity_inv_norm: invert rank so "more popular" => bigger number
    out["members_log"] = np.log1p(out["members"])
    # avoid divide-by-zero: popularity is rank (1 is best), so invert like 1/popularity
    out["popularity_inv"] = 1.0 / (out["popularity"] + 1e-6)

    scaler = MinMaxScaler()
    out[["score_norm","year_norm","members_log_norm","popularity_inv_norm"]] = scaler.fit_transform(
        out[["score","year","members_log","popularity_inv"]]
    )

    # 8) One-hot tags (top lists only to keep features compact)
    #    We keep the source of each tag space separate: genre__, theme__, demo__
    for g in topG:
        out[f"genre__{g}"] = genres_list.apply(lambda L: 1.0 if g in L else 0.0)
    for t in topT:
        out[f"theme__{t}"] = themes_list.apply(lambda L: 1.0 if t in L else 0.0)
    for d in topD:
        out[f"demo__{d}"] = demo_list.apply(lambda L: 1.0 if d in L else 0.0)

    # 9) Assemble feature matrix
    num_feats = ["score_norm","year_norm","members_log_norm","popularity_inv_norm"]
    genre_cols = [c for c in out.columns if c.startswith("genre__")]
    theme_cols = [c for c in out.columns if c.startswith("theme__")]
    demo_cols  = [c for c in out.columns if c.startswith("demo__")]

    feat_cols = num_feats + genre_cols + theme_cols + demo_cols
    X = out[feat_cols].astype("float32").values

    # 10) K-Means clustering (descriptive)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    out["cluster"] = labels

    # 11) PCA(2) for 2-D cluster plot
    pca = PCA(n_components=2, random_state=random_state)
    coords2d = pca.fit_transform(X).astype("float32")

    # 12) Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/features.npy", X.astype("float32"))
    np.save("artifacts/pca2d.npy", coords2d)
    joblib.dump(km, "artifacts/kmeans.pkl")

    meta = {
        "num_cols": num_feats,
        "genre_cols": genre_cols,
        "theme_cols": theme_cols,
        "demo_cols":  demo_cols,
        "feat_cols":  feat_cols
    }
    with open("artifacts/columns.json","w") as f:
        json.dump(meta, f, indent=2)

    # fast index maps for favorites lookup later
    out = out.reset_index(drop=True)
    title_to_id = {t:i for i,t in enumerate(out["title"].tolist())}
    id_to_title = {i:t for t,i in title_to_id.items()}
    with open("artifacts/indexes.json","w") as f:
        json.dump({"title_to_id": title_to_id, "id_to_title": id_to_title}, f)

    # 13) Save clean display table
    out.to_parquet("data/anime_clean.parquet", index=False)

    # 14) Mood weights (editable)
    #     These keys should match AVAILABLE GENRES/THEMES in your dataset.
    mood_weights = {
        "sad":       {"Slice of Life": 1.3, "Comedy": 1.25, "Music": 1.2, "Romance": 1.15},
        "excited":   {"Action": 1.3, "Shounen": 1.25, "Sports": 1.2, "Mecha": 1.15},
        "motivated": {"Sports": 1.25, "Seinen": 1.2, "Drama": 1.15, "Adventure": 1.15},
        "relaxed":   {"Slice of Life": 1.3, "Romance": 1.2, "Food": 1.2, "Fantasy": 1.1},
        "curious":   {"Mystery": 1.25, "Psychological": 1.25, "Sci-Fi": 1.15}
    }
    with open("artifacts/mood_weights.json","w") as f:
        json.dump(mood_weights, f, indent=2)

    print("✅ Done.")
    print("Saved:")
    print("  - data/anime_clean.parquet")
    print("  - artifacts/features.npy, kmeans.pkl, pca2d.npy, columns.json, indexes.json, mood_weights.json")
    print(f"Rows kept: {len(out):,}; Feature dims: {X.shape[1]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the Kaggle anime CSV (e.g., data/anime.csv)")
    ap.add_argument("--clusters", type=int, default=12, help="K-Means number of clusters")
    args = ap.parse_args()

    build_artifacts(csv_path=args.csv, n_clusters=args.clusters)

if __name__ == "__main__":
    main()
