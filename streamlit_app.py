# streamlit_app.py  (table fixes: single index, no NaN, image+description columns)
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ----- Page setup -----
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

# Sidebar width / padding fix so slider labels (e.g., 2027) aren't clipped
st.markdown("""
<style>
/* Widen sidebar a bit and add right padding so slider end label isn't cut */
[data-testid="stSidebar"] { width: 380px !important; }
[data-testid="stSidebar"] > div { padding-right: 14px; }
/* Subtle scrollbar so it doesn't visually block labels */
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 8px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
  background-color: rgba(128,128,128,0.25); border-radius: 4px;
}
/* Give sliders a little breathing room on the right */
[data-testid="stSidebar"] [data-baseweb="slider"] { padding-right: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🎌 Anime Recommender")
st.caption("Python-only app using K-Means (descriptive clusters) + KNN-style cosine (predictive).")

# ----- Load artifacts  -----
@st.cache_resource
def load_artifacts():
    required = [
        "data/anime_clean.parquet",
        "artifacts/features.npy",
        "artifacts/kmeans.pkl",
        "artifacts/pca2d.npy",
        "artifacts/columns.json",
        "artifacts/indexes.json",
        "artifacts/mood_weights.json",
    ]
    if not all(os.path.exists(p) for p in required):
        return None

    df = pd.read_parquet("data/anime_clean.parquet")
    X = np.load("artifacts/features.npy")
    km = joblib.load("artifacts/kmeans.pkl")
    pca2d = np.load("artifacts/pca2d.npy")

    with open("artifacts/columns.json") as f: cols_meta = json.load(f)
    with open("artifacts/indexes.json") as f: idx_meta = json.load(f)
    with open("artifacts/mood_weights.json") as f: mood_weights = json.load(f)

    return {
        "df": df, "X": X, "km": km, "pca2d": pca2d,
        "cols": cols_meta, "idx": idx_meta, "mood": mood_weights
    }

art = load_artifacts()
if art is None:
    st.error(
        "Artifacts not found:\n\n"
        "`python prep_data.py --csv data/anime.csv`"
    )
    st.stop()

df   = art["df"]
X    = art["X"]
cols = art["cols"]
idx  = art["idx"]
mood = art["mood"]

# ---------- Recommendation helpers ----------
def build_user_vector(X, cols_meta, title_to_id, favorites, fav_genres, mood_map, mood_choice):
    """
    Build a user profile vector:
    - mean of favorites
    - small bump for explicit favorite genres
    - multiply mood-weighted genres
    - L2-normalize
    """
    feat_cols = cols_meta["feat_cols"]
    v = np.zeros(X.shape[1], dtype="float32")

    if favorites:
        rows = [title_to_id[t] for t in favorites if t in title_to_id]
        if rows:
            v = X[rows].mean(axis=0)

    for g in fav_genres:
        col = f"genre__{g}"
        if col in feat_cols:
            j = feat_cols.index(col)
            v[j] += 0.2

    if mood_choice and (mood_choice in mood_map):
        for tag, w in mood_map[mood_choice].items():
            col = f"genre__{tag}"
            if col in feat_cols:
                j = feat_cols.index(col)
                v[j] *= float(w)

    n = np.linalg.norm(v) + 1e-8
    return v / n

def filter_candidates(df, rating_min, year_range, required_genres):
    m = (df["score"] >= rating_min) & (df["year"].between(year_range[0], year_range[1]))
    for g in required_genres:
        col = f"genre__{g}"
        if col in df.columns:
            m &= (df[col] > 0.0)
    return m

def rerank(profile_vec, X, df, cols_meta, candidate_idx, top_n, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    candidate_idx = np.array([i for i in candidate_idx if i not in exclude_ids], dtype=int)
    if candidate_idx.size == 0:
        return candidate_idx, pd.DataFrame()

    sims = cosine_similarity(profile_vec.reshape(1, -1), X[candidate_idx])[0]

    j_score = cols_meta["feat_cols"].index("score_norm")
    rating_bonus = X[candidate_idx, j_score]

    final = 0.7 * sims + 0.3 * rating_bonus
    order = np.argsort(-final)
    picked = candidate_idx[order][:top_n]

    # Include image + synopsis so we can present them in the table
    out = df.iloc[picked][[
        "title", "image", "synopsis", "year", "score", "popularity", "members",
        "genres_raw", "themes_raw", "demographics_raw"
    ]].copy()
    out["_similarity"]  = sims[order][:top_n]
    out["_blend_score"] = final[order][:top_n]
    return picked, out.reset_index(drop=True)

# ---------- Sidebar: Interactive Inputs ----------
st.sidebar.header("Your Preferences")

# Favorites (native type-to-search)
all_titles = df["title"].astype(str).tolist()
fav_sel = st.sidebar.multiselect(
    "Favorite anime (type to search)",
    options=all_titles,
    default=[]
)

# Genres
genre_tags = sorted(g.replace("genre__","") for g in cols.get("genre_cols", []))
genre_sel = st.sidebar.multiselect("Favorite genres", options=genre_tags, default=[])

# Rating slider
rating_min_val = float(df["score"].min())
rating_max_val = float(df["score"].max())
rating_min = st.sidebar.slider(
    "Minimum rating",
    min_value=float(round(rating_min_val,1)),
    max_value=float(round(rating_max_val,1)),
    value=float(round(df["score"].median(),1)),
    step=0.1
)

# Year slider (min fixed at 1970) + explicit caption to show full range
year_min_val = max(1970, int(df["year"].min()))
year_max_val = int(df["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min_val,
    max_value=year_max_val,
    value=(year_min_val, year_max_val)
)
st.sidebar.caption(f"Selected years: **{year_range[0]}–{year_range[1]}**")

# Mood
mood_options = ["(none)"] + list(mood.keys())
mood_choice = st.sidebar.selectbox("Mood", options=mood_options, index=0)

# N
top_n = st.sidebar.slider("Number of recommendations", 5, 30, 12)

run_btn = st.sidebar.button("Get Recommendations", type="primary")

# ----- Main layout: results + visuals -----
results_area = st.container()
col1, col2 = st.columns(2)
visual1_area = col1.container()
visual2_area = col2.container()
visual3_area = st.container()

# ----- Recommendations -----
with results_area:
    st.subheader("Recommendations")
    if run_btn:
        profile_vec = build_user_vector(
            X=X,
            cols_meta=cols,
            title_to_id=idx["title_to_id"],
            favorites=fav_sel,
            fav_genres=genre_sel,
            mood_map=mood,
            mood_choice=None if mood_choice == "(none)" else mood_choice
        )

        mask = filter_candidates(df, rating_min, year_range, genre_sel)
        candidates = np.where(mask.values)[0]
        fav_ids = {idx["title_to_id"][t] for t in fav_sel if t in idx["title_to_id"]}

        picked_idx, table = rerank(
            profile_vec, X, df, cols,
            candidate_idx=candidates,
            top_n=top_n,
            exclude_ids=fav_ids
        )

        # Keep essentials for visuals
        st.session_state["picked_idx"] = picked_idx
        st.session_state["profile_vec"] = profile_vec
        st.session_state["results_table"] = table  # raw (un-pretty) table with original columns
        st.session_state["show_visuals"] = True


        if table.empty:
            st.warning("No candidates matched your filters. Try relaxing the rating/year/genre constraints.")
        else:
            # ----- User-friendly presentation -----
            # Truncate long descriptions for a tidy table
            def tidy_desc(s, maxlen=220):
                if not isinstance(s, str):
                    return ""
                s = s.strip()
                return (s[:maxlen-1] + "…") if len(s) > maxlen else s

            pretty = table.rename(columns={
                "title": "Title",
                "image": "Image",
                "synopsis": "Description",
                "year": "Year",
                "score": "Rating",
                "popularity": "Popularity Rank",
                "members": "Members",
                "genres_raw": "Genres",
                "themes_raw": "Themes",
                "demographics_raw": "Demographics",
            }).copy()

            # blank instead of NaN
            pretty = pretty.replace({np.nan: "", pd.NA: "", None: ""})
            pretty["Description"] = pretty["Description"].apply(tidy_desc)

            # 1-based row numbers
            pretty.insert(0, "#", np.arange(1, len(pretty) + 1))

            # Select/Order columns for end users
            display_cols = ["#", "Image", "Title", "Year", "Rating", "Popularity Rank",
                            "Members", "Genres", "Themes", "Demographics", "Description"]
            pretty = pretty[[c for c in display_cols if c in pretty.columns]]

            # Render images inside the dataframe
            st.dataframe(
                pretty,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Image": st.column_config.ImageColumn(
                        "Image",
                        help="Poster/thumbnail",
                        width="small"
                    ),
                    "Description": st.column_config.TextColumn(
                        "Description",
                        help="Brief synopsis"
                    ),
                    "Popularity Rank": st.column_config.NumberColumn(
                        "Popularity Rank",
                        help="Lower number = more popular (1 is most popular)"
                    )
                }
            )
    else:
        st.info("Use the sidebar to set favorites/genres/filters/mood, then click **Get Recommendations**.")

    


# ----- Visual placeholders -----
# -------- Visual 1: Cluster Map (PCA-2D) --------
# -------- Visual 1: Cluster Map (PCA-2D) --------
with visual1_area:
    if st.session_state.get("show_visuals"):
        import matplotlib.pyplot as plt
        st.subheader("Visual 1: Cluster Map (PCA-2D)")
        coords2d = art["pca2d"]
        fig, ax = plt.subplots()
        ax.scatter(coords2d[:,0], coords2d[:,1], s=6, alpha=0.25)  # all points
        hi = np.array(st.session_state["picked_idx"], dtype=int)
        ax.scatter(coords2d[hi,0], coords2d[hi,1], s=50, edgecolors="black")  # highlight recs
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Anime clusters (K-Means)")
        st.pyplot(fig)

# -------- Visual 2: Genre × Rating Heatmap --------
with visual2_area:
    if st.session_state.get("show_visuals"):
        import matplotlib.pyplot as plt
        st.subheader("Visual 2: Genre × Rating Heatmap (avg rating)")
        genre_cols = cols.get("genre_cols", [])[:20]
        if genre_cols:
            labels = [g.replace("genre__","") for g in genre_cols]
            vals = []
            for gc in genre_cols:
                m = df[gc] > 0.0
                vals.append(df.loc[m, "score"].mean() if m.any() else np.nan)
            arr = np.array(vals).reshape(-1,1)
            fig2, ax2 = plt.subplots()
            im = ax2.imshow(arr, aspect="auto")
            ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
            ax2.set_xticks([0]); ax2.set_xticklabels(["Avg Rating"])
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)

# -------- Visual 3: Why this recommendation? --------
with visual3_area:
    if st.session_state.get("show_visuals"):
        import matplotlib.pyplot as plt
        st.subheader("Visual 3: Why this recommendation?")
        raw_table = st.session_state["results_table"]
        titles = raw_table["title"].astype(str).tolist()
        pick = st.selectbox("Choose a recommended title", titles, index=0)
        i = idx["title_to_id"].get(pick, None)
        if i is not None:
            profile = st.session_state["profile_vec"]
            feat_cols = cols["feat_cols"]
            tag_cols = cols.get("genre_cols", []) + cols.get("theme_cols", []) + cols.get("demo_cols", [])
            contrib = []
            for c in tag_cols:
                j = feat_cols.index(c)
                contrib.append((c.replace("genre__","").replace("theme__","").replace("demo__",""),
                                float(profile[j] * X[i, j])))
            contrib = sorted(contrib, key=lambda x: x[1], reverse=True)[:10]
            labels = [c for c,_ in contrib]
            scores = [s for _,s in contrib]
            fig3, ax3 = plt.subplots()
            ax3.barh(labels[::-1], scores[::-1])
            ax3.set_xlabel("Contribution"); ax3.set_ylabel("Tag")
            ax3.set_title(f"Top tag contributions for: {pick}")
            st.pyplot(fig3)

