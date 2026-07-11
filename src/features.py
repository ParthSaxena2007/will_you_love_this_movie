"""
features.py — Feature engineering for the personal rating predictor.

Design notes
------------
* All "taste statistics" (director mean, genre mean) are computed with
  out-of-fold or smoothed leave-one-out logic during training to prevent
  target leakage. At inference time the full-data statistics are used.
* Works on the native IMDb export columns; transparently uses enriched
  columns (cast, keywords, overview, language, country) when present.
"""
import numpy as np
import pandas as pd

LIKE_THRESHOLD = 8  # user's mode is 8; >=8 means "liked"
CURRENT_YEAR = 2026

# ----------------------------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Title Type"] == "Movie"].copy()          # movies only
    df["Date Rated"] = pd.to_datetime(df["Date Rated"], format="%d-%m-%Y", errors="coerce")
    df["Genres List"] = (
        df["Genres"].fillna("").str.split(",").apply(lambda xs: [g.strip() for g in xs if g.strip()])
    )
    df["Directors List"] = (
        df["Directors"].fillna("").str.split(",").apply(lambda xs: [d.strip() for d in xs if d.strip()])
    )
    df["Runtime (mins)"] = df["Runtime (mins)"].fillna(df["Runtime (mins)"].median())
    df = df.sort_values("Date Rated").reset_index(drop=True)
    df["rating_order"] = np.arange(len(df))               # drift feature
    df["like"] = (df["Your Rating"] >= LIKE_THRESHOLD).astype(int)
    return df

# ----------------------------------------------------------------------
def top_genres(df: pd.DataFrame, k: int = 18):
    from collections import Counter
    c = Counter(g for gl in df["Genres List"] for g in gl)
    return [g for g, _ in c.most_common(k)]

# ----------------------------------------------------------------------
def smoothed_mean(series_sum, series_cnt, global_mean, alpha=5.0):
    """Bayesian-smoothed mean: pulls low-count categories toward global mean."""
    return (series_sum + alpha * global_mean) / (series_cnt + alpha)

# ----------------------------------------------------------------------
class TasteEncoder:
    """
    Computes leakage-safe director & genre taste statistics.

    fit_transform(train_df)   -> per-row leave-one-out encodings
    transform(new_df/rows)    -> full-data encodings for inference
    """
    def __init__(self, alpha_director=4.0, alpha_genre=12.0):
        self.alpha_d = alpha_director
        self.alpha_g = alpha_genre

    # ---------- fitting -------------------------------------------------
    def fit(self, df: pd.DataFrame):
        self.global_mean_ = df["Your Rating"].mean()

        # director sums/counts
        d_sum, d_cnt = {}, {}
        for _, r in df.iterrows():
            for d in r["Directors List"]:
                d_sum[d] = d_sum.get(d, 0.0) + r["Your Rating"]
                d_cnt[d] = d_cnt.get(d, 0) + 1
        self.d_sum_, self.d_cnt_ = d_sum, d_cnt

        # genre sums/counts
        g_sum, g_cnt = {}, {}
        for _, r in df.iterrows():
            for g in r["Genres List"]:
                g_sum[g] = g_sum.get(g, 0.0) + r["Your Rating"]
                g_cnt[g] = g_cnt.get(g, 0) + 1
        self.g_sum_, self.g_cnt_ = g_sum, g_cnt
        return self

    # ---------- helpers --------------------------------------------------
    def _director_score(self, dirs, exclude_rating=None):
        vals = []
        for d in dirs:
            s, c = self.d_sum_.get(d, 0.0), self.d_cnt_.get(d, 0)
            if exclude_rating is not None and c > 0:
                s, c = s - exclude_rating, c - 1          # leave-one-out
            vals.append(smoothed_mean(s, c, self.global_mean_, self.alpha_d))
        return float(np.mean(vals)) if vals else self.global_mean_

    def _director_count(self, dirs, exclude=False):
        cnts = [max(self.d_cnt_.get(d, 0) - (1 if exclude else 0), 0) for d in dirs]
        return float(max(cnts)) if cnts else 0.0

    def _genre_score(self, genres, exclude_rating=None):
        vals = []
        for g in genres:
            s, c = self.g_sum_.get(g, 0.0), self.g_cnt_.get(g, 0)
            if exclude_rating is not None and c > 0:
                s, c = s - exclude_rating, c - 1
            vals.append(smoothed_mean(s, c, self.global_mean_, self.alpha_g))
        return float(np.mean(vals)) if vals else self.global_mean_

    # ---------- API -------------------------------------------------------
    def encode_row(self, row, loo=False):
        ex = row["Your Rating"] if loo else None
        return {
            "director_taste": self._director_score(row["Directors List"], ex),
            "director_seen_count": self._director_count(row["Directors List"], exclude=loo),
            "genre_taste": self._genre_score(row["Genres List"], ex),
        }

# ----------------------------------------------------------------------
def build_matrix(df: pd.DataFrame, genre_vocab, taste: TasteEncoder, loo: bool):
    """Return (X DataFrame, feature names)."""
    rows = []
    for _, r in df.iterrows():
        f = {}
        # --- core numeric
        f["imdb_rating"] = r["IMDb Rating"]
        f["log_votes"] = np.log1p(r["Num Votes"])
        f["runtime"] = r["Runtime (mins)"]
        f["year"] = r["Year"]
        f["movie_age"] = CURRENT_YEAR - r["Year"]
        f["genre_count"] = len(r["Genres List"])
        f["rating_order"] = r.get("rating_order", np.nan)
        # --- genre one-hots
        gs = set(r["Genres List"])
        for g in genre_vocab:
            f[f"g_{g}"] = int(g in gs)
        # --- notable pair interactions (from EDA)
        f["gx_scifi_thriller"] = int("Sci-Fi" in gs and "Thriller" in gs)
        f["gx_comedy_romance"] = int("Comedy" in gs and "Romance" in gs)
        f["gx_crime_mystery"] = int("Crime" in gs and "Mystery" in gs)
        # --- taste encodings (leakage-safe when loo=True)
        f.update(taste.encode_row(r, loo=loo))
        # --- IMDb-relative
        f["imdb_x_votes"] = f["imdb_rating"] * f["log_votes"]
        rows.append(f)
    X = pd.DataFrame(rows, index=df.index)
    return X

# ----------------------------------------------------------------------
def prepare_training(csv_path: str, k_genres: int = 18):
    df = load_and_clean(csv_path)
    vocab = top_genres(df, k_genres)
    taste = TasteEncoder().fit(df)
    X = build_matrix(df, vocab, taste, loo=True)   # leave-one-out -> no leakage
    y = df["Your Rating"].values.astype(float)
    y_like = df["like"].values
    return df, X, y, y_like, vocab, taste
