"""
predict.py — Phase 6: prediction interface.

Usage (programmatic):
    from predict import Predictor
    p = Predictor("models/rating_predictor.joblib")
    p.predict_title("Dream Girl 2")                  # title already in your ratings
    p.predict_metadata(dict(title="Interstellar",    # any new movie
                            imdb_rating=8.7, num_votes=2_400_000, runtime=169,
                            year=2014, genres=["Adventure","Drama","Sci-Fi"],
                            directors=["Christopher Nolan"]))

CLI:
    python src/predict.py "Movie Title" --imdb 8.7 --votes 2400000 \
        --runtime 169 --year 2014 --genres "Adventure,Drama,Sci-Fi" \
        --directors "Christopher Nolan"
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_matrix, CURRENT_YEAR


class Predictor:
    def __init__(self, bundle_path):
        b = joblib.load(bundle_path)
        self.stack, self.clf = b["stack"], b["clf"]
        self.gb_shap = b["gb_shap"]
        self.taste, self.vocab = b["taste"], b["vocab"]
        self.features = b["feature_names"]
        self.user_mean = b["user_mean"]
        self.max_order = b["max_rating_order"]
        self.ratings = b["ratings_df"]
        self.metrics = b["metrics_regression"]
        self.explainer = shap.TreeExplainer(self.gb_shap)
        # pre-compute training feature matrix for similarity retrieval
        self._Xtrain = build_matrix(self.ratings.assign(rating_order=self.max_order),
                                    self.vocab, self.taste, loo=False)
        self._mu = self._Xtrain.mean()
        self._sd = self._Xtrain.std().replace(0, 1)
        self._Xtrain_norm = (self._Xtrain - self._mu) / self._sd

    # ------------------------------------------------------------------
    def _normalize(self, X):
        return (X - self._mu) / self._sd

    def _row_from_metadata(self, md):
        return pd.DataFrame([{
            "IMDb Rating": md["imdb_rating"],
            "Num Votes": md["num_votes"],
            "Runtime (mins)": md.get("runtime", 120),
            "Year": md.get("year", CURRENT_YEAR),
            "Genres List": md.get("genres", []),
            "Directors List": md.get("directors", []),
            "rating_order": self.max_order + 1,     # "you'd rate it next"
            "Your Rating": np.nan,
        }])

    # ------------------------------------------------------------------
    def _humanize(self, name, value, shap_val, md):
        """Translate one raw feature/SHAP pair into a plain-English sentence.
        Returns None for features not worth surfacing to a normal user
        (they still appear in the raw/advanced list)."""
        up = shap_val > 0
        genres = md.get("genres") or []
        dirs = md.get("directors") or []

        if name == "director_taste":
            d = dirs[0] if dirs else "This film's director"
            return (f"{d} is one of your highest-rated directors."
                    if up else f"{d}'s films haven't historically been a strong match for you.")
        if name == "director_seen_count":
            if value >= 3:
                return ("You've rated several other films from this director before."
                        if up else "You've already rated several of this director's films — "
                                    "this one doesn't shift the picture much either way.")
            return ("This is fairly uncharted territory for this director, but early signs are positive."
                    if up else "You have little history with this director, which adds some uncertainty.")
        if name == "genre_taste":
            g = genres[0] if genres else "this genre"
            return (f"You consistently enjoy {g} films."
                    if up else f"{g} hasn't historically been a strong genre for you.")
        if name.startswith("g_"):
            g = name[2:]
            if g not in genres:
                return None
            return (f"You consistently enjoy {g} films."
                    if up else f"{g} films tend to underperform in your ratings.")
        if name in ("gx_scifi_thriller", "gx_comedy_romance", "gx_crime_mystery"):
            pretty = {"gx_scifi_thriller": "Sci-Fi/Thriller", "gx_comedy_romance": "Comedy/Romance",
                      "gx_crime_mystery": "Crime/Mystery"}[name]
            return (f"You respond well to the {pretty} combination."
                    if up else f"The {pretty} mix hasn't landed for you in the past.")
        if name == "imdb_rating":
            if value >= 8:
                return ("Movies with IMDb ratings above 8 usually score well with you." if up else None)
            if value < 6.5:
                return (None if up else "Lower IMDb-rated movies are a weaker match for your taste.")
            return None
        if name == "runtime":
            return ("The runtime matches movies you often enjoy."
                    if up else "The runtime is a bit outside your usual sweet spot.")
        if name in ("log_votes", "imdb_x_votes"):
            votes = md.get("num_votes", 0)
            if votes >= 200_000:
                return ("This is a widely-watched, highly-voted film — that tends to align with your taste."
                        if up else "Even big, widely-seen films like this haven't always been your favorites.")
            return ("Even without a huge vote count, this one still fits your taste profile."
                    if up else "Lower-profile films like this tend to be a slightly weaker match for you.")
        if name in ("movie_age", "year"):
            return ("This fits the era of film you tend to enjoy."
                    if up else "This is a bit outside the era you usually gravitate toward.")
        if name == "genre_count":
            return ("This movie blends several genres, which you tend to enjoy."
                    if up else "This movie sticks to a single genre, less typical of your favorites.")
        return None  # e.g. rating_order (drift) — not user-facing

    # ------------------------------------------------------------------
    def predict_metadata(self, md, top_reasons=5, n_similar=5):
        row = self._row_from_metadata(md)
        X = build_matrix(row, self.vocab, self.taste, loo=False)[self.features]

        pred = float(np.clip(self.stack.predict(X)[0], 1, 10))
        p_like = float(self.clf.predict_proba(X)[0, 1])

        # confidence: agreement of stack components + data support
        comp_preds = [np.clip(est.predict(X)[0], 1, 10) for _, est in self.stack.named_estimators_.items()]
        spread = float(np.std(comp_preds))
        director_seen = X["director_seen_count"].iloc[0]
        support = min(1.0, 0.5 + 0.1 * director_seen)      # more history with director => more confident
        confidence = float(np.clip(support * (1 - spread / 2.5), 0.05, 0.95))

        # recommendation
        if pred >= 7.8 and p_like >= 0.60: rec = "WATCH"
        elif pred >= 6.8 or p_like >= 0.45: rec = "MAYBE"
        else: rec = "SKIP"

        # SHAP reasons — raw (for Advanced Details) + humanized (for the main view)
        sv = self.explainer.shap_values(X)[0]
        order = np.argsort(np.abs(sv))[::-1]
        reasons_pos_raw, reasons_neg_raw = [], []
        reasons_pos_en, reasons_neg_en = [], []
        for i in order:
            name, val, s = self.features[i], X.iloc[0, i], sv[i]
            if abs(s) < 0.02: break
            raw_txt = f"{name} = {val:.2f} ({'+' if s > 0 else ''}{s:.2f})"
            (reasons_pos_raw if s > 0 else reasons_neg_raw).append(raw_txt)
            english = self._humanize(name, val, s, md)
            if english and english not in reasons_pos_en and english not in reasons_neg_en:
                (reasons_pos_en if s > 0 else reasons_neg_en).append(english)
            if len(reasons_pos_raw) >= top_reasons and len(reasons_neg_raw) >= top_reasons: break

        # similar movies from user history (cosine in normalized feature space)
        xq = self._normalize(X).values[0]
        M = self._Xtrain_norm.values
        cos = M @ xq / (np.linalg.norm(M, axis=1) * np.linalg.norm(xq) + 1e-9)
        top = np.argsort(cos)[::-1][:n_similar]
        similar = [
            {"title": self.ratings.iloc[i]["Title"],
             "year": int(self.ratings.iloc[i]["Year"]),
             "your_rating": int(self.ratings.iloc[i]["Your Rating"]),
             "similarity": round(float(cos[i]), 3)}
            for i in top
        ]

        return {
            "title": md.get("title", "?"),
            "predicted_rating": round(pred, 2),
            "probability_like": round(p_like, 3),
            "confidence": round(confidence, 2),
            "recommendation": rec,
            "reasons_positive_english": reasons_pos_en[:top_reasons],
            "reasons_negative_english": reasons_neg_en[:top_reasons],
            "reasons_positive": reasons_pos_raw[:top_reasons],
            "reasons_negative": reasons_neg_raw[:top_reasons],
            "similar_movies_from_your_history": similar,
            "model_MAE_note": f"OOF MAE {self.metrics['MAE']:.2f} — treat prediction as ±1 point",
            "oof_mae": round(self.metrics["MAE"], 2),
            "training_size": len(self.ratings),
            "model_used": "Stacking ensemble (GradientBoosting + XGBoost + Ridge)",
        }

    # ------------------------------------------------------------------
    def predict_title(self, title):
        """Look up a title inside your own ratings export (sanity/demo mode)."""
        hit = self.ratings[self.ratings["Title"].str.lower() == title.lower()]
        if hit.empty:
            hit = self.ratings[self.ratings["Title"].str.lower().str.contains(title.lower())]
        if hit.empty:
            return {"error": f"'{title}' not found in your ratings. Use predict_metadata() "
                             f"or run the enrichment module to fetch metadata by title."}
        r = hit.iloc[0]
        md = dict(title=r["Title"], imdb_rating=r["IMDb Rating"], num_votes=1,
                  genres=r["Genres List"], directors=r["Directors List"], year=int(r["Year"]))
        # use real vote count from stored df if available
        out = self.predict_metadata(md)
        out["actual_rating_you_gave"] = int(r["Your Rating"])
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("title")
    ap.add_argument("--imdb", type=float); ap.add_argument("--votes", type=int, default=100000)
    ap.add_argument("--runtime", type=float, default=120); ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--genres", type=str, default=""); ap.add_argument("--directors", type=str, default="")
    a = ap.parse_args()
    p = Predictor(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "rating_predictor.joblib"))
    if a.imdb is None:
        print(json.dumps(p.predict_title(a.title), indent=2))
    else:
        md = dict(title=a.title, imdb_rating=a.imdb, num_votes=a.votes, runtime=a.runtime,
                  year=a.year, genres=[g.strip() for g in a.genres.split(",") if g.strip()],
                  directors=[d.strip() for d in a.directors.split(",") if d.strip()])
        print(json.dumps(p.predict_metadata(md), indent=2))
