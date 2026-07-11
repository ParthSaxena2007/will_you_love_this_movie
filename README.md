# Will I Like It? — Personal Movie Rating Predictor

Trained on Parth's IMDb ratings export (414 movies). Predicts the rating you'd
give any movie, the probability you'll like it (rating ≥ 8), a confidence
score, a Watch/Maybe/Skip call, SHAP-based reasons, and the closest movies
from your own history.

## Honest performance (repeated 5-fold cross-validation, leakage-safe)

| Metric | Value |
|---|---|
| MAE (regression) | **0.94** points |
| RMSE | 1.26 |
| R² | 0.37 |
| Predictions within ±1 point | 64.7% |
| ROC-AUC (like ≥ 8 classifier) | 0.72 |
| F1 (like) | 0.68 |

Baselines beaten: "predict my mean" (MAE 1.21) and "IMDb rating + my average
offset" (MAE 1.06). The model adds ~10% real accuracy over the naive strategy.

**Interpretation:** with 414 training movies and no plot/cast metadata,
±1 point is the honest resolution. The classifier is well calibrated: when it
says 70% chance you'll like something, you like it about 70% of the time.
The single strongest signal in your taste is **director history** — your past
ratings of a director's films predict your next one better than any genre.

## Architecture

```
imdb_data.csv ──► features.py ──► train_final.py ──► models/rating_predictor.joblib
                     │                                        │
   (leakage-safe LOO taste encoding:                 predict.py / app.py
    director mean, genre mean, drift,                  │
    genre one-hots, popularity, runtime)      enrich.py (TMDb, your machine)
```

* **Model:** stacking ensemble — GradientBoosting + XGBoost + Ridge, blended by
  Ridge meta-learner. Won a 12-model tournament (see `reports/model_tournament.csv`).
* **Leakage protection:** director/genre taste features use leave-one-out
  encoding during training. Naive encoding would fake ~0.3 MAE improvement.
* **Drift feature:** you've grown ~0.8 points more generous over time; the
  model knows new predictions happen in your "generous era".
* **Classifier:** isotonic-calibrated GradientBoosting for P(like).
* **Explanations:** SHAP TreeExplainer per prediction.

## Files

```
src/features.py        feature engineering + TasteEncoder
src/train_compare.py   Phase 4 model tournament (12 models + 2 baselines)
src/train_final.py     tuning, final training, evaluation, SHAP, bundle export
src/predict.py         Predictor class + CLI
src/enrich.py          TMDb enrichment with caching (needs your machine + API key)
app.py                 Streamlit web app
models/rating_predictor.joblib   trained bundle
reports/               tournament results, metrics, SHAP summary, eval plots
```

## Run it on your laptop

```bash
pip install -r requirements.txt
python src/predict.py "The Prestige" --imdb 8.5 --votes 1500000 \
    --runtime 130 --year 2006 --genres "Drama,Mystery,Thriller" \
    --directors "Christopher Nolan"
```

Web app with automatic metadata lookup:
```bash
export TMDB_API_KEY=...        # free key from themoviedb.org
streamlit run app.py
```

## Deploy (Streamlit Cloud, free)

1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io → New app → pick the repo, main file `app.py`.
3. Add `TMDB_API_KEY` in App secrets.

## Continuous learning (Phase 8)

Re-export your ratings CSV from IMDb, replace the file, and run:
```bash
python src/train_final.py
```
Takes ~2 minutes; the app picks up the new bundle automatically on restart.

## Known limitations (read this)

1. **414 movies is small.** MAE will improve as you rate more; expect real
   gains at ~800+ ratings.
2. **No cast/plot/keyword features yet** in the trained model — the sandbox
   couldn't reach TMDb. Run `python src/enrich.py` on your machine, then
   retrain; expect a further 5-10% MAE improvement from cast/keyword/language
   features (the feature code activates automatically when columns exist).
3. **Selection bias:** you rate movies you *chose* to watch. The model has
   never seen "movies you avoided", so SKIP predictions are extrapolations.
4. **Episode/series excluded** — the model is movies-only by design; episode
   ratings follow a different psychology (you only log shows you love).
5. **In-dataset titles score too well.** Predicting a movie you already rated
   partially "remembers" it through director/genre averages. Judge the model
   on movies you haven't rated.
