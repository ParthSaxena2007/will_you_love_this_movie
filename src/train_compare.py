"""
train_compare.py — Phase 4/5: model tournament with honest baselines.

Every model is evaluated with repeated 5-fold CV on identical folds.
Baselines included so we know whether ML is adding anything:
  B0: predict the user's global mean
  B1: predict IMDb rating + user's mean offset
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/rating_predictor/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, StackingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from features import prepare_training

CSV = "/mnt/user-data/uploads/imdb_data.csv"
df, X, y, y_like, vocab, taste = prepare_training(CSV)
print(f"Movies: {len(df)} | Features: {X.shape[1]}")

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

def cv_scores(model, X, y):
    maes, rmses, r2s = [], [], []
    for tr, te in cv.split(X):
        m = model
        from sklearn.base import clone
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        p = np.clip(m.predict(X.iloc[te]), 1, 10)
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        r2s.append(r2_score(y[te], p))
    return np.mean(maes), np.std(maes), np.mean(rmses), np.mean(r2s)

results = {}

# ---------------- Baselines ----------------
# B0: global mean
maes = []
for tr, te in cv.split(X):
    pred = np.full(len(te), y[tr].mean())
    maes.append(mean_absolute_error(y[te], pred))
results["B0: predict my mean"] = (np.mean(maes), np.std(maes), np.nan, 0.0)

# B1: IMDb + offset
maes, rmses, r2s = [], [], []
for tr, te in cv.split(X):
    offset = (y[tr] - X["imdb_rating"].values[tr]).mean()
    pred = np.clip(X["imdb_rating"].values[te] + offset, 1, 10)
    maes.append(mean_absolute_error(y[te], pred))
    rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
    r2s.append(r2_score(y[te], pred))
results["B1: IMDb + my offset"] = (np.mean(maes), np.std(maes), np.mean(rmses), np.mean(r2s))

# ---------------- Models ----------------
models = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
    "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.3)),
    "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                  max_depth=3, subsample=0.8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=3,
                            subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                            random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                              random_state=42, verbose=-1),
}

for name, model in models.items():
    mae, sd, rmse, r2 = cv_scores(model, X, y)
    results[name] = (mae, sd, rmse, r2)
    print(f"{name:22s} MAE {mae:.3f} ±{sd:.3f} | RMSE {rmse:.3f} | R² {r2:.3f}")

# ---------------- Stacking (top learners) ----------------
stack = StackingRegressor(
    estimators=[
        ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                         subsample=0.8, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.8,
                             colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbosity=0)),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=5.0))),
    ],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1,
)
mae, sd, rmse, r2 = cv_scores(stack, X, y)
results["Stacking (GB+XGB+Ridge)"] = (mae, sd, rmse, r2)
print(f"{'Stacking':22s} MAE {mae:.3f} ±{sd:.3f} | RMSE {rmse:.3f} | R² {r2:.3f}")

# ---------------- Save results ----------------
out = pd.DataFrame(
    [(k, *v) for k, v in results.items()],
    columns=["Model", "MAE", "MAE_sd", "RMSE", "R2"]
).sort_values("MAE")
out.to_csv("/home/claude/rating_predictor/reports/model_tournament.csv", index=False)
print("\n=== Ranked ===")
print(out.to_string(index=False))
