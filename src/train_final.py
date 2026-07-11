"""
train_final.py — tune top models, train final stack, build classifier,
generate evaluation report + SHAP explanations, save deployable bundle.
"""
import os, sys, warnings, json, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             brier_score_loss)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

from features import prepare_training, TasteEncoder, build_matrix, top_genres

CSV = "data/imdb_data.csv"
OUT = "."

df, X, y, y_like, vocab, taste = prepare_training(CSV)
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# 1. Hyperparameter tuning on the top-3 components of the stack
# =====================================================================
print("Tuning Ridge alpha...")
ridge_gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                        {"ridge__alpha": [1, 3, 5, 10, 20, 40]},
                        scoring="neg_mean_absolute_error", cv=cv5).fit(X, y)
print("  best:", ridge_gs.best_params_, f"MAE {-ridge_gs.best_score_:.3f}")

print("Tuning GradientBoosting...")
gb_gs = GridSearchCV(GradientBoostingRegressor(random_state=42),
                     {"n_estimators": [200, 300, 500],
                      "learning_rate": [0.03, 0.05],
                      "max_depth": [2, 3],
                      "subsample": [0.8]},
                     scoring="neg_mean_absolute_error", cv=cv5, n_jobs=-1).fit(X, y)
print("  best:", gb_gs.best_params_, f"MAE {-gb_gs.best_score_:.3f}")

print("Tuning XGBoost...")
xgb_gs = GridSearchCV(XGBRegressor(random_state=42, verbosity=0, subsample=0.8),
                      {"n_estimators": [300, 500],
                       "learning_rate": [0.03, 0.05],
                       "max_depth": [2, 3],
                       "colsample_bytree": [0.7],
                       "reg_lambda": [1.0, 3.0]},
                      scoring="neg_mean_absolute_error", cv=cv5, n_jobs=-1).fit(X, y)
print("  best:", xgb_gs.best_params_, f"MAE {-xgb_gs.best_score_:.3f}")

# =====================================================================
# 2. Final stacked regressor with tuned components
# =====================================================================
stack = StackingRegressor(
    estimators=[
        ("gb", GradientBoostingRegressor(random_state=42, **gb_gs.best_params_)),
        ("xgb", XGBRegressor(random_state=42, verbosity=0, subsample=0.8, **xgb_gs.best_params_)),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=ridge_gs.best_params_["ridge__alpha"]))),
    ],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1,
)

# ----- honest CV metrics for the final architecture
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
oof_pred = cross_val_predict(stack, X, y, cv=cv5, n_jobs=-1)
oof_pred = np.clip(oof_pred, 1, 10)
final_metrics = {
    "MAE": float(mean_absolute_error(y, oof_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y, oof_pred))),
    "R2": float(r2_score(y, oof_pred)),
    "within_1_point_pct": float(np.mean(np.abs(y - oof_pred) <= 1) * 100),
}
print("\nFinal stack OOF:", json.dumps(final_metrics, indent=2))

# =====================================================================
# 3. Like/dislike classifier (calibrated)
# =====================================================================
clf_base = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=2, subsample=0.8, random_state=42)
clf = CalibratedClassifierCV(clf_base, method="isotonic", cv=5)
oof_proba = cross_val_predict(clf, X, y_like, cv=cv5, method="predict_proba", n_jobs=-1)[:, 1]
oof_cls = (oof_proba >= 0.5).astype(int)
clf_metrics = {
    "ROC_AUC": float(roc_auc_score(y_like, oof_proba)),
    "Precision": float(precision_score(y_like, oof_cls)),
    "Recall": float(recall_score(y_like, oof_cls)),
    "F1": float(f1_score(y_like, oof_cls)),
    "Brier": float(brier_score_loss(y_like, oof_proba)),
    "base_rate_like": float(y_like.mean()),
}
print("Classifier OOF:", json.dumps(clf_metrics, indent=2))

# =====================================================================
# 4. Fit final models on ALL data (production models)
# =====================================================================
# For production, taste encodings use FULL data (no LOO) — legitimate at inference.
X_full = build_matrix(df, vocab, taste, loo=False)
stack.fit(X_full, y)
clf.fit(X_full, y_like)

# GB alone for SHAP (tree explainer needs a tree model)
gb_shap = GradientBoostingRegressor(random_state=42, **gb_gs.best_params_).fit(X_full, y)

# =====================================================================
# 5. Explanations: feature importance, permutation, SHAP
# =====================================================================
expl = shap.TreeExplainer(gb_shap)
shap_vals = expl.shap_values(X_full)

plt.figure()
shap.summary_plot(shap_vals, X_full, show=False, max_display=18)
plt.tight_layout(); plt.savefig(f"{OUT}/reports/shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()

perm = permutation_importance(stack, X_full, y, n_repeats=8, random_state=42,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
perm_df = pd.DataFrame({"feature": X_full.columns,
                        "importance": perm.importances_mean}).sort_values("importance", ascending=False)
perm_df.to_csv(f"{OUT}/reports/permutation_importance.csv", index=False)
print("\nTop 12 permutation importances:")
print(perm_df.head(12).to_string(index=False))

# ----- residual + calibration plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].scatter(oof_pred, y - oof_pred, alpha=0.4, s=18)
axes[0].axhline(0, color="red", ls="--"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs predicted (OOF)")
axes[1].scatter(y, oof_pred, alpha=0.4, s=18)
axes[1].plot([1, 10], [1, 10], "r--"); axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
axes[1].set_title(f"Actual vs predicted (MAE {final_metrics['MAE']:.2f})")
bins = np.linspace(0, 1, 11); binids = np.digitize(oof_proba, bins) - 1
frac = [y_like[binids == b].mean() if (binids == b).sum() > 4 else np.nan for b in range(10)]
axes[2].plot(bins[:-1] + 0.05, frac, "o-"); axes[2].plot([0, 1], [0, 1], "r--")
axes[2].set_xlabel("Predicted P(like)"); axes[2].set_ylabel("Observed frequency")
axes[2].set_title("Classifier calibration")
plt.tight_layout(); plt.savefig(f"{OUT}/reports/evaluation_plots.png", dpi=130); plt.close()

# =====================================================================
# 6. Save deployable bundle
# =====================================================================
bundle = {
    "stack": stack, "clf": clf, "gb_shap": gb_shap, "explainer_bg": X_full.iloc[:50],
    "taste": taste, "vocab": vocab, "feature_names": list(X_full.columns),
    "metrics_regression": final_metrics, "metrics_classifier": clf_metrics,
    "train_rows": len(df), "like_threshold": 8,
    "user_mean": float(y.mean()),
    "max_rating_order": int(df["rating_order"].max()),
    "ratings_df": df[["Const", "Title", "Year", "Your Rating", "IMDb Rating",
                      "Num Votes", "Runtime (mins)", "Genres", "Directors",
                      "Genres List", "Directors List", "rating_order"]],
}
joblib.dump(bundle, f"{OUT}/models/rating_predictor.joblib")
json.dump({"regression": final_metrics, "classification": clf_metrics},
          open(f"{OUT}/reports/metrics.json", "w"), indent=2)
print(f"\nBundle saved: {OUT}/models/rating_predictor.joblib")
