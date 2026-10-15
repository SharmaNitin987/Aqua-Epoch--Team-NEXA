from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "water_quality.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Potability"
INVALID_IF_NEGATIVE = ["pH", "Hardness", "Fluoride"]

def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found. Available columns: {df.columns.tolist()}")

    # The target is required for supervised training. Drop rows where it is missing.
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No rows with a valid 0/1 Potability label were found.")

    X = df.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)

    # Impossible negative measurements are treated as missing and imputed by the pipeline.
    for col in INVALID_IF_NEGATIVE:
        if col in X.columns:
            X.loc[X[col] < 0, col] = float("nan")

    # Remove columns that are entirely empty; retain the exact feature list for the app.
    X = X.dropna(axis=1, how="all")
    features = X.columns.tolist()
    if len(features) == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.08, random_state=42
        ),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        try:
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except (AttributeError, ValueError):
            auc = float("nan")
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": auc,
        }
        results.append(metrics)
        fitted[name] = (pipe, metrics)
        print(f"\n{name}")
        print(classification_report(y_test, pred, zero_division=0))
        print("Confusion matrix [rows=actual 0/1, columns=predicted 0/1]:")
        print(confusion_matrix(y_test, pred))
        print("Metrics:", metrics)

    # Prioritize F1 for the positive (potable=1) class; report other metrics too.
    best_name = max(results, key=lambda m: m["f1"])["model"]
    best_pipe, best_metrics = fitted[best_name]
    joblib.dump(best_pipe, MODEL_DIR / "water_potability_model.joblib")

    medians = X_train.median(numeric_only=True).reindex(features).fillna(0)
    metadata = {
        "features": features,
        "defaults": {k: float(v) for k, v in medians.items()},
        "best_model": best_name,
        "metrics": best_metrics,
        "target_meanings": {"0": "Not potable (according to dataset label)", "1": "Potable (according to dataset label)"},
        "rows_used": int(len(df)),
        "rows_dropped_missing_target": int(pd.read_csv(DATA_PATH)[TARGET].isna().sum()),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(ROOT / "model_comparison.csv", index=False)
    print(f"\nBest model by positive-class F1: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'water_potability_model.joblib'}")
    print(f"Saved metrics to: {ROOT / 'model_comparison.csv'}")

if __name__ == "__main__":
    main()
