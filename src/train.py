"""Step 5 - train, compare models, evaluate, save the best one.  Run:  python -m src.train"""
import json
import mlflow
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL,
    CLEAN_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC,
    RANDOM_STATE,
    TARGET,
    THRESHOLD,
)


def load_data(path=CLEAN_PATH):
    df = pd.read_csv(path)
    return df[NUMERIC + CATEGORICAL], df[TARGET]


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Churn rate: {y.mean():.1%}")

    # Compare models with cross-validation on the TRAIN set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, clf in candidates().items():
        scores = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())
        print(f"{name:<24} CV ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest model: {best_name}")

    # Refit the winner on all training data, score once on the untouched test set
    model = build_pipeline(candidates()[best_name]).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "best_model": best_name,
        "cv_roc_auc": cv_scores,
        "test": {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn")
    with mlflow.start_run(run_name=best_name):
        mlflow.log_params(candidates()[best_name].get_params())
        mlflow.log_param("model", best_name)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        for name, score in cv_scores.items():
            mlflow.log_metric(f"cv_roc_auc_{name}", score)
        for key, value in metrics["test"].items():
            if key != "confusion_matrix":
                mlflow.log_metric(f"test_{key}", value)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

    print("\nTest set results:")
    for key, value in metrics["test"].items():
        print(f"  {key}: {value if key == 'confusion_matrix' else round(value, 3)}")
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()