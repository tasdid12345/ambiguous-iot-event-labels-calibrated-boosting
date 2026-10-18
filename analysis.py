"""Compare a standard random forest (baseline) with a class-balanced, calibrated
LightGBM (improved) on OpenML dataset 1478 (smartphone HAR, used as a proxy for
ambiguous IoT sensor events). Main metric: macro-F1 on a fixed held-out split."""
import json
import sys
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import openml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
SEEDS = [42, 43, 44]          # both models are randomized -> run every seed
DATA_ID = 1478
MAX_ROWS = 200_000            # subsample guard (dataset has ~10k rows)
TEST_SIZE = 0.2
SOURCE_URL = "https://www.openml.org/d/1478"


def fetch_with_retry(fn, tries=5):
    """Run a download function, retrying with exponential backoff on failure
    (covers HTTP 429/5xx and network errors raised by the openml client)."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # openml raises its own exception types
            if i == tries - 1:
                raise
            wait = 2 ** i
            print(f"  download failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)


def load_data():
    """Download OpenML 1478 and return numeric features and string labels."""
    print(f"Downloading OpenML dataset {DATA_ID} via openml ...")
    ds = fetch_with_retry(lambda: openml.datasets.get_dataset(DATA_ID))
    df = fetch_with_retry(lambda: ds.get_data(dataset_format="dataframe")[0])
    target = ds.default_target_attribute
    X = df.drop(columns=[target]).astype(float)
    y = df[target].astype(str)
    if len(X) > MAX_ROWS:  # subsample very large data
        idx = X.sample(n=MAX_ROWS, random_state=SEED).index
        X, y = X.loc[idx], y.loc[idx]
    assert np.isfinite(X.to_numpy()).all(), "non-finite feature values"
    print(f"Loaded {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes")
    return X.to_numpy(), y.to_numpy()


def evaluate(proba, classes, y_true):
    """Shared evaluation: macro-F1 (main), top-1 accuracy, top-2 coverage."""
    top1 = classes[proba.argmax(axis=1)]
    top2 = classes[np.argsort(-proba, axis=1)[:, :2]]
    return {
        "f1": f1_score(y_true, top1, average="macro"),
        "acc": accuracy_score(y_true, top1),
        "top2": float((top2 == y_true[:, None]).any(axis=1).mean()),
    }


def tune_lgbm(X, y):
    """Small random search (6 configs, 3-fold stratified CV) on the TRAINING set only."""
    grid = {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31], "min_child_samples": [10, 30]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_p, best_s = None, -np.inf
    for p in ParameterSampler(grid, n_iter=6, random_state=SEED):
        m = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1, **p)
        s = cross_val_score(m, X, y, cv=cv, scoring="f1_macro", n_jobs=1).mean()
        print(f"  CV macro-F1 {s:.4f} for {p}")
        if s > best_s:
            best_p, best_s = p, s
    print(f"Selected LightGBM params {best_p} (CV macro-F1 {best_s:.4f})")
    return best_p, best_s


def main():
    X, y = load_data()
    y = LabelEncoder().fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Split: {len(X_tr)} train / {len(X_te)} test (stratified, seed {SEED})")

    params, cv_f1 = tune_lgbm(X_tr, y_tr)  # tuning uses training CV only

    scores = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"Seed {s}: fitting baseline random forest")
        rf = RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1).fit(X_tr, y_tr)
        scores["baseline"].append(evaluate(rf.predict_proba(X_te), rf.classes_, y_te))

        print(f"Seed {s}: fitting class-balanced, sigmoid-calibrated LightGBM")
        base = lgb.LGBMClassifier(class_weight="balanced", random_state=s, verbose=-1, n_jobs=-1, **params)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_tr, y_tr)
        scores["improved"].append(evaluate(cal.predict_proba(X_te), cal.classes_, y_te))

    def agg(name, key):
        v = np.array([d[key] for d in scores[name]], dtype=float)
        return float(v.mean()), float(v.std(ddof=1))

    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    b_acc, _ = agg("baseline", "acc")
    i_acc, _ = agg("improved", "acc")
    b_t2, _ = agg("baseline", "top2")
    i_t2, _ = agg("improved", "top2")
    metrics = {
        "baseline_score": b_f1, "baseline_score_std": b_f1_sd,
        "improved_score": i_f1, "improved_score_std": i_f1_sd,
        "improvement": i_f1 - b_f1,
        "baseline_f1": b_f1, "improved_f1": i_f1,
        "baseline_top1_accuracy": b_acc, "improved_top1_accuracy": i_acc,
        "baseline_top2_coverage": b_t2, "improved_top2_coverage": i_t2,
        "improved_cv_macro_f1": float(cv_f1),
    }
    if not all(np.isfinite(list(metrics.values()))):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    # Optional chart: baseline vs improved macro-F1 (mean +/- std over seeds)
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline RF", "Balanced+calibrated LGBM"], [b_f1, i_f1],
           yerr=[b_f1_sd, i_f1_sd], capsize=6, color=["grey", "steelblue"])
    ax.set_ylabel("Test macro-F1 (mean over seeds)")
    ax.set_title("Macro-F1 on held-out split")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a class-balanced, calibrated LightGBM classifier beat a standard random forest at resolving ambiguous smartphone sensor events into activity labels, measured by macro-F1 on the same held-out split?",
        "method": ("Stratified 80/20 split (seed 42) of OpenML 1478 smartphone HAR data. Both models use the same split, "
                   "features, evaluation function and seeds 42-44. LightGBM hyperparameters are chosen by 3-fold CV on the "
                   "training set only; the test set is used once per seed for the final evaluation."),
        "dataset": {
            "name": "OpenML 1478 Human Activity Recognition with Smartphones",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test set)",
        "higher_is_better": True,
        "baseline": "Standard RandomForestClassifier (500 trees); top-1 predicted label is the event interpretation.",
        "improved": ("Class-balanced LightGBM with small CV hyperparameter search, wrapped in sigmoid "
                     "calibration (3-fold); also yields a top-2 candidate set per event."),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "notes": (f"Proxy experiment: HAR sensor windows stand in for ambiguous IoT events; the paper's own data and "
                  f"model are not used. Ambiguity is not labelled in the data, so top-2 coverage is a diagnostic. "
                  f"Selected LightGBM params: {params}. Baseline top-2 coverage is derived from RF probabilities and is "
                  f"reported only for comparison. Std is the sample std over 3 seeds. The openml client controls its own "
                  f"HTTP timeouts; retries wrap the download calls."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline macro-F1 {b_f1:.4f} +/- {b_f1_sd:.4f} | Improved {i_f1:.4f} +/- {i_f1_sd:.4f} "
          f"| improvement {i_f1 - b_f1:+.4f}")
    print("Wrote results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)
