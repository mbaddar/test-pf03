# Week 03 — Machine Learning

> **Assumes Week 02:** an **offline-first feature table** workflow (raw immutable → processed idempotent, one row per unit of analysis, Parquet artifacts, fail-fast QA, report exports).
>
> **Week 03 goal:** turn that feature table into a **ship-ready baseline ML system**: define → split → baseline → train → evaluate → interpret → save → predict → report (locally on CPU, offline-first).

---

## Week outcomes

By the end of Week 03, you can:

- Turn a Week 02 **feature table** (`data/processed/*.parquet`) into an ML-ready dataset contract (X/y, schema, leakage checks, training/serving alignment).
- Choose a **split strategy** that matches real-world usage (holdout vs CV, stratified, group, time-based) and avoid leakage.
- Build **credible evaluation**: baselines first, cross-validation for selection, holdout for reporting, and a simple bootstrap CI for one key metric.
- Build a **mandatory scikit-learn baseline pipeline** (preprocess + model + CV) so you’re not dependent on AutoML tools.
- Use **PyCaret** to compare/tune/finalize models reproducibly (with artifacts you can hand off).
- Handle **class imbalance** practically: report PR metrics, choose thresholds, and apply class weights when appropriate.
- Do **error analysis** (slices + worst cases) and turn model failures into actionable data pipeline fixes (feedback into Week 02 ETL).
- Apply practical **interpretability** (feature importance + SHAP) and communicate limits responsibly.
- Package inference as a **batch prediction CLI** with schema guardrails and versioned artifacts.
- Capture **reproducibility essentials**: dataset hash, git commit, and environment/package versions.
- Produce a short **Model Card** + README commands + a minimal **monitoring plan** so a teammate can reproduce results offline and understand how it should be watched in production.

---

## Tool stack (opinionated, minimal)

**Goal:** small, stable stack that works on CPU laptops and supports offline-first repos.

- **pandas + pyarrow (Parquet)**
  - For loading `data/processed/features.parquet` and saving predictions/metrics tables.
  - Avoid CSV-only pipelines once schema stabilizes (dtype drift + slow + fragile).

- **scikit-learn**
  - For splits, metrics, baselines, and the “manual → PyCaret” mental bridge (pipelines + CV).
  - You will ship at least one scikit-learn baseline run artifact.

- **PyCaret (classification + regression)**
  - For consistent preprocessing + model comparison + tuning + saving pipelines.
  - Avoid “magic runs” without artifacts/metadata; treat `setup()` as a **contract**.

- **plotly (+ kaleido for export)** *(optional)*
  - For consistent reporting visuals (Week 02 style) when you need custom plots beyond PyCaret’s built-ins.
  - Avoid mixing multiple plotting libraries; prefer a single export pathway to `reports/figures/`.

- **typer**
  - For clean `train` / `predict` commands.
  - Avoid complex CLI frameworks; keep commands shallow and reproducible.

- **logging (stdlib) + json (stdlib)**
  - For run visibility + audit trails (metadata, parameters, dataset version, git commit, environment).
  - Avoid print-only training scripts.

**Not in core Week 03 stack:** deep learning frameworks, distributed compute, heavy experiment trackers (unless your capstone demands it later).

---

## Project conventions

### Repo layout (extends Week 02)

```text
project/
  pyproject.toml
  README.md

  data/
    raw/                 # immutable inputs
    cache/               # downloaded snapshots / API cache (safe to delete)
    processed/           # feature tables (idempotent): features.parquet
    external/            # small reference files (manual drops)
    samples/             # tiny in-repo snapshots for offline demos (optional)

  models/
    registry/            # "model registry" (lightweight)
      latest -> runs/<run_id>/      # optional symlink/shortcut (or a text pointer)
    runs/
      <run_id>/
        model/                     # saved PyCaret model pipeline
        metrics/
        plots/
        tables/
        schema/
        env/
        run_meta.json

  reports/
    figures/
    model_card.md
    eval_summary.md                # short narrative + caveats

  notebooks/
    03_ml_diagnostics.ipynb        # optional; reads artifacts, not raw

  src/
    project_name/
      __init__.py
      config.py
      ml/
        __init__.py
        train.py
        predict.py
        schema.py
        metrics.py
        baselines.py
        utils.py
```

### Naming + artifact rules (non-negotiable for “ship-ready”)

- **Run folder:** deterministic, timestamped, and unique.
  - Example: `models/runs/2026-01-04T21-35-10Z__clf__session42__v001/`

- **Artifacts produced every run:**
  - `model/` — saved model pipeline (PyCaret `save_model`)
  - `metrics/` — JSON + CSV (CV summary, holdout metrics, bootstrap CI, threshold metrics)
  - `plots/` — diagnostic images (confusion matrix / PR curve / residuals, etc.) *(best-effort; don’t fail the run if a plot isn’t supported)*
  - `tables/` — leaderboards (`compare_models`), holdout predictions, **holdout_input** (features-only table used to test the predict CLI)
  - `schema/` — input contract for inference (**required feature columns**, optional passthrough IDs, dtype policy, datetime policy)
  - `env/` — environment capture (`python_version`, `platform`, `pip_freeze.txt`)
  - `run_meta.json` — dataset version, git commit, parameters, timestamp, session_id, chosen threshold (if any)

- **Offline-first:** the training command reads **only** from `data/processed/*.parquet`.

### Workflow-first structure (repeat this every project)

> **Define → Split → Baseline → Compare → Tune → Validate → Interpret → Finalize → Ship artifacts**

Keep iteration tight; each run should add either:
- better data quality, or
- better evaluation/reporting, or
- better inference reliability.

---

# 1) Problem framing

**What it’s for:** turn “build a model” into a concrete spec you can evaluate and ship.

### Canonical operations / patterns

- Define the **unit of analysis** (one row = one prediction decision).
- Define the **target**:
  - classification: label meaning + **positive class** definition
  - regression: unit/scale (e.g., dollars), clip rules, log transform decision
- Define **success metric** and why it matches the product:
  - “What decision do we make with this prediction?”
  - “What’s the cost of false positives vs false negatives?”
- Define **constraints**:
  - CPU-only local training
  - inference latency (batch vs online)
  - feature availability at prediction time
- Define a **baseline** you can beat:
  - “predict majority class” / “predict mean”
  - or a simple heuristic used today
- Define “**good enough to ship in Week 03**”:
  - reproducible training + holdout report + saved pipeline + batch predictor + model card + minimal monitoring plan

**Capstone reminder (Week 5 deadline: Jan 15, 2026):**
- Pick a model direction that matches your capstone’s likely deployment constraints (interpretability, latency, update cadence).

### Pitfalls + checks

- “Metric chosen because it’s common” (instead of matching the decision).
- Target definition uses future information (target leakage).
- Building a model when a rule-based baseline already meets needs.
- Evaluating on data that doesn’t match real use (wrong split strategy).

### Minimal template: ML spec (paste into `reports/model_card.md` early)

```text
Problem:
- Predict: <target> for <unit of analysis>
- Decision enabled: <what action changes?>
- Constraints: CPU-only training; offline-first; batch inference

Data:
- Feature table: data/processed/features.parquet
- Unit: <one row per ...>
- Target column: <name>, positive class: <...> (if binary)

Splits:
- Holdout: <strategy>, rationale: <...>
- CV: <k/folds strategy>, leakage mitigation: <...>

Metrics:
- Primary: <metric>, reason: <ties to decision>
- Secondary: <metric>, reason: <...>
- Baseline: <dummy/heuristic>

Shipping:
- Artifacts: model pipeline + schema + metrics + plots + env + run metadata
- Known limitations: <segments/time ranges/data quality risks>
- Monitoring sketch: <what drift/perf signals you’d track>
```

---

# 2) Dataset readiness from a Week 02 feature table

**What it’s for:** ensure the table is safe to model and safe to use at inference.

### Canonical operations / patterns

- Start from `data/processed/features.parquet` (one row per unit).
- Identify:
  - `target` column
  - **ID columns** (never predictive features; may be passthrough fields at inference)
  - time columns (potential leakage + split strategy driver; often excluded as raw identifiers)
  - group columns (entities that must not cross folds, e.g., `user_id`)
- Policy for missingness (consistent and explicit):
  - prefer **missingness flags** for informative missingness
  - impute only with documented strategy (median/mode)
- Establish an **inference schema contract** (generated from training):
  - **required feature columns** (what the model consumes)
  - **optional ID passthrough columns** (kept if present; never required)
  - dtype normalization rules (numeric coercion, datetime parsing, categoricals to strings)
  - categorical “unknown” policy

### Pitfalls + checks

- **Leakage features**:
  - computed using future info (post-outcome)
  - direct proxies for the target (e.g., “refund_flag” for predicting refunds)
- IDs accidentally used as features (hash-like uniqueness → overfit).
- Duplicates:
  - same entity appears in both train and test due to incorrect row definition
- Target missingness:
  - mixing unlabeled rows into training without handling
- Datetime dtype drift:
  - timestamps turning into strings (causes silent feature breakage)

### Minimal checklist: ML readiness gates

- [ ] Exactly one row per unit of analysis (no duplicate keys).
- [ ] Target exists and is non-null for training rows.
- [ ] IDs/time fields are explicitly handled (drop from features or use split logic).
- [ ] No “future-known” features (document feature availability at prediction time).
- [ ] Missingness strategy defined (flags vs imputation).
- [ ] Inference schema contract can be generated from training data:
  - required feature columns
  - optional ID passthrough columns
  - dtype + datetime parsing policy
  - target is forbidden at inference input

### Minimal code template: load + basic QA gates

```python
from pathlib import Path
import pandas as pd

def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    assert len(df) > 0, "feature table is empty"
    return df

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def assert_no_duplicate_key(df: pd.DataFrame, key_cols: list[str]) -> None:
    dup = df.duplicated(subset=key_cols, keep=False)
    assert not dup.any(), f"Duplicate rows for key: {key_cols} (n={dup.sum()})"
```

---

# 3) Splitting strategies and leakage prevention

**What it’s for:** make evaluation match how the model will be used in production.

### Canonical operations / patterns

- Use **holdout + CV**:
  - CV: model selection / stability
  - Holdout: final reporting
- Choose split strategy based on the prediction situation:

**A) Random holdout (OK when i.i.d.)**
- Use when rows are independent-ish and time isn’t the primary axis.
- Use stratification for classification.

**B) Group-aware split (when leakage via entities)**
- Use when multiple rows per entity (user/customer/device/patient).
- Ensure entity does not appear in both train and test.

**C) Time-based split (when forecasting / drift likely)**
- Use when the model predicts future behavior or data changes over time.
- Holdout should be the most recent time window.
- Sorting by time is non-negotiable.

### Pitfalls + checks (put these here, not later)

- **Random split on time data** → “future leaks into train” → inflated metrics.
- Duplicates across splits (exact duplicates or near duplicates) → inflated metrics.
- Group leakage (same user in train and test) → inflated metrics.
- Target leakage in feature engineering (feature uses label or post-event fields).

### Minimal code templates (scikit-learn splits)

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, TimeSeriesSplit

def random_split(df: pd.DataFrame, *, target: str, test_size=0.2, seed=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def group_cv_splits(df: pd.DataFrame, *, group_col: str, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col]
    for train_idx, test_idx in gkf.split(df, groups=groups):
        yield train_idx, test_idx

def time_cv_splits(df: pd.DataFrame, *, n_splits=5):
    # Assumes df is already sorted by time ascending
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df):
        yield train_idx, test_idx
```

**Important note (group holdout):**
Group-aware CV (GroupKFold) does **not automatically** guarantee a group-safe holdout split. For group problems, do a **manual group holdout split** first, then apply GroupKFold only inside CV.

---

# 4) Evaluation is the product

**What it’s for:** choose metrics that drive correct decisions and avoid misleading results.

## 4.1 Metric choice (decision-first)

### Classification: choose based on what mistakes cost

- **Accuracy**
  - Use when classes are balanced and errors cost roughly equal.
  - Breaks on imbalanced data (can be “high” while model is useless).
- **F1**
  - Use when you care about balancing precision/recall and positives are relatively rare.
  - Sensitive to threshold; can hide poor calibration.
- **ROC-AUC**
  - Use for ranking quality when class imbalance isn’t extreme.
  - Can look good even when precision is terrible at real operating points.
- **PR-AUC (Average Precision)**
  - Use when positives are rare and you care about performance on the positive class.
  - More informative for highly imbalanced problems.

**Rule of thumb:**
If positive rate < ~10%, report **PR-AUC + precision/recall at a chosen operating threshold** (don’t rely on ROC-AUC alone).

**Multi-class note (keep it simple in Week 03):**
If your project is multi-class, report at least:
- accuracy
- macro or weighted F1  
…and defer multi-class AUC/PR to a later deep dive unless you’re confident in the setup.

### Regression: choose based on how errors hurt

- **MAE**
  - Use when outliers exist and you want stable “typical error”.
- **RMSE**
  - Use when large errors are disproportionately costly (penalizes big misses).
- **R²**
  - Use as a descriptive “variance explained” metric, not as the only ship metric.
  - Can mislead when target range is narrow or non-stationary.

### Pitfalls + checks

- Reporting only one metric.
- Comparing models on CV but “forgetting” holdout performance.
- Picking a threshold implicitly (0.5) without checking tradeoffs.
- Reporting metrics without sample sizes and class balance.

---

## 4.2 Baselines (always first)

**What it’s for:** prove you’re adding value before spending time tuning.

### Canonical baselines

- **Dummy baseline**
  - classification: majority class
  - regression: mean/median
- **Simple linear baseline**
  - logistic regression / ridge regression
- **Business heuristic baseline** (if applicable)
  - “flag if last_purchase_days > 90”

### Non-negotiable Week 03 rule
You must produce and save at least one **scikit-learn baseline pipeline** run (preprocessing + model + CV metrics), even if you later use PyCaret.

### Pitfalls + checks

- Skipping baselines → no idea if model is actually good.
- Comparing models without the same split strategy and preprocessing.

---

## 4.3 Uncertainty intuition (operational)

**What it’s for:** communicate variability without heavy theory.

- Use **CV variability** (mean ± std across folds) for stability.
- Use a simple **bootstrap CI on holdout** for one primary metric (optional but recommended).

### Minimal code template: bootstrap CI for a holdout metric

```python
from __future__ import annotations
import numpy as np

def bootstrap_ci(y_true, y_pred_or_proba, metric_fn, *, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_pred_or_proba)
    assert len(y_true) == len(y_hat) and len(y_true) > 0

    idx = np.arange(len(y_true))
    stats = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        stats.append(metric_fn(y_true[b], y_hat[b]))

    stats = np.asarray(stats, dtype=float)
    return {
        "estimate": float(metric_fn(y_true, y_hat)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }
```

---

## 4.4 Class imbalance (core)

**What it’s for:** make your evaluation and threshold choice reflect reality when positives are rare.

### Required reporting for imbalanced binary classification
- positive rate in train and holdout
- PR-AUC (Average Precision)
- precision/recall/F1 at your chosen operating threshold
- confusion matrix at that threshold

### Practical techniques (minimal)
- **Class weights** (often the first thing to try):
  - helps many linear models and tree models focus on the minority class
  - doesn’t change the data distribution; changes the loss
- **Resampling** (use carefully):
  - can help when the minority class is extremely small
  - can create leakage in time/group settings if done incorrectly  
  - if you resample, do it **inside CV folds only** (never before splitting)

### Pitfalls + checks
- Reporting only ROC-AUC on highly imbalanced tasks.
- Choosing threshold 0.5 by habit.
- Oversampling before splitting (leakage).
- “We improved AUC” but precision at usable recall is still unacceptable.

---

# 5) Minimum theory for tabular ML (just enough to avoid AutoML roulette)

**What it’s for:** build intuition for why models behave differently, without heavy math.

### 5.1 Model families (what tends to work when)

- **Linear / Logistic Regression**
  - Strong baseline, fast, stable, interpretable.
  - Needs sensible preprocessing (imputation + encoding; scaling helps for some solvers).
  - Regularization (L2/L1) prevents overfitting in high-dimensional one-hot features.

- **Tree-based models (Decision Tree, Random Forest, Gradient Boosting)**
  - Good at nonlinear relationships and feature interactions.
  - More tolerant of monotonic transformations; may handle missingness patterns better (varies by implementation).
  - Risk of overfitting if too deep / too many trees or if leakage exists.

- **Distance-based models (KNN) and margin-based models (SVM)**
  - Sensitive to feature scaling and high dimensionality.
  - Can be slow on large datasets; often not first choice for big tabular pipelines.

### 5.2 Bias–variance intuition (practical)
- If a model is too simple, it underfits → both CV and holdout are poor.
- If a model is too complex, it overfits → CV improves, holdout degrades.
- Leakage can mimic “amazing generalization” → suspiciously perfect metrics.

### 5.3 What you should take away
- Always baseline first.
- Prefer simpler models unless you have evidence they’re insufficient.
- Treat split strategy and leakage prevention as first-class engineering decisions.

---

# 6) PyCaret workflow (classification first, regression mirrored)

**What it’s for:** run reproducible experiments quickly while keeping evaluation honest and artifacts shippable.

## 6.1 Mental model: `setup()` is a contract, not a convenience

### Canonical operations / patterns
- Treat `setup()` as:
  - schema + preprocessing + split definition + CV configuration
- Everything you don’t specify becomes an implicit decision. Make key decisions explicit:
  - target, remove_columns, split strategy, session_id, fold strategy, stratification/grouping, feature handling

### Pitfalls + checks
- “It worked in the notebook” but you can’t reproduce because:
  - seed not fixed
  - split strategy changed
  - artifacts not saved
- Leakage via preprocessing done before `setup()` (don’t scale/encode globally yourself).

---

## 6.2 Minimal “manual → PyCaret” bridge (reduce magic)

**What it’s for:** understand what PyCaret is doing: a preprocessing + model pipeline evaluated with CV.

### Minimal scikit-learn baseline pipeline (classification)

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def sklearn_baseline_auc(df: pd.DataFrame, *, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())
```

### Equivalent PyCaret intent
- `setup()` defines preprocessing + CV
- `compare_models()` runs many models with that same pipeline discipline

---

## 6.3 Training flow (classification)

**What it’s for:** produce a best-effort baseline model with a clean evaluation story and saved artifacts.

### Canonical operations / patterns
- `setup()` with explicit:
  - `target=...`
  - `remove_columns=[id_cols...]`
  - `session_id=...`
  - split strategy (stratified/group/time as needed)
  - CPU-friendly settings
- `create_model('dummy')` baseline
- `compare_models()` shortlist
- `tune_model()` with a **budget**
- Evaluate on holdout via `predict_model()`
- **Choose threshold once** (if binary classification requires it) and record it
- `finalize_model()` only after you’ve captured holdout results
- `save_model()` + save tables/plots + write metadata + environment capture

### Pitfalls + checks (where teams get burned)
- Tuning repeatedly against the holdout (holdout becomes “train”).
- Finalizing too early (you lose a clean holdout evaluation story).
- Keeping ID columns (model memorizes rather than learns).
- Letting time leak (random split when you should time-split).
- Computing AUC/PR-AUC using the wrong probability column (must be **P(positive class)**).

---

### Minimal code template: classification training script (artifact-first)

> Template assumes you already have `data/processed/features.parquet`. Adjust column names.
>
> **Binary probability note:** when you compute ROC-AUC / PR-AUC, ensure you use **P(y=positive)** (not “confidence of predicted label”). Use `raw_score=True` and select the positive class probability column.

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json, logging, os, subprocess, hashlib, sys, platform

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

from pycaret.classification import (
    setup, create_model, compare_models, tune_model, finalize_model,
    predict_model, plot_model, pull, save_model
)

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainCfg:
    features_path: Path
    target: str

    # columns
    id_cols: tuple[str, ...] = ("id",)          # optional passthrough identifiers
    time_col: str | None = None                # if set, sort by time before splitting
    group_col: str | None = None               # if set, use group-safe splitting

    # binary classification specifics
    pos_label: str | int = 1                   # define your positive class
    threshold_strategy: str = "fixed"          # "fixed" | "min_precision" | "max_f1"
    threshold_value: float = 0.5
    min_precision: float = 0.8

    # splitting / reproducibility
    session_id: int = 42
    train_size: float = 0.8
    fold: int = 5

    # model selection
    sort_metric: str = "AUC"     # PyCaret metric name
    tune_metric: str = "AUC"
    tune_iters: int = 25

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _pip_freeze() -> str:
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"# pip freeze failed: {e!r}\n"

def _score_col_for_pos_label(df_pred: pd.DataFrame, pos_label: str | int) -> str | None:
    # When using raw_score=True, PyCaret commonly returns columns like "Score_0", "Score_1"
    key = f"Score_{pos_label}"
    if key in df_pred.columns:
        return key

    # Fallback: if exactly two Score_ columns exist, warn and choose the second by sort order
    score_cols = sorted([c for c in df_pred.columns if c.startswith("Score_")])
    if len(score_cols) == 2:
        log.warning("Could not find Score_%s; using %s. Verify this is P(positive).", pos_label, score_cols[1])
        return score_cols[1]

    # Final fallback: prediction_score sometimes exists; it may be score of predicted label (NOT always positive proba)
    if "prediction_score" in df_pred.columns:
        log.warning("Using prediction_score; verify it is P(positive) before trusting AUC/PR-AUC.")
        return "prediction_score"
    if "Score" in df_pred.columns:
        log.warning("Using Score; verify it is P(positive) before trusting AUC/PR-AUC.")
        return "Score"
    return None

def _choose_threshold(y_true, y_score, *, strategy: str, fixed: float, min_precision: float) -> float:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if strategy == "fixed":
        return float(fixed)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall are len(thresholds)+1
    thresholds = np.asarray(thresholds, dtype=float)

    if strategy == "min_precision":
        ok = np.where(precision[:-1] >= min_precision)[0]
        if len(ok) == 0:
            log.warning("No threshold achieves min_precision=%.3f on holdout; falling back to %.2f", min_precision, fixed)
            return float(fixed)
        best = ok[np.argmax(recall[ok])]
        return float(thresholds[best])

    if strategy == "max_f1":
        # compute F1 for each threshold
        f1s = []
        for i, t in enumerate(thresholds):
            y_pred = (y_score >= t).astype(int)
            # if labels aren’t {0,1}, user should map them earlier; keep Week 03 simple
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        best = int(np.argmax(f1s))
        return float(thresholds[best])

    raise ValueError(f"Unknown threshold strategy: {strategy}")

def _holdout_metrics(df_pred: pd.DataFrame, *, target: str, pos_label: str | int, threshold: float) -> dict:
    # Prediction label column name differs by version
    pred_col = "prediction_label" if "prediction_label" in df_pred.columns else "Label"
    y_true = df_pred[target]

    # If binary labels aren’t {0,1}, thresholding assumes y_score aligns with pos_label; keep strict in Week 03
    y_pred = df_pred[pred_col]

    out = {
        "n_holdout": int(len(df_pred)),
        "positive_rate_holdout": float((y_true == pos_label).mean()) if len(pd.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Binary metrics at decision threshold (if we have a positive-class probability)
    score_col = _score_col_for_pos_label(df_pred, pos_label)
    if score_col is not None and len(pd.unique(y_true)) == 2:
        y_score = df_pred[score_col]
        y_pred_t = (y_score >= threshold).astype(int)
        # assumes pos_label maps to 1; if not, map labels earlier in data prep
        out.update({
            "threshold": float(threshold),
            "precision": float(precision_score((y_true == pos_label).astype(int), y_pred_t, zero_division=0)),
            "recall": float(recall_score((y_true == pos_label).astype(int), y_pred_t, zero_division=0)),
            "f1": float(f1_score((y_true == pos_label).astype(int), y_pred_t, zero_division=0)),
            "roc_auc": float(roc_auc_score((y_true == pos_label).astype(int), y_score)),
            "pr_auc": float(average_precision_score((y_true == pos_label).astype(int), y_score)),
        })
    else:
        # Multi-class fallback: report macro/weighted F1 if desired (kept minimal here)
        if len(pd.unique(y_true)) > 2:
            out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return out

def run_train(cfg: TrainCfg, *, root: Path, run_tag: str = "clf") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = f"{ts}__{run_tag}__session{cfg.session_id}"
    run_dir = root / "models" / "runs" / run_id

    for d in ["metrics", "plots", "tables", "schema", "env", "model"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("Run dir: %s", run_dir)

    df = pd.read_parquet(cfg.features_path)
    assert cfg.target in df.columns, f"Missing target: {cfg.target}"
    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)

    # Optional: enforce time sorting for time-based evaluation
    if cfg.time_col:
        assert cfg.time_col in df.columns, f"Missing time_col: {cfg.time_col}"
        df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # Build schema contract: required features vs optional IDs
    feature_cols = [c for c in df.columns if c not in {cfg.target, *cfg.id_cols}]
    schema = {
        "target": cfg.target,
        "required_feature_columns": feature_cols,
        "optional_id_columns": [c for c in cfg.id_cols if c in df.columns],
        "feature_dtypes": {c: str(df[c].dtype) for c in feature_cols},
        "datetime_columns": [c for c in feature_cols if "datetime" in str(df[c].dtype).lower()],
        "policy_unknown_categories": "tolerant (OneHotEncoder handle_unknown=ignore)",
        "forbidden_columns": [cfg.target],
    }
    (run_dir / "schema" / "input_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Environment capture
    (run_dir / "env" / "pip_freeze.txt").write_text(_pip_freeze(), encoding="utf-8")
    env_meta = {
        "python_version": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
    }
    (run_dir / "env" / "env_meta.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

    # Keep PyCaret outputs inside run_dir
    cwd = Path.cwd()
    os.chdir(run_dir)

    try:
        exp = setup(
            data=df,
            target=cfg.target,
            remove_columns=[c for c in cfg.id_cols if c in df.columns],
            session_id=cfg.session_id,
            train_size=cfg.train_size,
            fold=cfg.fold,
            data_split_stratify=True,
            n_jobs=-1,
            html=False,
            verbose=False,
        )

        # Baseline
        _ = create_model("dummy")
        pull().to_csv(run_dir / "tables" / "baseline_dummy_cv.csv", index=False)

        # Compare
        best = compare_models(sort=cfg.sort_metric, n_select=1, turbo=True)
        pull().to_csv(run_dir / "tables" / "compare_models.csv", index=False)

        # Tune (budgeted)
        tuned = tune_model(best, optimize=cfg.tune_metric, n_iter=cfg.tune_iters, choose_better=True)
        pull().to_csv(run_dir / "tables" / "tune_results.csv", index=False)

        # Holdout evaluation (BEFORE finalize)
        holdout_pred = predict_model(tuned, raw_score=True)  # try to include per-class probabilities
        holdout_pred.to_parquet(run_dir / "tables" / "holdout_predictions.parquet", index=False)

        # Save a holdout *input* file for predict CLI testing (features + optional IDs; no target)
        holdout_input_cols = schema["required_feature_columns"] + schema["optional_id_columns"]
        holdout_input = holdout_pred[holdout_input_cols].copy()
        holdout_input.to_parquet(run_dir / "tables" / "holdout_input.parquet", index=False)

        # Choose threshold once (binary only; do not iterate on holdout)
        score_col = _score_col_for_pos_label(holdout_pred, cfg.pos_label)
        threshold = cfg.threshold_value
        if score_col is not None and holdout_pred[cfg.target].nunique() == 2:
            y_true_bin = (holdout_pred[cfg.target] == cfg.pos_label).astype(int)
            y_score = holdout_pred[score_col]
            threshold = _choose_threshold(
                y_true_bin, y_score,
                strategy=cfg.threshold_strategy,
                fixed=cfg.threshold_value,
                min_precision=cfg.min_precision,
            )

        metrics = _holdout_metrics(holdout_pred, target=cfg.target, pos_label=cfg.pos_label, threshold=threshold)
        (run_dir / "metrics" / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # Diagnostics (best-effort; don’t fail run if plot not supported)
        plots_dir = run_dir / "plots"
        os.chdir(plots_dir)
        for p in ["confusion_matrix", "auc", "pr", "calibration"]:
            try:
                plot_model(tuned, plot=p, save=True)
            except Exception as e:
                log.warning("Plot failed (%s): %s", p, e)
        os.chdir(run_dir)

        # Finalize for shipping (trained on full dataset)
        final = finalize_model(tuned)
        save_model(final, str((run_dir / "model" / "pycaret_model").with_suffix("")))

        # Run metadata
        meta = {
            "run_id": run_id,
            "timestamp_utc": ts,
            "task": "classification",
            "session_id": cfg.session_id,
            "features_path": str(cfg.features_path),
            "features_sha256": _sha256(cfg.features_path),
            "git_commit": _git_commit(),
            "cfg": {k: str(v) for k, v in asdict(cfg).items()},
            "holdout_metrics": metrics,
            "threshold": metrics.get("threshold"),
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log.info("Done. Model saved under %s", run_dir / "model")
        return run_dir

    finally:
        os.chdir(cwd)
```

---

## 6.4 Regression workflow (mirror changes only)

**What it’s for:** same pipeline, different module + metrics.

### Canonical operations / patterns
- Swap imports:
  - `from pycaret.regression import setup, compare_models, ...`
- Change metrics:
  - primary: MAE or RMSE
  - diagnostics: residual plots, error distribution
- Stratification doesn’t apply; focus on time/group splits when relevant.

### Pitfalls + checks
- Using RMSE when outliers are data errors (RMSE becomes “outlier detector”).
- Ignoring target scale (e.g., predicting dollars: MAE of 50 might be great or awful depending on baseline).

### Minimal changes (conceptual)
- `sort_metric`: `"MAE"` or `"RMSE"`
- Holdout metrics: `mae`, `rmse`, `r2`
- Plots: `plot_model(model, plot="residuals")`, `plot="error"`, `plot="prediction_error"`

---

# 7) Classification diagnostics: thresholds, tradeoffs, calibration

**What it’s for:** make the model usable for a real decision (not just “AUC is good”).

### Canonical operations / patterns
- Always inspect:
  - confusion matrix at a chosen threshold
  - PR curve (especially when positives are rare)
  - calibration (if probabilities drive decisions)
- Choose threshold based on the actual requirement:
  - maximize F1 (balanced)
  - maximize recall at a minimum precision
  - minimize expected cost (cost matrix)

### Pitfalls + checks
- Default threshold (0.5) treated as “correct”.
- ROC-AUC reported without PR-AUC on imbalanced data.
- Threshold tuned on holdout repeatedly (holdout becomes training signal).
- Probabilities used as “confidence” without calibration check.

### Minimal code template: pick a threshold from holdout scores

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def threshold_for_min_precision(y_true, y_score, *, min_precision=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall arrays are 1 longer than thresholds
    ok = np.where(precision[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return None
    # choose threshold with best recall subject to precision constraint
    best = ok[np.argmax(recall[ok])]
    return float(thresholds[best])
```

**Week 03 requirement:** store the chosen threshold (and its rationale) in:
- `models/runs/<run_id>/run_meta.json`
- `reports/model_card.md`

---

# 8) Regression diagnostics: residuals and error slices

**What it’s for:** find where the model is wrong and why (and whether it’s fixable with data).

### Canonical operations / patterns
- Report:
  - MAE/RMSE + baseline MAE/RMSE
  - error distribution (median error, p90 absolute error)
- Inspect:
  - residual vs prediction (heteroscedasticity)
  - worst errors
  - error slices by segments (category/time buckets)

### Pitfalls + checks
- R² only (can look “fine” while absolute error is unacceptable).
- Model dominated by a few large outliers (fix data, clip, or change metric).
- Leakage via target-derived features (residuals suspiciously tiny).

### Minimal code template: error slices

```python
import pandas as pd
import numpy as np

def error_slices(df: pd.DataFrame, *, y_true_col: str, y_pred_col: str, slice_col: str) -> pd.DataFrame:
    out = df.copy()
    out["abs_err"] = (out[y_true_col] - out[y_pred_col]).abs()
    return (
        out.groupby(slice_col, dropna=False)
           .agg(n=("abs_err", "size"), mae=("abs_err", "mean"), p90=("abs_err", lambda s: np.quantile(s, 0.9)))
           .reset_index()
           .sort_values("mae", ascending=False)
    )
```

---

# 9) Error analysis and model debugging loop

**What it’s for:** systematically improve by fixing the right thing (data vs model vs evaluation).

### Canonical operations / patterns
- Start with **holdout predictions table**:
  - sort worst predictions
  - inspect raw features for those rows
- Slice by:
  - category segments (region/product/tier)
  - time windows (month/week)
  - data quality indicators (missingness flags)
- Feed findings back to Week 02 pipeline:
  - fix joins, missingness, label definition, deduping, feature availability

### Pitfalls + checks (common failure modes)
- **Leakage symptoms:**
  - near-perfect metrics
  - feature importance dominated by obviously post-event features
  - huge train/CV vs holdout gap
- **Overfitting symptoms:**
  - CV improves with tuning but holdout degrades
  - overly complex models win leaderboard but fail in slices
- **Data quality symptoms:**
  - worst errors correlate with missingness / category typos
  - model fails on rare categories (need better category policy)
- **Label noise symptoms:**
  - cases where features strongly suggest the opposite label; check labeling rules

---

# 10) Interpretability (practical): feature importance + SHAP

**What it’s for:** answer “what signals drive predictions?” and “does this make sense?” (not “prove causality”).

### Canonical operations / patterns
- Start with:
  - global feature importance (quick sanity check)
  - partial dependence or SHAP summary for deeper understanding (optional)
- Use interpretability for:
  - debugging leakage
  - validating that model uses plausible signals
  - communicating limitations

### Pitfalls + checks
- **Correlation ≠ causation**: importance explains model behavior, not real-world causal effects.
- Leakage features often look “important” because they encode the label.
- SHAP can be slow; run it on a sample and only for the final candidate model.
- Don’t over-interpret tiny differences in importance.

### Minimal PyCaret patterns (conceptual)
- `plot_model(model, plot="feature")` — global importance (where supported)
- `plot_model(model, plot="shap")` / `interpret_model(model)` — SHAP workflows (if available in your PyCaret install)

---

# 11) Optional track: unsupervised learning (clustering for segmentation)

**What it’s for:** segment entities (customers/items/transactions) when you don’t have labels.

> **Optional:** If time is tight, keep clustering as a bonus lab. Week 03 core shipping competency is supervised ML + reliable evaluation + inference packaging.

### Canonical operations / patterns
- Use clustering for:
  - segmentation for strategy (“cluster personas”)
  - anomaly surfacing (small weird clusters)
  - feature engineering (cluster_id as a feature downstream)
- Preprocess is everything:
  - scale numeric features
  - handle categoricals carefully (one-hot can explode)
  - remove IDs and time identifiers
- Evaluate with:
  - silhouette score (rough guidance)
  - cluster sizes (avoid tiny nonsense clusters)
  - qualitative sanity checks (do clusters mean anything?)

### Pitfalls + checks
- Clustering is sensitive to scaling; unscaled features dominate.
- High-cardinality categories → huge sparse vectors → unstable clusters.
- Clusters are not ground truth; don’t claim “accuracy”.

### Minimal code template: KMeans clustering pipeline

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def fit_kmeans(df: pd.DataFrame, *, k: int = 5):
    num = df.select_dtypes(include=["number"]).columns
    cat = df.select_dtypes(exclude=["number"]).columns

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ]
    )

    model = Pipeline([
        ("pre", pre),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42)),
    ])

    model.fit(df)
    return model
```

---

# 12) Packaging for inference (batch prediction CLI + schema guardrails)

**What it’s for:** make predictions reliably on new data without notebooks, with input validation and clear outputs.

### Canonical operations / patterns
- Inference is a **separate entrypoint** (`src/.../ml/predict.py`):
  - load saved model pipeline
  - load input data (CSV/Parquet)
  - validate schema (required feature columns + dtype normalization + target forbidden)
  - run `predict_model()` and write outputs
- Outputs are explicit:
  - include `prediction`, and `score`/`probability` when applicable
  - include stable identifier columns if provided (so you can join back)

### Pitfalls + checks
- Input data missing columns → crash or silent wrong predictions.
- Dtype drift (numbers as strings) → NaNs after coercion.
- Datetimes drift into strings → broken features.
- Unknown categories → either crash (strict encoders) or silent all-zeros (handle unknown policy).
- Training-time columns accidentally included at inference (target leakage via pipeline misuse).

### Minimal code template: schema validation (tolerant, with IDs + datetimes)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd

@dataclass(frozen=True)
class InputSchema:
    required_feature_columns: list[str]
    feature_dtypes: dict[str, str]
    optional_id_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    forbidden_columns: list[str] = field(default_factory=list)
    policy_unknown_categories: str = "tolerant"

    @staticmethod
    def load(path: Path) -> "InputSchema":
        d = json.loads(path.read_text(encoding="utf-8"))
        return InputSchema(
            required_feature_columns=d["required_feature_columns"],
            feature_dtypes=d["feature_dtypes"],
            optional_id_columns=d.get("optional_id_columns", []),
            datetime_columns=d.get("datetime_columns", []),
            forbidden_columns=d.get("forbidden_columns", []),
            policy_unknown_categories=d.get("policy_unknown_categories", "tolerant"),
        )

def validate_and_align(df: pd.DataFrame, schema: InputSchema) -> tuple[pd.DataFrame, pd.DataFrame]:
    # fail-fast on forbidden columns (e.g., target)
    forbidden_present = [c for c in schema.forbidden_columns if c in df.columns]
    assert not forbidden_present, f"Forbidden columns present in inference input: {forbidden_present}"

    missing = [c for c in schema.required_feature_columns if c not in df.columns]
    assert not missing, f"Missing required feature columns: {missing}"

    out = df.copy()

    # capture optional IDs if present (passthrough)
    id_cols_present = [c for c in schema.optional_id_columns if c in out.columns]
    passthrough = out[id_cols_present].copy() if id_cols_present else pd.DataFrame(index=out.index)

    # dtype normalization
    for c, dt in schema.feature_dtypes.items():
        if c not in out.columns:
            continue
        if c in schema.datetime_columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
        elif "int" in dt.lower() or "float" in dt.lower():
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].astype("string")

    X = out[schema.required_feature_columns].copy()
    return X, passthrough
```

### Minimal code template: batch prediction script (PyCaret)

```python
from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
from pycaret.classification import load_model, predict_model  # swap module for regression

from project_name.ml.schema import InputSchema, validate_and_align

log = logging.getLogger(__name__)

def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def write_tabular(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def run_predict(*, model_path_stem: Path, schema_path: Path, input_path: Path, output_path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    schema = InputSchema.load(schema_path)
    df = read_tabular(input_path)
    X, ids = validate_and_align(df, schema)

    model = load_model(str(model_path_stem))  # PyCaret expects stem without .pkl
    pred = predict_model(model, data=X, raw_score=True)

    # keep IDs for join-back if present
    if len(ids.columns) > 0:
        pred = pd.concat([ids.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

    write_tabular(pred, output_path)
    log.info("Wrote predictions: %s (%s rows)", output_path, len(pred))
```

### Required “training–serving skew” check
After a training run:
1. Use the saved `tables/holdout_input.parquet` (features-only).
2. Run your predict CLI against it.
3. Confirm outputs match your training run expectations (and optionally re-compute holdout metrics from CLI outputs).

This catches schema drift, dtype coercion problems, and accidental leakage.

---

# 13) Run metadata + lightweight “model registry”

**What it’s for:** reproducibility, auditability, and clean handoffs (especially in teams).

### Canonical operations / patterns
- Every training run writes `run_meta.json` with:
  - timestamp UTC
  - dataset path + hash
  - git commit
  - config (target, split params, metric, threshold)
  - key results (holdout metrics)
- Also capture environment versions:
  - python version, platform, `pip freeze`
- Keep a “registry pointer”:
  - `models/registry/latest.txt` containing the run_id
  - or a symlink `models/registry/latest -> models/runs/<run_id>/` (platform-dependent)

### Pitfalls + checks
- “Which model is deployed?” becomes unanswerable.
- Team members can’t reproduce results if dataset version or commit is missing.
- Metrics exist only in notebooks (not saved artifacts).
- Different environments produce inconsistent results (no env capture).

### Minimal template: `latest.txt` pointer

```text
# models/registry/latest.txt
2026-01-04T21-35-10Z__clf__session42
```

---

# 14) Reporting & handoff (Model Card + README + monitoring sketch)

**What it’s for:** make your work usable by someone else (or future you) without re-learning everything.

### Canonical operations / patterns
- `reports/model_card.md` includes:
  - problem definition + decision
  - dataset + feature table version + environment note
  - split strategy (and why)
  - metrics: baseline vs selected model; CV vs holdout
  - threshold choice (if classification)
  - limitations + known failure slices
  - **monitoring plan sketch** (what to watch, retraining trigger)
  - how to run train/predict
- `README.md` includes:
  - one command to train
  - one command to predict
  - what artifacts are produced + where

### Pitfalls + checks
- Reporting only CV leaderboard (not a real product claim).
- No mention of leakage risks or data limitations.
- No reproduction steps.
- No monitoring plan (model performance will silently degrade).

### Minimal Model Card template (copy/paste)

```markdown
# Model Card — <project/model name>

## Problem
- Target: <col> (definition)
- Unit of analysis: <one row = ...>
- Decision: <what action uses predictions?>

## Data
- Feature table: data/processed/features.parquet
- Snapshot/version: <hash or date>
- Key exclusions:
  - IDs dropped from features: <...>
  - Leakage columns removed: <...>

## Splits
- Holdout: <random/group/time>, rationale: <...>
- CV: <k-fold / groupkfold / timeseries>, folds: <k>
- Leakage mitigations: <time sorting, group constraints, etc.>

## Metrics
- Class balance (if classification): train pos rate = <...>, holdout pos rate = <...>
- Baseline (dummy + sklearn linear): <metrics>
- Selected model: CV mean±std: <...>
- Holdout: <primary metric + CI>, plus <secondary metrics>
- Threshold (if classification): <value + rationale + precision/recall at threshold>

## Diagnostics & Interpretability
- Key plots: <confusion matrix/PR curve/calibration/residuals>
- Top features: <summary>
- SHAP (optional): <summary>

## Limitations
- Known weak slices: <...>
- Data quality risks: <missingness/label noise/drift>
- Not causal; correlations only.

## Monitoring sketch (minimum)
- Input drift checks: <missingness, category rates, numeric distribution shift>
- Performance monitoring: <metric>, label availability: <how/when labels arrive>
- Retraining trigger: <time cadence or drift/perf thresholds>
- High-risk slices to watch: <...>

## Reproducibility
- Git commit: <...>
- Dataset hash: <...>
- Environment: see models/runs/<run_id>/env/pip_freeze.txt

## How to run
- Train: `python -m project_name.ml.train`
- Predict: `python -m project_name.ml.predict --input data/new.parquet --output outputs/preds.parquet`
```

---

# 15) Offline-first robustness + Colab fallback (when PyCaret install is heavy)

**What it’s for:** keep progress moving even if local installs fail or compute is slow.

### Canonical operations / patterns
- **Local CPU baseline always works:** scikit-learn baseline pipeline + metrics.
- **PyCaret fallback:** run PyCaret training in Colab, then copy artifacts back to repo.

### Minimal Colab fallback pattern
- In Colab:
  - upload or mount your repo (Drive or zip)
  - install deps:
    - `pip install pycaret`
  - run your training module on the same `data/processed/features.parquet`
- Copy back into your repo:
  - the entire `models/runs/<run_id>/` folder
  - `reports/model_card.md` updates
  - any generated figures/tables
- Commit artifacts + metadata to GitHub (small commits; keep run folders tidy).

### Pitfalls + checks
- Training on a different dataset copy than the repo’s feature table.
- Losing run metadata when copying artifacts back.
- Colab-only results not reproducible locally (ensure scripts are the source of truth).

---

## Techniques appendix / index (mini internal wiki for later notebooks)

Create these as small, focused notebooks later (optional deep dives):

- **01_leakage_case_studies.ipynb**
  - classic leakage patterns (time, joins, post-event features) + how to detect
- **02_time_splits_and_backtesting.ipynb**
  - rolling windows, drift intuition, time-based evaluation narratives
- **03_threshold_tuning.ipynb**
  - precision/recall constraints, cost-based thresholds, reporting operating points
- **04_calibration_recipes.ipynb**
  - reliability curves + when to calibrate probabilities
- **05_pr_curves_for_imbalance.ipynb**
  - PR-AUC vs ROC-AUC on rare positives, selecting decision thresholds
- **06_bootstrap_ci_for_metrics.ipynb**
  - AUC/PR-AUC/MAE bootstrap CIs on holdout
- **07_shap_recipes.ipynb**
  - fast SHAP on samples, interpreting summary plots, common misreads
- **08_imbalance_handling.ipynb**
  - class weights, resampling, when it helps vs when it breaks
- **09_group_cv_patterns.ipynb**
  - entity leakage, GroupKFold vs stratification tradeoffs
- **10_model_monitoring_sketch.ipynb**
  - drift signals, data contracts, basic monitoring plan
- **11_fairness_checks_basics.ipynb**
  - slice metrics across sensitive/proxy groups, reporting responsibly

---

# Deliverables (definition of “done”)

By end of Week 03, your repo should contain:

## 1) Reproducible training/evaluation command
- [ ] A module or CLI command that:
  - reads `data/processed/features.parquet`
  - trains a supervised model (PyCaret)
  - writes artifacts to `models/runs/<run_id>/...`
- [ ] Includes:
  - split strategy clearly encoded (holdout + CV)
  - baselines saved (PyCaret dummy + **scikit-learn baseline**)
  - CV leaderboard + tune results saved (`pull()` tables)
  - holdout predictions table saved
  - **holdout_input.parquet** saved (features-only) for predict-CLI skew checks
  - diagnostic plots saved to disk (best-effort)
  - run metadata (`run_meta.json`) with dataset hash + git commit + params + threshold (if any)
  - environment capture saved (`env/pip_freeze.txt` + python/platform info)

## 2) Saved model pipeline artifact(s)
- [ ] Saved PyCaret model pipeline under `models/runs/<run_id>/model/`
- [ ] Input schema contract under `models/runs/<run_id>/schema/input_schema.json`:
  - required feature columns
  - optional passthrough IDs
  - dtype policy + datetime parsing policy
  - forbidden columns (target)

## 3) Batch inference script/CLI
- [ ] A `predict` script that:
  - loads the saved model
  - validates required feature columns
  - fails fast if target is present
  - coerces numeric + datetime dtypes
  - writes predictions to CSV/Parquet
  - preserves optional IDs if present
- [ ] Uses the same schema contract that training produced.
- [ ] Demonstrates the **training–serving skew check** by predicting on `holdout_input.parquet`.

## 4) Reporting & handoff
- [ ] `reports/model_card.md` with:
  - problem, data, split, metrics, threshold decision, limitations, monitoring sketch, how to run
- [ ] README updated with:
  - train/evaluate command
  - predict command
  - artifact locations and meanings

## 5) Optional (nice-to-have) app demo
- [ ] A tiny Streamlit “predictor” UI that calls the **same inference code** used by batch prediction (no duplicated logic).
