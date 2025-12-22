---
pagetitle: "W2 D2"
title: "Data Work (ETL + EDA)"
subtitle: "AI Professionals Bootcamp | Week 2"
date: 2025-12-22
---

## Policy: GenAI usage

- ✅ Allowed: **clarifying questions** (definitions, error explanations)
- ❌ Not allowed: generating code, writing solutions, or debugging by copy-paste
- If unsure: ask the instructor first

::: callout-tip
**In this course:** you build skill by typing, running, breaking, and fixing.
:::

# Day 2: Data Quality + Cleaning Basics

**Goal:** make your dataset **trustworthy** by adding checks + cleaning for **missingness**, **duplicates**, and **text normalization**. 

::: {.muted}
Bootcamp • SDAIA Academy
:::

::: {.notes}
Say: “Today we stop trusting data by vibes. We add checks + cleaning so your EDA won’t lie.”
Do: show missingness report, a dedupe, and normalized categories.
Ask: “What’s worse: a pipeline that crashes early or one that silently gives wrong numbers?”
:::

---

## Today’s Flow

* **Session 1 (60m):** Data quality checks (fail fast)
* *Asr Prayer (20m)*
* **Session 2 (60m):** Missingness (measure → decide → flag)
* *Maghrib Prayer (20m)*
* **Session 3 (60m):** Duplicates + text normalization
* *Isha Prayer (20m)*
* **Hands-on (120m):** Implement `quality.py` + cleaning transforms + `run_day2_clean.py`

---

## Learning Objectives

By the end of today, you can: 

* write lightweight **data quality checks** (columns, non-empty, uniqueness, ranges)
* create a **missingness report** and add **missing flags**
* dedupe using a clear **business rule**
* normalize text categories so groupby results are consistent
* produce `data/processed/orders_clean.parquet` + a saved missingness report

---

## Warm-up (5 minutes) {.smaller}

Run Day 1 and confirm you still produce Parquet outputs.

**macOS/Linux**

```bash
source .venv/bin/activate
python scripts/run_day1_load.py
python -c "import pandas as pd; print(pd.read_parquet('data/processed/orders.parquet').dtypes)"
```

**Windows PowerShell**

```powershell
.\\.venv\\Scripts\\Activate.ps1
python scripts\\run_day1_load.py
python -c "import pandas as pd; print(pd.read_parquet('data/processed/orders.parquet').dtypes)"
```

**Checkpoint:** `orders.parquet` exists and `user_id` is `string`.

---

## Common setup issue: `ModuleNotFoundError: bootcamp_data` {.smaller}

If you see:

```text
ModuleNotFoundError: No module named 'bootcamp_data'
```

it usually means Python can’t see your **`src/`** folder.

**Fix (recommended, once per repo):** install your project in editable mode from the repo root:

```bash
pip install -e .
```

**Fix (quick + works for scripts):** add this to the top of `scripts/*.py` **before** importing `bootcamp_data`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
```

**Import checkpoint (works even without install):**

```bash
python -c "import sys; sys.path.insert(0, 'src'); import bootcamp_data; print('bootcamp_data import: ok')"
```

---

## Where we are in the weekly project

We are building the canonical flow: 

1. Load ✅ (Day 1)
2. Verify ✅ (starts today)
3. Clean ✅ (starts today)
4. Transform (Day 3)
5. Analyze (Day 4)
6. Visualize (Day 4)
7. Conclude (Day 5)

# Session 1

::: {.muted}
Data quality checks (fail fast)
:::

---

## Session 1 objectives

By the end of this session, you can: 

* explain “**fail fast**” and why it saves time
* implement checks for:

  * required columns
  * non-empty datasets
  * key integrity (unique + not null)
  * basic ranges (non-negative amounts)

---

## Context: most data bugs are silent

Bad data rarely crashes your code.

Instead, it produces:

* wrong totals
* wrong joins
* wrong charts
* confident wrong conclusions

Your job: turn assumptions into checks. 

---

## Concept: “fail fast”

**Fail fast** means:

* validate assumptions early
* stop the pipeline **with a clear error**
* fix the root cause before analysis

::: callout-tip
A pipeline that crashes early is annoying.
A pipeline that silently lies is dangerous.
:::

---

## What should we check first?

Start with the highest ROI checks: 

* Do we have the columns we need?
* Do we have **any rows**?
* Are keys present and reasonable?

---

## Check 1 — required columns

**Concept**

* “If column X is missing, nothing else matters.”
* Make the error message obvious.

**Example**

```python
def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
```

::: {.muted}
Pattern: required columns check.
:::

---

## Micro-exercise: break it on purpose (5 minutes)

1. Load `orders.parquet` into a DataFrame.
2. Call
```python
require_columns(
    df,
    [
        "order_id", "user_id", "amount",
        "created_at", "status", "NOT_A_COL"
    ],
)
```

**Checkpoint:** your code raises an error that names the missing column.

---

## Solution (example)

```python
import pandas as pd

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

orders = pd.read_parquet("data/processed/orders.parquet")
require_columns(orders, ["order_id","user_id","NOT_A_COL"])
```

---

## Quick Check

**Question:** Why do we check columns *before* other checks?

. . .

**Answer:** because other checks might crash with confusing errors if columns don’t exist.

---

## Check 2 — non-empty dataset

A surprising number of pipelines accidentally produce zero rows. 

Examples:

* wrong filter
* bad date window
* file path points to empty file

---

## Example: `assert_non_empty`

```python
def assert_non_empty(df, name="df"):
    assert len(df) > 0, f"{name} has 0 rows"
```

::: {.muted}
Fail fast on empty inputs.
:::

---

## Micro-exercise: when should it fail? (3 minutes)

You filtered orders like this:

```python
paid = orders[orders["status"] == "paid"]
```

When might `paid` be empty even if there are paid orders?

**Checkpoint:** give 2 reasons.

---

## Solution: common reasons

* `status` has inconsistent casing (`"Paid"`, `"PAID"`, `"paid"`) → exact match misses rows
* `status` has whitespace (`"paid "`) → exact match misses rows

(We fix this in Session 3.)

---

## Check 3 — key integrity (unique + not null)

Keys are how tables connect. If keys are wrong, everything is wrong. 

Common expectations:

* `users.user_id` is **unique**
* `orders.order_id` is **unique**
* key columns usually should not be missing

---

## Example: `assert_unique_key`

```python
def assert_unique_key(df, key, allow_na=False):
    if not allow_na:
        assert df[key].notna().all(), f"{key} contains NA"
    dup = df[key].duplicated(keep=False) & df[key].notna()
    assert not dup.any(), f"{key} not unique; {dup.sum()} duplicate rows"
```

::: {.muted}
Key uniqueness pattern.
:::

---

## Micro-exercise: pick the right key (4 minutes)

Which key should be unique?

A) `orders.user_id`
B) `users.user_id`
C) `orders.status`

**Checkpoint:** choose and explain in 1 sentence.

---

## Solution: pick the right key

* **B) `users.user_id`** should be unique (one row per user)
* `orders.user_id` repeats (many orders per user)
* `orders.status` repeats (category)

---

## Quick Check

**Question:** If you run `assert_unique_key(users, "user_id")` and it fails, what does that imply?

. . .

**Answer:** you don’t have a true “user lookup” table (joins will be unsafe).

---

## Check 4 — range checks (practical)

Examples: 

* `amount >= 0`
* `quantity >= 0`
* percentages in `[0, 1]`

But: missing values exist → handle them safely.

---

## Example: range check that ignores missing

```python
def assert_in_range(s, lo=None, hi=None, name="value"):
    x = s.dropna()
    if lo is not None:
        assert (x >= lo).all(), f"{name} below {lo}"
    if hi is not None:
        assert (x <= hi).all(), f"{name} above {hi}"
```

---

## Micro-exercise: where do checks go? (3 minutes)

When should you run checks?

A) only at the end
B) only at the start
C) before AND after major transforms

**Checkpoint:** choose A/B/C.

---

## Solution: where do checks go?

**C) before AND after** major transforms. 

* before: catch bad inputs early
* after: confirm your transform didn’t break invariants

---

## Session 1 recap

* Data bugs are usually silent → add checks
* Start with high ROI:

  * required columns
  * non-empty
  * key integrity
  * ranges
* Checks should run before and after major steps 

## Where today’s code lives (separation of concerns) {.smaller}

Keep responsibilities separate so your pipeline stays debuggable:

* `src/bootcamp_data/quality.py` → **checks** (fail fast)
* `src/bootcamp_data/transforms.py` → **pure transforms** (`df -> df`)
* `src/bootcamp_data/io.py` → I/O (read/write)
* `scripts/run_day2_clean.py` → thin entrypoint (wires everything together)

::: callout-tip
Rule: don’t mix I/O, checks, and transforms in one giant function.
:::

# Asr break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we will measure missingness and decide what to do about it.

# Session 2

::: {.muted}
Missingness (measure → decide → flag)
:::

---

## Session 2 objectives

By the end of this session, you can: 

* measure missingness per column (counts + percentages)
* explain why blanket `.dropna()` is risky
* decide between **drop / impute / flag**
* create `missingness_report(df)`
* add `*_isna` boolean flags for important columns

---

## Context: missingness is information

Missing values can mean: 

* unknown
* not applicable
* not recorded due to a bug
* filtered away upstream

You must measure it before you “fix” it.

---

## Concept: missingness report

We want a table like:

* column name
* number missing
* percent missing

---

## Example: `missingness_report(df)` 

```python
def missingness_report(df):
    n = len(df)
    return (
        df.isna().sum()
          .rename("n_missing")
          .to_frame()
          .assign(p_missing=lambda t: t["n_missing"] / n)
          .sort_values("p_missing", ascending=False)
    )
```

---

## Micro-exercise: run the report (6 minutes)

1. Load `data/processed/orders.parquet`
2. Run `missingness_report(orders)`
3. Print the top 5 rows

**Checkpoint:** you can name the most-missing column.

---

## Solution (example)

```python
import pandas as pd

orders = pd.read_parquet("data/processed/orders.parquet")

rep = missingness_report(orders)
print(rep.head(5))
```

::: {.notes}
Expected: amount may have missing if parsing coerced invalid values; quantity may have missing; created_at may have invalid strings but not NA yet.
:::

---

## Quick Check

**Question:** What’s the worst default missingness “fix”?

. . .

**Answer:** `.dropna()` everywhere (it silently deletes data and biases results). 

---

## Decide: drop vs impute vs flag

A simple decision rule (analytics/EDA): 

* **Drop**: only when rows are invalid and unusable
* **Impute**: only with strong justification
* **Flag**: often the best default (`*_isna`)

::: callout-tip
Flags let you analyze “missingness patterns” later (missing is not random).
:::

---

## Example: missingness flags

```python
def add_missing_flags(df, cols):
    out = df.copy()
    for c in cols:
        out[f"{c}__isna"] = out[c].isna()
    return out
```

::: {.muted}
Flag pattern for EDA safety.
:::

---

## Micro-exercise: add flags (5 minutes)

Add flags for:

* `amount`
* `quantity`

**Checkpoint:** your DataFrame has `amount__isna` and `quantity__isna`.

---

## Solution (example)

```python
orders2 = add_missing_flags(orders, ["amount", "quantity"])
output = orders2[["amount", "amount__isna", "quantity", "quantity__isna"]]
print(output.head())
```

---

## Why flags help later

With flags, you can ask:

* Are missing amounts more common in a country?
* Are missing quantities associated with a specific status?
* Did missingness spike after a certain date?

This prevents “cleaning away the story.” 

---

## Micro-exercise: one missingness question (4 minutes)

Pick one:

A) “What % of rows have missing `amount`?”
B) “Is missing `amount` higher for `status == 'refund'`?”
C) “Do missing values cluster by user?”

**Checkpoint:** write the question as a single sentence.

---

## Solution: good missingness questions

All A/B/C are valid.

The key: quantify + compare (don’t guess). 

---

## Session 2 recap

* Always measure missingness first
* Avoid blanket `.dropna()`
* Default to **flagging** missing values for analysis
* Save a missingness report as an artifact you can reference later 

# Maghrib break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we will handle duplicates and normalize text categories.

# Session 3

::: {.muted}
Duplicates + text normalization
:::

---

## Session 3 objectives

By the end of this session, you can: 

* define duplicates using **business keys**
* dedupe with a clear rule (keep latest / most complete)
* normalize text categories (trim + casefold + whitespace)
* apply a mapping dictionary for controlled category cleanup

---

## Context: duplicates cause double counting

Duplicates can:

* inflate revenue
* inflate counts
* break uniqueness assumptions for joins

And they often appear from re-extraction or bad joins. 

---

## Concept: duplicates are defined by business keys

Not all duplicates are exact row duplicates.

Examples:

* two rows share same `order_id` but different timestamps
* same user record appears twice with different countries

Define duplicates by what should be unique.

---

## Micro-exercise: define duplicate keys (4 minutes)

For orders, what is the most likely “should be unique” key?

A) `order_id`
B) `status`
C) `amount`

**Checkpoint:** choose A/B/C.

---

## Solution: duplicate keys

**A) `order_id`** should be unique (one row per order). 

---

## Concept: dedupe rule must be explicit

Common dedupe policies: 

* keep the **latest** record (by timestamp)
* keep the **most complete** record (fewest missing values)
* keep the **highest priority** source

If you don’t define a rule, you might delete real events.

---

## Example: keep latest (by timestamp) 

```python
def dedupe_keep_latest(df, key_cols, ts_col):
    return (
        df.sort_values(ts_col)
          .drop_duplicates(subset=key_cols, keep="last")
          .reset_index(drop=True)
    )
```

::: callout-note
If you don’t have a reliable timestamp yet, you can still dedupe by “keep last row seen” (temporary).
:::

---

## Quick Check

**Question:** Why is `drop_duplicates()` without `subset=` risky?

. . .

**Answer:** it only removes exact row duplicates and may miss key-level duplicates.

---

## Context: text normalization prevents fake categories

Common failure:

* `"Paid"`, `"paid"`, `"PAID"`, `" paid "` become four groups in `groupby()`.

We normalize text before analysis. 

---

## Concept: normalize text safely

A good default: 

* `strip()` → remove edges
* `casefold()` → robust lowercase
* collapse internal whitespace

---

## Example: `normalize_text(series)` 

```python
import re

_ws = re.compile(r"\s+")

def normalize_text(s):
    return (
        s.astype("string")
         .str.strip()
         .str.casefold()
         .str.replace(_ws, " ", regex=True)
    )
```

---

## Micro-exercise: normalize `status` (6 minutes)

1. Load `orders.parquet`
2. Print `orders["status"].unique()`
3. Create a normalized version and print unique values again

**Checkpoint:** you reduce categories (e.g., `Paid/PAID/paid` → `paid`).

---

## Solution (example)

```python
orders = orders.assign(status_norm=normalize_text(orders["status"]))
print("before:", orders["status"].unique())
print("after:", orders["status_norm"].unique())
```

---

## Concept: controlled mapping (synonyms/typos)

After normalization, map known variants:

* `"refunded"` → `"refund"`
* `"payment complete"` → `"paid"`

Keep the mapping small and explicit. 

---

## Example: `apply_mapping(series, mapping)` 

```python
def apply_mapping(s, mapping):
    return s.map(lambda x: mapping.get(x, x))
```

---

## Micro-exercise: write a tiny mapping (4 minutes)

Create:

```python
mapping = {
  "paid": "paid",
  "refund": "refund",
}
```

Add one more synonym you’ve seen (or imagine you’ll see).

**Checkpoint:** mapping leaves unknown values unchanged.

---

## Solution (example)

```python
mapping = {
    "paid": "paid",
    "refund": "refund",
    "refunded": "refund",
}

orders = orders.assign(
    status_clean=apply_mapping(orders["status_norm"], mapping),
)
print(orders["status_clean"].unique())
```

---

## Best practice: keep the raw column

When possible:

* keep `status_raw` (original)
* create `status_clean` (normalized + mapped)

This supports audits and debugging. 

---

## Session 3 recap

* Define duplicates by business keys
* Dedupe with an explicit “keep” rule
* Normalize text to prevent fake categories
* Use controlled mapping (small, explicit, auditable) 

# Isha break {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

## 20 minutes

**When you return:** we’ll implement the checks + cleaning in your repo and write cleaned outputs.

# Hands-on

::: {.muted}
Build: `quality.py` + cleaning transforms + cleaned Parquet outputs
:::

---

## Hands-on success criteria (today)

By the end, you should have: 

* `src/bootcamp_data/quality.py` with core checks
* new cleaning helpers in `src/bootcamp_data/transforms.py`
* a script `scripts/run_day2_clean.py` that writes:

  * `data/processed/orders_clean.parquet`
  * `reports/missingness_orders.csv` (or `.md`)
* at least one new commit pushed to GitHub

---

## Updated project layout (what you add today)

```text
src/bootcamp_data/
  config.py
  io.py
  transforms.py      # add missingness + normalize + dedupe helpers
  quality.py         # NEW today
scripts/
  run_day1_load.py
  run_day2_clean.py  # NEW today
reports/
  missingness_orders.csv
data/processed/
  orders.parquet
  orders_clean.parquet
```

---

## Task 1 — Create `quality.py` (20 minutes) {.smaller}

Create `src/bootcamp_data/quality.py` with:

* `require_columns(df, cols)`
* `assert_non_empty(df, name="df")`
* `assert_unique_key(df, key, allow_na=False)`
* `assert_in_range(series, lo=None, hi=None, name="value")` (ignore missing)

**Checkpoint:** you can import all functions from `bootcamp_data.quality`.

```bash
python -c "import sys; sys.path.insert(0, 'src'); \
           from bootcamp_data.quality import require_columns, \
           assert_non_empty, assert_unique_key, assert_in_range; \
           print('quality imports: ok')"
```

---

## Hint — keep checks lightweight

::: callout-tip
Start with simple `assert` statements + clear messages.
You can upgrade to custom exceptions later.
:::

Keep the checks readable and small (high ROI). 

---

## Solution — `src/bootcamp_data/quality.py` {.smaller}


```python
import pandas as pd

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def assert_non_empty(df: pd.DataFrame, name: str = "df") -> None:
    assert len(df) > 0, f"{name} has 0 rows"

def assert_unique_key(df: pd.DataFrame, key: str, *, allow_na: bool = False) -> None:
    if not allow_na:
        assert df[key].notna().all(), f"{key} contains NA"
    dup = df[key].duplicated(keep=False) & df[key].notna()
    assert not dup.any(), f"{key} not unique; {dup.sum()} duplicate rows"

def assert_in_range(s: pd.Series, lo=None, hi=None, name: str = "value") -> None:
    x = s.dropna()
    if lo is not None:
        assert (x >= lo).all(), f"{name} below {lo}"
    if hi is not None:
        assert (x <= hi).all(), f"{name} above {hi}"
```

---

## Task 2 — Add missingness helpers (15 minutes)

In `src/bootcamp_data/transforms.py`, add:

* `missingness_report(df) -> DataFrame`
* `add_missing_flags(df, cols) -> DataFrame`

**Checkpoint:** you can run missingness report on `orders.parquet`.

---

## Solution — missingness helpers (append to `transforms.py`) {.smaller}


```python
import pandas as pd

def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna().sum()
        .rename("n_missing")
        .to_frame()
        .assign(p_missing=lambda t: t["n_missing"] / len(df))
        .sort_values("p_missing", ascending=False)
    )

def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}__isna"] = out[c].isna()
    return out
```

---

## Task 3 — Add text normalization helpers (20 minutes)

In `src/bootcamp_data/transforms.py`, add:

* `normalize_text(series) -> series`
* `apply_mapping(series, mapping) -> series`

**Checkpoint:** `normalize_text` turns `Paid/PAID/ paid ` into `paid`.

---

## Solution — text normalization helpers

```python
import re
import pandas as pd

_ws = re.compile(r"\s+")

def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(_ws, " ", regex=True)
    )

def apply_mapping(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return s.map(lambda x: mapping.get(x, x))
```

---

## Task 4 — Add a dedupe helper (10 minutes)

In `src/bootcamp_data/transforms.py`, add:

* `dedupe_keep_latest(df, key_cols, ts_col) -> df`

**Checkpoint:** function exists and returns a DataFrame.

---

## Solution — dedupe helper {.smaller}

```python
import pandas as pd

def dedupe_keep_latest(df: pd.DataFrame, key_cols: list[str], ts_col: str) -> pd.DataFrame:
    return (
        df.sort_values(ts_col)
          .drop_duplicates(subset=key_cols, keep="last")
          .reset_index(drop=True)
    )
```

---

## Task 5 — Write `run_day2_clean.py` (25 minutes) {.smaller}

Create `scripts/run_day2_clean.py` that:

1. loads raw CSVs (orders + users)
2. runs basic checks (columns + non-empty)
3. enforces schema (from Day 1)
4. creates a missingness report and saves it to `reports/`
5. normalizes `status` into `status_clean`
6. adds missing flags for `amount` and `quantity`
7. writes `orders_clean.parquet`

**Checkpoint:** script runs end-to-end and writes outputs.

---

## Hint — order of operations

Good order:

* verify columns + non-empty (fast)
* enforce schema (types)
* missingness report (so you see problems)
* clean text + add flags
* write processed output

::: callout-warning
Don’t “validate uniqueness” before you dedupe (if you expect duplicates).
Validate after cleaning.
:::

---

## Solution — `scripts/run_day2_clean.py` {.smaller auto-animate=true}

```python
import logging
import sys
from pathlib import Path

from bootcamp_data.config import make_paths
from bootcamp_data.io import read_orders_csv, read_users_csv, write_parquet
from bootcamp_data.transforms import (
    enforce_schema,
    missingness_report,
    add_missing_flags,
    normalize_text,
    apply_mapping,
)
from bootcamp_data.quality import (
    require_columns,
    assert_non_empty,
)

log = logging.getLogger(__name__)
def main() -> None:
    ... # continue on the next slide

if __name__ == "__main__":
    main()
```

---

## Solution — `scripts/run_day2_clean.py` {.smaller auto-animate=true}

```python
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = make_paths(ROOT)

    log.info("Loading raw inputs")
    orders_raw = read_orders_csv(p.raw / "orders.csv")
    users = read_users_csv(p.raw / "users.csv")
    log.info("Rows: orders_raw=%s, users=%s", len(orders_raw), len(users))

    require_columns(orders_raw, ["order_id","user_id","amount","quantity","created_at","status"])
    require_columns(users, ["user_id","country","signup_date"])
    assert_non_empty(orders_raw, "orders_raw")
    assert_non_empty(users, "users")

    orders = enforce_schema(orders_raw)

    # Missingness artifact (do this early — before you “fix” missing values)
    rep = missingness_report(orders)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rep_path = reports_dir / "missingness_orders.csv"
    rep.to_csv(rep_path, index=True)
    log.info("Wrote missingness report: %s", rep_path)

    ... # continue on the next slide
```

::: aside
This is the body of `main()` defined in the previous slide.
:::

---

## Solution — `scripts/run_day2_clean.py` {.smaller auto-animate=true}

```python
    # Text normalization + controlled mapping
    status_norm = normalize_text(orders["status"])
    mapping = {"paid": "paid", "refund": "refund", "refunded": "refund"}
    status_clean = apply_mapping(status_norm, mapping)

    orders_clean = (
        orders.assign(status_clean=status_clean)
              .pipe(add_missing_flags, cols=["amount", "quantity"])
    )

    # Task 7: add at least one `assert_in_range(...)` check here (fail fast)

    write_parquet(orders_clean, p.processed / "orders_clean.parquet")
    write_parquet(users, p.processed / "users.parquet")
    log.info("Wrote processed outputs: %s", p.processed)
```

::: aside
This is the continuation of `main()`'s body defined in the previous slide.
:::

---

## Hint — order of operations

Good order:

* verify columns + non-empty (fast)
* enforce schema (types)
* missingness report (so you see problems)
* clean text + add flags
* write processed output

::: callout-warning
Don’t “validate uniqueness” before you dedupe (if you expect duplicates).
Validate after cleaning.
:::

---

## Task 6 — Run + verify artifacts (15 minutes)

Run the Day 2 script and verify:

* `data/processed/orders_clean.parquet` exists
* `reports/missingness_orders.csv` exists
* `status_clean` has consistent categories
* missing flags exist

**Checkpoint:** you can load `orders_clean.parquet` and inspect columns.

---

## Solution — run + verify

```bash
python scripts/run_day2_clean.py
python -c "import pandas as pd; \
           df=pd.read_parquet('data/processed/orders_clean.parquet'); \
           print(df.columns.tolist()); \
           print(df[\
             ['status','status_clean','amount__isna','quantity__isna'] \
           ].head())"
```

---

## Task 7 — Add one “fail fast” check to the script (10 minutes)

Add one check to `run_day2_clean.py`:

* `assert_in_range(orders_clean["amount"], lo=0, name="amount")`
* `assert_in_range(orders_clean["quantity"], lo=0, name="quantity")`

**Checkpoint:** script still runs (or fails with a clear message if data violates the rule).

---

## Solution — range checks (example)

```python
from bootcamp_data.quality import assert_in_range

assert_in_range(orders_clean["amount"], lo=0, name="amount")
assert_in_range(orders_clean["quantity"], lo=0, name="quantity")
```

---

## Git checkpoint (5 minutes)

* `git status`
* commit with message: `"w2d2: quality checks + cleaning + orders_clean output"`
* push to GitHub

**Checkpoint:** repo shows your new commit online.

---

## Solution — git commands

```bash
git add -A
git commit -m "w2d2: quality checks + cleaning + orders_clean output"
git push
```

---

## Debug playbook (Day 2 edition)

If something fails:

1. Print `df.dtypes`
2. Print `df.head()` and `df.isna().sum()`
3. Check text categories: `df["status"].value_counts(dropna=False)`
4. Re-run in small steps (comment out later steps temporarily)

::: callout-tip
Most cleaning bugs are “I assumed a value format that isn’t true.”
:::

---

## Stretch goals (optional)

If you finish early:

* Add `assert_unique_key(users, "user_id")` (users must be a lookup table)
* Save a “cleaning summary” JSON (row counts + missingness top 3)
* Add `README.md` section “Day 2: how to run cleaning”

---

## Exit Ticket

In 1–2 sentences:

**Why is `.dropna()` dangerous, and why are missing flags a safer default for EDA?** 

---

## What to do after class (Day 2 assignment) {.smaller}

**Due:** before Day 3 starts

1. Ensure `scripts/run_day2_clean.py` runs from a fresh terminal
2. Push your changes to GitHub
3. Confirm these files exist in your repo:

   * `src/bootcamp_data/quality.py`
   * `data/processed/orders_clean.parquet`
   * `reports/missingness_orders.csv`

**Deliverable:** GitHub repo link + screenshot of your `reports/missingness_orders.csv` opened in a viewer/spreadsheet.

::: callout-tip
Add 1–2 small commits today. Don’t wait until the end of the week.
:::

# Thank You! {background-image='{{< brand logo anim >}}' background-opacity='0.1'}

<div style="width: 300px">{{< brand logo full >}}</div>
