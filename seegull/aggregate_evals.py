import os
from glob import glob
import pandas as pd

# Base directory where eval_prob.py wrote results
BASE_DIR = "./eval_results_full"
OUT_CSV = "summary_eval_results.csv"

def coerce_bool(series):
    """
    Robustly convert a pandas Series to boolean.
    Accepts True/False, 1/0, 'True'/'False', 'true'/'false', etc.
    Anything not truthy becomes False.
    """
    if series.dtype == bool:
        return series
    # Handle NaNs, strings, ints
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

def parse_model_lang_from_path(csv_path: str):
    """
    Expect: eval_results/<model_name>/<lang>/<file>.csv
    Fallback to parsing from filename if needed.
    """
    parts = csv_path.replace("\\", "/").split("/")
    model, lang = None, None
    try:
        # .../eval_results/<model>/<lang>/<file>.csv
        model = parts[-3]
        lang = parts[-2]
    except Exception:
        pass

    if (not model) or (not lang):
        # fallback to filename pattern: <model>_<lang>.csv
        fname = os.path.basename(csv_path).rsplit(".", 1)[0]
        if "_" in fname:
            # take last token as lang, rest as model
            *m_tokens, lang = fname.split("_")
            model = "_".join(m_tokens) if m_tokens else fname

    return model or "UNKNOWN_MODEL", lang or "UNKNOWN_LANG"

def summarize_csv(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Skipping {csv_path}: {e}")
        return None

    if "is_correct" not in df.columns or "predicted_label" not in df.columns:
        print(f"[WARN] Missing required columns in {csv_path}; skipping.")
        return None

    total = len(df)
    is_correct = coerce_bool(df["is_correct"])
    correct = int(is_correct.sum())
    unknown = int((df["predicted_label"].astype(str) == "Unknown").sum())
    other_wrong = int(total - correct - unknown)

    acc = correct / total if total else 0.0
    unk_rate = unknown / total if total else 0.0
    other_wrong_rate = other_wrong / total if total else 0.0

    model, lang = parse_model_lang_from_path(csv_path)

    return {
        "model": model,
        "language": lang,
        "file": csv_path,
        "total": total,
        "correct": correct,
        "unknown": unknown,
        "other_wrong": other_wrong,
        "accuracy": round(acc, 6),
        "unknown_rate": round(unk_rate, 6),
        "other_wrong_rate": round(other_wrong_rate, 6),
    }

def main():
    rows = []
    pattern = os.path.join(BASE_DIR, "**", "*.csv")
    for csv_path in glob(pattern, recursive=True):
        # Skip any non-eval CSVs if they exist
        # e.g., you could add a guard like:
        # if not os.path.basename(csv_path).startswith("..."):
        #     continue
        summary = summarize_csv(csv_path)
        if summary:
            rows.append(summary)

    if not rows:
        print(f"No CSV files found under {BASE_DIR}")
        return

    out_df = pd.DataFrame(rows).sort_values(["model", "language"]).reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved summary to {OUT_CSV}")
    # Optional: print a quick pivot (accuracy by model/lang)
    try:
        pivot = out_df.pivot_table(index="model", columns="language", values="accuracy")
        print("\nAccuracy pivot (model × language):")
        print(pivot)
    except Exception:
        pass

if __name__ == "__main__":
    main()
