#!/usr/bin/env python3
"""
Build a 10x10 ratio matrix from:
1) merged_results.csv  (must have: 'unlearn_lang','eval_lang','Prob. Forget')
2) 10 fine-tuned CSVs, one per evaluated language (single row with 'Prob. Forget').

Cell(i, j) = ProbForget_unlearned[unlearn_lang=i, eval_lang=j] / ProbForget_finetuned(eval_lang=j)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
DATA_DIR = "/home/mila/a/alireza.farashah/TOFU/results/aya_npo_10lang"
BASE_DIR = "/home/mila/a/alireza.farashah/TOFU/results/aya_graddiff_10lang"
MERGED_FILE = os.path.join(DATA_DIR, "merged_results.csv")
FINETUNED_GLOB = os.path.join(
    BASE_DIR, "tofu_finetuned_5epoch_aya_10_lang*_retain_result01_unlearn_graddif_*_*.csv"
)

UNLEARN_COL = "unlearn_lang"
EVAL_COL    = "eval_lang"
PROB_COL    = "Prob. Retain"

OUT_PIVOT_CSV = os.path.join(DATA_DIR, "unlearned_prob_retain_pivot.csv")
OUT_RATIO_CSV = os.path.join(DATA_DIR, "prob_retain_ratio_matrix.csv")
OUT_HEATMAP   = os.path.join(DATA_DIR, "prob_retain_ratio_heatmap.png")


LANG_ORDER = ["en", "fr", "ru", "ar", "ja", "fa", "hi", "ko", "iw", "id"]

# ------------------------------------------------

def main():
    # 1) load merged + pivot to 10x10
    merged = pd.read_csv(MERGED_FILE)
    for c in [UNLEARN_COL, EVAL_COL, PROB_COL]:
        if c not in merged.columns:
            raise KeyError(f"Missing '{c}' in {MERGED_FILE}. Found: {list(merged.columns)}")

    heat_raw = merged.pivot_table(index=UNLEARN_COL, columns=EVAL_COL,
                                  values=PROB_COL, aggfunc="mean")

    # langs = sorted(set(heat_raw.index).union(set(heat_raw.columns)))
    langs = [l for l in LANG_ORDER if l in heat_raw.index or l in heat_raw.columns]
    heat_raw = heat_raw.reindex(index=langs, columns=langs)
    heat_raw.to_csv(OUT_PIVOT_CSV)

    # 2) baselines from fine-tuned files (single-row CSVs)
    finetuned_files = glob.glob(FINETUNED_GLOB)

    # infer language code from filename suffix (e.g., *_en.csv -> "en")
    lang_to_file = {}
    for fp in finetuned_files:
        base = os.path.basename(fp)
        lang_code = os.path.splitext(base)[0].split("_")[-1]
        lang_to_file[lang_code] = fp

    baselines = {}
    for lang in langs:
        fp = lang_to_file.get(lang)
        if fp and os.path.exists(fp):
            df_ft = pd.read_csv(fp)
            if PROB_COL not in df_ft.columns:
                raise KeyError(f"Expected '{PROB_COL}' in {fp}. Found: {list(df_ft.columns)}")
            # single-row file → take the only value
            baselines[lang] = float(df_ft[PROB_COL].iloc[0])
        else:
            baselines[lang] = np.nan  # missing → column ratios become NaN

    baseline_s = pd.Series(baselines).reindex(langs)

    # 3) ratio matrix = unlearned / fine-tuned baseline (by evaluated-language column)
    ratio = heat_raw.copy()
    for col in ratio.columns:
        denom = baseline_s[col]
        ratio[col] = ratio[col] / denom if pd.notna(denom) and denom != 0 else np.nan

    ratio.to_csv(OUT_RATIO_CSV)

    # 4) draw heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(ratio.values, aspect='auto')
    ax.set_xticks(np.arange(len(langs)))
    ax.set_yticks(np.arange(len(langs)))
    ax.set_xticklabels(langs, rotation=45, ha='right')
    ax.set_yticklabels(langs)
    ax.set_xlabel("Evaluated language (columns)")
    ax.set_ylabel("Unlearning language (rows)")
    ax.set_title("Ratio: Prob. Forget (unlearned / fine-tuned baseline)")

    vals = ratio.values
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Ratio")

    plt.tight_layout()
    plt.savefig(OUT_HEATMAP, dpi=200)
    plt.close(fig)

    print(f"Languages: {langs}")
    print(f"Saved raw pivot → {OUT_PIVOT_CSV}")
    print(f"Saved ratio matrix → {OUT_RATIO_CSV}")
    print(f"Saved heatmap → {OUT_HEATMAP}")

if __name__ == "__main__":
    main()
