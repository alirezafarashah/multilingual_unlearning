import pandas as pd
import glob
import os
import re

# Relative path to your CSV files (adjust as needed)
path = "results/aya_graddiff_10lang/tofu_finetuned_5epoch_aya_retain_10_lang_2e5_full__retain_result01_unlearn_graddif_2e-05_*.csv"

# Output file (same folder)
output_file = "results/aya_graddiff_10lang/merged_results_retain.csv"

all_dfs = []

for file in glob.glob(path):
    filename = os.path.basename(file).replace(".csv", "")
    
    # Regex to extract languages
    match = re.search(r"forget\d+_\d+_([a-z]+)__.*_([a-z]+)$", filename)
    if match:
        unlearn_lang = match.group(1)
        eval_lang = match.group(2)
    else:
        unlearn_lang = "unknown"
        eval_lang = "unknown"

    # Load CSV
    df = pd.read_csv(file)

    # Add columns
    df["unlearn_lang"] = unlearn_lang
    df["eval_lang"] = eval_lang
    df["source_file"] = filename  # keep original file name

    all_dfs.append(df)

# Merge all
merged_df = pd.concat(all_dfs, ignore_index=True)

# Save to CSV
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as {output_file}")
