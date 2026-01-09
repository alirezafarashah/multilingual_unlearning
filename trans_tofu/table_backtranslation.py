import csv
import glob
import os

def get_language_name(code):
    """Optional: Helper to make the table look nicer. Add more if needed."""
    mapping = {
        'ar': 'Arabic',
        'de': 'German',
        'es': 'Spanish',
        'fr': 'French',
        'hi': 'Hindi',
        'it': 'Italian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'en': 'English'
    }
    return mapping.get(code, code.upper()) # Fallback to uppercase code if not found

def main():
    # 1. Find all CSV files matching the pattern
    # Assumes files are named like "bleu_results_ar.csv", "bleu_results_fr.csv"
    file_pattern = "bleu_results_*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found matching 'bleu_results_*.csv'")
        return

    summary_data = []

    print(f"Found {len(files)} files. Processing...")

    # 2. Loop through files and extract data
    for filepath in files:
        # Extract language code from filename (e.g., 'bleu_results_ar.csv' -> 'ar')
        filename = os.path.basename(filepath)
        try:
            # Split by '_' and take the last part, then remove .csv
            lang_code = filename.split('_')[-1].replace('.csv', '')
            lang_name = get_language_name(lang_code)
        except:
            lang_code = filename
            lang_name = filename

        global_score = None

        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Look for the row we created in the previous step
                # We check for "GLOBAL_AVERAGE" or "GLOBAL_DATASET" just in case
                if row['column'] in ['GLOBAL_AVERAGE', 'GLOBAL_DATASET']:
                    global_score = float(row['bleu'])
                    break
            
            # Fallback: If no global row found, maybe take the average of the other rows?
            # For now, we assume the script ran correctly and the row exists.
            if global_score is None:
                print(f"Warning: Could not find 'GLOBAL_AVERAGE' in {filename}")
                global_score = 0.0

        summary_data.append({
            "lang_code": lang_code,
            "language": lang_name,
            "score": global_score
        })

    # 3. Sort the data (Optional: Sort by Score High->Low)
    summary_data.sort(key=lambda x: x['score'], reverse=True)

    # 4. Generate Markdown Table Content
    md_lines = []
    md_lines.append(f"# Translation Back-Translation Quality Summary")
    md_lines.append(f"")
    md_lines.append(f"| Language | Code | BLEU Score |")
    md_lines.append(f"| :--- | :---: | :--- |")
    
    for item in summary_data:
        # Format score to 2 decimal places
        md_lines.append(f"| {item['language']} | {item['lang_code']} | {item['score']:.2f} |")

    output_text = "\n".join(md_lines)

    # 5. Save to .md file
    output_filename = "bleu_summary_table.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\nSuccess! Markdown table saved to: {output_filename}")
    print("-" * 30)
    print(output_text) # Print to console for verification

if __name__ == "__main__":
    main()