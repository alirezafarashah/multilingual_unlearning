import asyncio
import argparse

from datasets import load_dataset, load_from_disk
from googletrans import Translator
from sacrebleu import corpus_bleu
from tqdm import tqdm


# -------------------------------------------------------------------
# 1. SAME TRANSLATION STRUCTURE AS YOUR FIRST CODE
# -------------------------------------------------------------------

async def translate_text_with_retries(translator, text, dest_lang='fr', retries=3, delay=2):
    """Translates a text or a list of texts with retries in case of failure."""
    for attempt in range(retries):
        try:
            # If the input is a list, translate each item individually
            if isinstance(text, list):
                translations = []
                for item in text:
                    translated_item = await translator.translate(item, dest=dest_lang)
                    if not translated_item or not hasattr(translated_item, "text"):
                        raise ValueError(f"Unexpected translation response: {translated_item}")
                    translations.append(translated_item.text)
                return translations

            # Handle a single string
            translation = await translator.translate(text, dest=dest_lang)
            if not translation or not hasattr(translation, "text"):
                raise ValueError(f"Unexpected translation response: {translation}")
            return translation.text

        except Exception as e:
            print(f"Error during translation (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)  # Use asyncio.sleep instead of time.sleep
            else:
                return text  # Return original text or list if translation fails


async def parallel_translate(column, dest_lang='es', max_workers=8):
    """Translates a column of text in parallel using asyncio."""
    translator = Translator()  # Create a new instance for safety

    # Same structure: create one task per row
    tasks = [translate_text_with_retries(translator, text, dest_lang) for text in column]

    # Wrap in chunks for a visible progress bar, but keep structure
    translated = []
    for start in tqdm(range(0, len(tasks), max_workers), desc=f"Translating to {dest_lang}"):
        chunk = tasks[start:start + max_workers]
        results = await asyncio.gather(*chunk)
        translated.extend(results)

    return translated


# -------------------------------------------------------------------
# 2. BACK-TRANSLATION MODULE (REUSES THE SAME STRUCTURE)
# -------------------------------------------------------------------

def back_translate_module(data, num_workers=16, text_columns=None):
    """
    Back-translates the given dataset's columns to English using the same
    translation structure as the original code.
    Returns a dict: {column_name: list_of_backtranslated_strings}
    """
    if text_columns is None:
        text_columns = data.column_names

    back_translated = {col: [] for col in text_columns}

    async def process_translation():
        for col in text_columns:
            print(f"\nBack-translating column: {col}")
            back_translated[col] = await parallel_translate(
                data[col],
                dest_lang='en',
                max_workers=num_workers
            )

    asyncio.run(process_translation())
    return back_translated


# -------------------------------------------------------------------
# 3. BLEU COMPUTATION
# -------------------------------------------------------------------

def compute_bleu_for_column(original_texts, back_translated_texts):
    """
    Compute corpus BLEU between original English and back-translated English
    for a single column.
    """
    # Make sure lengths match
    n = min(len(original_texts), len(back_translated_texts))
    original_texts = [str(x) for x in original_texts[:n]]
    back_translated_texts = [str(x) for x in back_translated_texts[:n]]

    # sacrebleu.corpus_bleu expects:
    # hypotheses: List[str]
    # references: List[List[str]] (one list per reference set)
    refs = [original_texts]
    hyps = back_translated_texts

    bleu = corpus_bleu(hyps, refs)
    return bleu.score


# -------------------------------------------------------------------
# 4. MAIN SCRIPT
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Back-translate translated dataset to English and compute BLEU vs original English."
    )

    parser.add_argument(
        "--original_data_path",
        type=str,
        default="locuslab/TOFU",
        help="Path or HF hub id of the ORIGINAL (English) dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="full",
        help="Split of the original dataset (e.g., 'full', 'forget05_perturbed', etc.).",
    )
    parser.add_argument(
        "--translated_path",
        type=str,
        required=True,
        help="Path to the translated dataset on disk (output of save_to_disk, e.g. 'full_fr').",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for back-translation.",
    )
    parser.add_argument(
        "--text_columns",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of text columns to evaluate, e.g.: "
            "--text_columns prompt continuation. "
            "If not provided, all columns are used."
        ),
    )
    args = parser.parse_args()

    # 1) Load original dataset (English)
    print("Loading original dataset...")
    original_dataset = load_dataset(args.original_data_path, args.split)["train"]

    # 2) Load translated dataset from disk
    print("Loading translated dataset from disk...")
    translated_dataset_dict = load_from_disk(args.translated_path)
    translated_dataset = translated_dataset_dict["train"]

    if len(original_dataset) != len(translated_dataset):
        print(
            f"WARNING: original length ({len(original_dataset)}) != "
            f"translated length ({len(translated_dataset)})"
        )

    # 3) Decide which columns to evaluate
    if args.text_columns is None:
        text_columns = original_dataset.column_names
        print(f"No text_columns specified; using all columns: {text_columns}")
    else:
        text_columns = args.text_columns
        print(f"Evaluating text columns: {text_columns}")

    # 4) Back-translate translated dataset to English
    back_translated_data = back_translate_module(
        data=translated_dataset,
        num_workers=args.num_workers,
        text_columns=text_columns,
    )

    # 5) Compute BLEU per column
    print("\n========== BLEU SCORES (original EN vs back-translated EN) ==========")
    for col in text_columns:
        orig_col = original_dataset[col]
        back_col = back_translated_data[col]

        bleu_score = compute_bleu_for_column(orig_col, back_col)
        print(f"Column '{col}': BLEU = {bleu_score:.2f}")

    print("Done.")


if __name__ == "__main__":
    main()
