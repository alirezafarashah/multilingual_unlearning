import asyncio
import pandas as pd
from googletrans import Translator
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import argparse


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
    tasks = [translate_text_with_retries(translator, text, dest_lang) for text in column]
    
    translated_texts = await asyncio.gather(*tasks)  # Run translations concurrently
    return translated_texts


def translate_module(data, dest_lang, num_workers=16, output_name="translated"):
    """Translates all columns of the dataset asynchronously."""
    translated_data = {col: [] for col in data.column_names}

    async def process_translation():
        for col in data.column_names:
            print(f"Translating column: {col}")
            translated_data[col] = await parallel_translate(data[col], dest_lang=dest_lang, max_workers=num_workers)

    asyncio.run(process_translation())  # Run async translation

    # Save translated data to a new dataset
    translated_dataset = Dataset.from_dict(translated_data)
    dataset_dict = DatasetDict({"train": translated_dataset})
    output_path = f"{output_name}_{dest_lang}"
    dataset_dict.save_to_disk(output_path)
    print(f"Translated dataset saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Tofu Dataset")
    parser.add_argument("--data_path", type=str, default="locuslab/TOFU", help="Path to the dataset")
    parser.add_argument("--split", type=str, default="forget05_perturbed", help="Split of the dataset to translate")
    parser.add_argument("--dest_lang", type=str, default="fr", help="Target language for translation")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers for translation")
    args = parser.parse_args()

    # Load dataset
    try:
        data = load_dataset(args.data_path, args.split)["train"]
        translate_module(data=data, dest_lang=args.dest_lang, num_workers=args.num_workers, output_name=args.split)
    except Exception as e:
        print(f"Error loading or translating dataset: {e}")

