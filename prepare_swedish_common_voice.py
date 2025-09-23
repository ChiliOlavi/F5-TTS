import os
import shutil
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def prepare_swedish_common_voice(output_dir="data/swedish_tts_pinyin", split="train"):
    """
    Prepares the Swedish Common Voice dataset.

    Downloads the dataset from Hugging Face, copies mp3 files,
    and creates a metadata.csv file.

    Args:
        output_dir (str): The directory to save the dataset.
        split (str): The dataset split to use (e.g., 'train', 'validation', 'test').
    """
    # Load the dataset
    print(f"Loading Common Voice 'sv-SE' dataset, split: {split}...")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "sv-SE", trust_remote_code=True, split=split)

    # Create output directories
    mp3s_dir = os.path.join(output_dir, "wavs")
    os.makedirs(mp3s_dir, exist_ok=True)

    metadata = []

    # Process each item in the dataset
    for i, item in tqdm(enumerate(ds)):
        audio_path = item['audio']['path']
        sentence = item['sentence']

        # Define the output file path
        audio_filename = f"audio_{i+1:04d}.mp3"
        output_path = os.path.join(mp3s_dir, audio_filename)

        # Copy mp3 file
        shutil.copy(audio_path, output_path)

        # Add to metadata
        metadata.append({"audio_file": f"wavs/{audio_filename}", "text": sentence})

    # Create and save the metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, sep="|")

    print(f"Dataset preparation complete. Files are in {output_dir}")


if __name__ == "__main__":
    # Example usage:
    prepare_swedish_common_voice(output_dir="data/swedish_tts_pinyin", split="validated")