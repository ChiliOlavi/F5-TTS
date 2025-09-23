import os
import shutil
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
os.environ['HF_HOME'] = 'G:/.cache/huggingface'
print(os.getenv('HF_HOME'))


def prepare_swedish_common_voice(output_dir="data/swedish_parliament_data", split="train"):
    """
    Prepares the Swedish Common Voice dataset.

    Downloads the dataset from Hugging Face, copies mp3 files,
    and creates a metadata.csv file.

    Args:
        output_dir (str): The directory to save the dataset.
        split (str): The dataset split to use (e.g., 'train', 'validation', 'test').
    """
    # Load the dataset
    print(f"Loading RixVox dataset, split: {split}...")
    ds = load_dataset("KBLab/rixvox", split=split, trust_remote_code=True, cache_dir="G:/.cache/huggingface")

    # Create output directories
    mp3s_dir = os.path.join(output_dir, "wavs")
    os.makedirs(mp3s_dir, exist_ok=True)

    metadata = []
    saved_files_count = 0

    # Process each item in the dataset
    for item in tqdm(ds):
        audio_array = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        
        # Calculate duration
        duration = len(audio_array) / sampling_rate

        # Filter for clips between 5 and 15 seconds
        if duration < 5 or duration > 15:
            continue

        audio_path = item['audio']['path']
        sentence = item['text']

        # Define the output file path
        saved_files_count += 1
        audio_filename = f"audio_{saved_files_count:04d}.wav"
        output_path = os.path.join(mp3s_dir, audio_filename)

        # Copy wav file
        shutil.copy(audio_path, output_path)

        # Add to metadata
        metadata.append({"audio_file": f"wavs/{audio_filename}", "text": sentence})

    # Create and save the metadata CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, sep="|")

    total_files = len(ds)
    if total_files > 0:
        percentage = (saved_files_count / total_files) * 100
        print(f"Included {saved_files_count} of {total_files} files ({percentage:.2f}%).")

    print(f"Dataset preparation complete. Files are in {output_dir}")


if __name__ == "__main__":
    # Example usage:
    
    prepare_swedish_common_voice(output_dir="data/swedish_parliament_data", split="train[:30%]")