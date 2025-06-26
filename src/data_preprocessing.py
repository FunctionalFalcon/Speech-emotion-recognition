import os
import shutil
import random
import torchaudio
from glob import glob
from pathlib import Path
import torch
from torch_audiomentations import Compose, AddColoredNoise, PitchShift, Shift, Gain 


# Map emotions cho RAVDESS dataset
emotion_map = {
    '01': 'neutral',
    '02': 'calm', 
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise'
}


def setup_paths():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    input_folder = REPO_ROOT / "data" / "RAVDESS"
    output_folder = REPO_ROOT / "src" / "augmented_data" / "RAVDESS"
    return input_folder, output_folder

# augmentation pipeline
def augmentation_pipeline():
    return Compose([
        AddColoredNoise(min_snr_in_db=10, max_snr_in_db=30, p=0.8),
        PitchShift(sample_rate = 16000, min_transpose_semitones=-4, max_transpose_semitones=4, p=0.6),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.7),
        Shift(min_shift=-0.3, max_shift=0.3, p=0.5)
    ])

def split_dataset(input_folder, ratios = [0.7, 0.15, 0.15]):
    all_files = glob(os.path.join(input_folder, "**", "*.wav"), recursive=True)
    random.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val    
    return {
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:]
    }    

def augment_audio_file(input_path, output_path, transform, target_sample_rate = 16000):
    try:
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != target_sample_rate:
            resample = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resample(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim = 0, keepdim= True)
        if transform:
            waveform = waveform.unsqueeze(0)
            augmented = transform(
                samples = waveform,
                sample_rate = target_sample_rate)
            augmented = augmented.squeeze(0)
        else:
            augmented = waveform
        torchaudio.save(output_path, augmented, target_sample_rate)
        return True
    except Exception as e:
        print(f"Transform failed. Error: {e}")
        return False

def process_file(path, split, split_dir, transform = None):
    """Process a single raw audio file based on split- 
    copy/augment and organize by emotion"""
    basename = os.path.basename(path)
    name_without_ext = os.path.splitext(basename)[0]

    emo_id = name_without_ext.split('-')[2]
    emo_name = emotion_map[emo_id]
    emo_folder = os.path.join(split_dir, emo_name)
    os.makedirs(emo_folder, exist_ok= True)

    original_path = os.path.join(emo_folder, basename)
    shutil.copy(path, original_path)
    if split == "train":
        for idx in range(3):
            aug_name = f"{name_without_ext}_aug{idx}.wav"
            aug_path = os.path.join(emo_folder, aug_name)
            augment_audio_file(path, aug_path, transform)
    return True

def main():
    random.seed(1337)
    input_folder, output_folder = setup_paths()
    augmentation_transforms = augmentation_pipeline()
    split_map = split_dataset(input_folder)

    # Process each split
    for split, files in split_map.items():
        split_dir = os.path.join(output_folder, split)
        os.makedirs(split_dir, exist_ok= True)
        print(f"Processing {split} split...")
        for i, path in enumerate(files):
            success = process_file(path, split, split_dir, transform = augmentation_transforms)
            if (i + 1 ) % 100 == 0:
                print(f"Processed {i+1}/{len(files)} files in {split}")

    print("Done splitting and augmenting data.")


if __name__ == "__main__":
    main()

