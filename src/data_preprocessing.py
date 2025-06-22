import os
import shutil
import random
from glob import glob

# input và output
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "RAVDESS")
output_folder = os.path.join(os.path.dirname(__file__), "augmented_data", "RAVDESS")

splits = ["train", "val", "test"]
ratios = [0.7, 0.15, 0.15]
random.seed(1337)

# all file wav from RAVDESS
all_files = glob(os.path.join(input_folder, "**", "*.wav"), recursive=True)
random.shuffle(all_files)

n_total = len(all_files)
n_train = int(n_total * ratios[0])
n_val = int(n_total * ratios[1])
n_test = n_total - n_train - n_val

split_map = {
    "train": all_files[:n_train],
    "val": all_files[n_train:n_train + n_val],
    "test": all_files[n_train + n_val:]
}

for split, files in split_map.items():
    split_dir = os.path.join(output_folder, split)
    os.makedirs(split_dir, exist_ok=True)
    for path in files:
        shutil.copy(path, os.path.join(split_dir, os.path.basename(path)))

print("Done splitting data.")

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

# Sắp xếp files theo emotion
for split in splits:
    folder = os.path.join(output_folder, split)
    for file in os.listdir(folder):
        # RAVDESS filename format: xx-xx-xx-xx-xx-xx-xx.wav
        # Emotion code: pos 3
        emo_id = file.split('-')[2]
        if emo_id in emotion_map:
            emo_name = emotion_map[emo_id]
            emo_folder = os.path.join(folder, emo_name)
            os.makedirs(emo_folder, exist_ok=True)
            shutil.move(os.path.join(folder, file), os.path.join(emo_folder, file))

print("Files sorted by emotions.")