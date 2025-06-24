# Load RAVDESS dataset & Preprocessing

import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

class RAVDESSDataSet(Dataset):
    def __init__(self, dir, transform = None, target_sample_rate = 16000, duration = 3):
        self.paths = list(Path(dir).glob("**/*.wav"))
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=50,
            f_max= self.target_sample_rate // 2,
        )        
        self.classes = sorted([d.name for d in Path(dir).iterdir()])
        self.class_to_idx = {
            'neutral': 0,
            'calm': 1, 
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5,
            'disgust': 6,
            'surprise': 7
            }
        print(f"Found {len(self.paths)} sample in directory")
        print(f"Labels: {self.class_to_idx}")
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample_path = self.paths[idx]
        label_name = sample_path.parent.name
        label = self.class_to_idx[label_name]

        waveform, sample_rate = torchaudio.load(sample_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq= sample_rate, new_freq= self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim = 0, keepdim = True)
        melspectro = self._mel(waveform).log1p()
        return melspectro.squeeze(0), label
#        print(f"Shape: {melspectro.shape}")
#        print(sample_path)
#        print(f"Label: {self.label_name}, IDX: {self.label}")

def collate_pad(batch):
    feats, labels = zip(*batch)
    lengths = [f.shape[1] for f in feats]
    max_len = max(lengths)
    padded = torch.zeros(len(feats), 1, 128, max_len)
    for i, feat in enumerate(feats):
        padded[i, 0, :, : feat.shape[1]] = feat
    return padded, torch.tensor(labels), torch.tensor(lengths)


#REPO_ROOT = Path(__file__).resolve().parent  
#DATA_DIR = REPO_ROOT / "augmented_data" / "RAVDESS" / "train"
#a = RAVDESSDataSet(dir=DATA_DIR)
#x,y = a.__getitem__(0)

