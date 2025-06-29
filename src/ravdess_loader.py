# Load RAVDESS dataset & Preprocessing

import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

class RAVDESSDataSet(Dataset):
    def __init__(self, dir, features = 'mel', transform = None, target_sample_rate = 16000, duration = 3):
        self.paths = list(Path(dir).glob("**/*.wav"))
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
        self.features = features
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=50,
            f_max= self.target_sample_rate // 2,
        )        
        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate = self.target_sample_rate,
            n_mfcc = 15,
            melkwargs = {
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels':128,
                'f_min':50,
                'f_max': self.target_sample_rate // 2,
            }
        )
        self.spec_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate = self.target_sample_rate,
            n_fft = 1024,
            hop_length = 512
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

    def __compute_zcr__(self,waveform, frame_length = 1024, hop_length = 512):
        #waveform shape: (1,T), raw input of .wav files
        x = waveform.squeeze(0)
        x = x.unfold(0,frame_length,hop_length)
        crossings = ((x[:, :-1] * x[:, 1:]) < 0).float()
        zcr = crossings.sum(dim=1) / frame_length  # (num_frames,)
        return zcr.unsqueeze(0) 
    
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
        mel = self._mel(waveform).log1p()
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        
        target_T = mel.shape[-1]
        if self.features == "mel":
            return mel, label

        mfcc = self._mfcc(waveform)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        mfcc = F.interpolate(mfcc, size=target_T, mode='linear', align_corners=False)

        zcr = self.__compute_zcr__(waveform)
        zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-8)
        zcr = F.interpolate(zcr.unsqueeze(0), size=target_T, mode='linear', align_corners=False)

        feat = torch.cat([mel,mfcc,zcr],dim = 1)
        
        return feat.squeeze(0), label # (1, 144, T)

def collate_pad(batch):
    feats, labels = zip(*batch)
    lengths = [f.shape[1] for f in feats]
    max_len = max(lengths)
    padded = torch.zeros(len(feats), 1, 144, max_len)
    for i, feat in enumerate(feats):
        padded[i, 0, :, : feat.shape[1]] = feat
    return padded, torch.tensor(labels), torch.tensor(lengths)

# TEST
#REPO_ROOT = Path(__file__).resolve().parent  
#DATA_DIR = REPO_ROOT / "augmented_data" / "RAVDESS" / "train"
#a = RAVDESSDataSet(dir=DATA_DIR, features= "not mel")
#x,y = a.__getitem__(0)
#print(f"debug: {x}")