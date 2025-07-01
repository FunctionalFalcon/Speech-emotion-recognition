from flask import Flask, render_template, request, jsonify
import torchaudio
import numpy as np
import torch
import soundfile as sf
import os
from datetime import datetime 
from pathlib import Path
import torch.nn.functional as F
from model.eclra import ECLRA  

# Set up the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Using device:", DEVICE)
if DEVICE.type == 'cuda':
    print("   > GPU:", torch.cuda.get_device_name(0))
    
# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, duration=3):
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=50,
            f_max=sample_rate // 2,
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=15,
            melkwargs={
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
                'f_min': 50,
                'f_max': sample_rate // 2,
            }
        )

        self.spec_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512
        )

    def compute_zcr(self, waveform, frame_length=1024, hop_length=512):
        x = waveform.squeeze(0)
        x = x.unfold(0, frame_length, hop_length)
        crossings = ((x[:, :-1] * x[:, 1:]) < 0).float()
        zcr = crossings.sum(dim=1) / frame_length
        return zcr.unsqueeze(0)

    def extract_features(self, waveform, original_sample_rate):
        # Resample if needed
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # MEL
        mel = self.mel_transform(waveform).log1p()
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        target_T = mel.shape[-1]

        # MFCC
        mfcc = self.mfcc_transform(waveform)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        mfcc = F.interpolate(mfcc, size=target_T, mode='linear', align_corners=False)

        # ZCR
        zcr = self.compute_zcr(waveform)
        zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-8)
        zcr = F.interpolate(zcr.unsqueeze(0), size=target_T, mode='linear', align_corners=False)

        # Concatenate all
        features = torch.cat([mel, mfcc, zcr], dim=1)

        return features  # Shape: (1, 144, T)

# ------------------ Flask App ------------------
app = Flask(__name__)

# Load model (replace with your actual model)
MODEL_PATH = REPO_ROOT / "Emotion_app/model/ECLRA.pth"
model = ECLRA(n_classes=8).to(DEVICE)
assert MODEL_PATH.exists(), f" Model file not found: {MODEL_PATH}"

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print(" Model loaded successfully.")
except:
    # Fallback for demo purposes - in production, you should have a proper model
    model = None
    print("Warning: Using mock model - replace with a real trained model")

# Emotion labels and colors (customize based on your model)
emotion_labels = {
    0: {"label": "Neutral", "color": "#808080"},
    1: {"label": "Calm", "color": "#00FF00"},
    2: {"label": "Happy", "color": "#FFFF00"},
    3: {"label": "Sad", "color": "#0000FF"},
    4: {"label": "Angry", "color": "#FF0000"},
    5: {"label": "Fearful", "color": "#800080"},
    6: {"label": "Disgusted", "color": "#FFA500"},
    7: {"label": "Surprised", "color": "#FFC0CB"}
}



@torch.no_grad()
def predict(audio_tensor, lengths):
    audio_tensor = audio_tensor.to(DEVICE)
    lengths = lengths.to(DEVICE)

    logits = model(audio_tensor, lengths)
    probs = torch.softmax(logits, dim=1)
    predicted = torch.argmax(probs, dim=1)

    result = {
        "predicted_class": emotion_labels[int(predicted)],
        "probabilities": {emotion_labels[i]: float(probs[0][i]) for i in range(8)}
    }
    return result



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    os.makedirs("uploads", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"uploads/recording_{timestamp}.wav"
    audio_file.save(audio_path)

    try:
        waveform, sr = torchaudio.load(audio_path)

        extractor = AudioFeatureExtractor(sample_rate=16000, duration=3)
        features = extractor.extract_features(waveform, sr)  # (1, 144, T)
        features = features.unsqueeze(0)
        
        T = features.shape[-1]  # Get the length of the feature sequence

        # Pad and reshape
        padded = torch.zeros((1, 1, 144, T))  # batch size 1
        padded[0, 0, :, :T] = features
        lengths = torch.tensor([T])
        with torch.no_grad():
            features = features.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(features, lengths)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]

        emotion_data = emotion_labels.get(predicted_class, {"label": "Unknown", "color": "#666666"})

        return jsonify({
            "emotion": emotion_data["label"],
            "confidence": float(confidence),
            "color": emotion_data["color"]
        })

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({"error": "Could not process audio"}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)