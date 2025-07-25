
# Speech Emotion Recognition with ECLRA

This project implements a speech emotion recognition (SER) system using a hybrid CNN + BiLSTM + Attention model called ECLRA, trained on the RAVDESS dataset.
Highlights
Model: ECLRA, consists of MBConv (EfficientNet-style) + BiLSTM + Attention 

Preprocessing: Mel Spectrograms, MFCC, ZCR

Augmentation: Noise, pitch/gain/time shift

Metrics: Accuracy, F1, UA Recall



# Quick Start

### Step 1: Clone repo
```bash
git clone https://github.com/FunctionalFalcon/Speech-emotion-recognition.git
cd Speech-emotion-recognition
```
### Step 2: Prepare data
Place RAVDESS dataset in data/RAVDESS/

### Step 3: Preprocess
```bash
cd src/data
python preprocessing.py
```
### Step 4: Train
```bash
cd python train.py
```
### Step 5: Evaluate
```bash
python test.py
```


# Model Architecture
Input → MBConv Blocks → BiLSTM × 2 → Attention → FC → Softmax


# Performance
Metric	Score
Val Accuracy	~77%
UA Recall	~75%

# How to use
```bash
python Emotion_app/app.py
```
