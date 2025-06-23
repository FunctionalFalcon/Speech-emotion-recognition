import torch
import torch.nn as nn

class CRNN(nn.Module):
    """Simple CNN followed by BiLSTM for SER."""

    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (64, T/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (32, T/4)
        )
        self.rnn = nn.LSTM(
            input_size=64 * 32, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, lengths):
        # x: (B, 1, 128, Tpad)
        x = self.conv(x)  # (B, C=64, F=32, T=Tp/4)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T, feat)
        # update lengths (integer division by 4 due to pooling twice)
        lengths = (lengths // 4)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Temporal average pooling over valid timesteps
        idx = torch.arange(out.size(1), device= x.device)
        mask = idx.unsqueeze(0) < lengths.unsqueeze(1)
        out = (out * mask.unsqueeze(2)).sum(1) / lengths.unsqueeze(1)
        return self.classifier(out)

