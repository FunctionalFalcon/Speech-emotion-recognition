import torch
import torch.nn as nn

class ECLR(nn.Module):
    """Simple CNN followed by BiLSTM for SER."""

    def __init__(self, n_classes):
        super().__init__()
        r = 4 # MBConv expansion rate
        C1, C2, C3 = 32,64,128

        self.bl1 = nn.Sequential(
            nn.Conv2d(1, C1, kernel_size=3, padding=1),
            nn.BatchNorm2d(C1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (32,64, T/2)
        )
        

        self.bl2 = nn.Sequential(
            nn.Conv2d(C1, C1, kernel_size=3, padding=1,groups = C1),
            nn.Conv2d(C1, C2, kernel_size=1),
            nn.BatchNorm2d(C2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (64,32, T/4)
        )


        self.bl3 = nn.Sequential(
            nn.Conv2d(C2, C2*r, kernel_size=1),
            nn.BatchNorm2d(C2*r),
            nn.ReLU(),
            nn.Conv2d(C2*r,C2*r, kernel_size=3, padding = 1, groups = C2*r),  #depthwise conv
            nn.BatchNorm2d(C2*r),
            nn.ReLU(),
            nn.Conv2d(C2*r, C3, kernel_size = 1),
            nn.BatchNorm2d(C3),
            nn.MaxPool2d((2,2))     # (128,16, T/8)
        )


        self.lstm = nn.LSTM(
            input_size = C3 * 16,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            bidirectional = True
        )


        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x, lengths):
        # x: (B, 1, 128, Tpad)
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)  # (B, C, F=16, T=Tp/8)
        
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T, feat)

        # update lengths (integer division by 8 due to pooling thrice)
        lengths = (lengths // 8)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Temporal average pooling over valid timesteps
        idx = torch.arange(out.size(1), device= x.device)
        mask = idx.unsqueeze(0) < lengths.unsqueeze(1)
        out = (out * mask.unsqueeze(2)).sum(1) / lengths.unsqueeze(1)

        return self.classifier(out)

