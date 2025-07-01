import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Attention Pooling ------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim=256, attention_dim=64):
        super(AttentionPooling, self).__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x, mask=None):
        # x: (batch, time, features)
        scores = self.attn_layer(x).squeeze(-1)  # (batch, time)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)       # (batch, time)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, features)
        return context

# ------------------ MBConv Block ------------------
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, dropout=0.3):
        super(MBConvBlock, self).__init__()
        mid_channels = in_channels * expansion
        self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.project = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.expand(x)))
        out = F.relu(self.bn2(self.dwconv(out)))
        out = self.bn3(self.project(out))
        se_weight = self.se(out)
        out = out * se_weight
        out = self.pool(out)
        out = self.dropout(out)
        return out

# ------------------ Full Model ------------------
class ECLRA(nn.Module):
    def __init__(self, n_classes=8, input_freq_dim=144):
        super(ECLRA, self).__init__()

        self.input_freq_dim = input_freq_dim
        self.downsample_factor = 2 ** 4  # 4 maxpool layers (stem + 3 convs)
        final_H = self.input_freq_dim // self.downsample_factor

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        # MBConv Blocks
        self.mbconv1 = MBConvBlock(16, 64)
        self.mbconv2 = MBConvBlock(64, 128)
        self.mbconv3 = MBConvBlock(128, 128)

        # BiLSTM
        self.bi_lstm1 = nn.LSTM(input_size=128 * final_H, hidden_size=128, batch_first=True, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.4)

        # Attention Pooling
        self.attn_pool = AttentionPooling(input_dim=256, attention_dim=64)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, lengths):
        # x: (B, 1, 144, T)
        x = self.stem(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)  # (B, 128, H=144/16=9, T=Tp/16)

        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, H)
        x = x.view(B, T, -1)  # (B, T, C*H), should match LSTM input

        # Update lengths due to 4x pooling in time dim
        lengths = lengths // 16

        # First BiLSTM
        self.bi_lstm1.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bi_lstm1(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        x = self.lstm_dropout(x)

        # Second BiLSTM
        self.bi_lstm2.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bi_lstm2(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        x = self.lstm_dropout(x)

        # Attention mask
        idx = torch.arange(x.size(1), device=x.device)
        mask = idx.unsqueeze(0) < lengths.unsqueeze(1)

        # Attention pooling
        x = self.attn_pool(x, mask=mask)  # (B, 256)

        return self.classifier(x)
