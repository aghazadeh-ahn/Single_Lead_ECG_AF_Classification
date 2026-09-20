import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, pool_size: int = 2, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)

class CRNN(nn.Module):
    """
    CNN-BiLSTM Architecture for Single-Lead ECG Classification:
    - Feature Extractor: 4-stage 1D CNN with large kernels
    - Temporal Aggregator: 2-layer Bidirectional LSTM
    - Classifier: Fully-Connected Head with Dropout
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 1, 
                 cnn_channels=(32, 64, 128, 256), 
                 lstm_hidden: int = 128, 
                 lstm_layers: int = 2, 
                 dropout: float = 0.4):
        super().__init__()

        # 1. Feature Extractor (CNN)
        conv_layers = []
        curr_in = in_channels
        for out_ch in cnn_channels:
            conv_layers.append(ConvBlock(curr_in, out_ch, kernel_size=15, pool_size=2, dropout=0.2))
            curr_in = out_ch
        self.feature_extractor = nn.Sequential(*conv_layers)

        # 2. Sequence Modeler (Bi-LSTM)
        # Input shape to LSTM: (batch, seq_len_downsampled, cnn_channels[-1])
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # 3. Classifier Head (BiLSTM output has dimension 2 * lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)

        # CNN Feature extraction -> (batch, ch_out, downsampled_len)
        feat = self.feature_extractor(x)

        # Reshape for LSTM -> (batch, seq_len, features)
        feat = feat.permute(0, 2, 1)

        # LSTM Temporal processing
        lstm_out, _ = self.lstm(feat)  # (batch, seq_len, hidden*2)

        # Global Average Pooling over time sequence
        out_pooled = torch.mean(lstm_out, dim=1)  # (batch, hidden*2)

        logits = self.classifier(out_pooled)
        return logits

if __name__ == "__main__":
    model = CRNN(num_classes=4)
    dummy_input = torch.randn(8, 1, 9000)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # torch.Size([8, 4])
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
