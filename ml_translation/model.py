import torch
from torch import nn
from torch.nn import functional as F

class MTDyuFr(nn.Module):
    def __init__(self, src_len: int, tg_len: int, embedding_dim: int = 128, dropout: float = 0.1, padding_idx=-100):
        super(MTDyuFr, self, ).__init__()
        self.embedding = nn.Embedding(src_len, embedding_dim, max_norm=True, padding_idx=padding_idx)
        # Encoder
        self.encoder_gru1 = nn.GRU(embedding_dim, 128, bidirectional=True)
        self.encoder_gru2 = nn.GRU(256, 128, bidirectional=True,) # 256 because bidirectional doubles the features
        
        # Decoder
        self.decoder_gru = nn.GRU(256, 128, bidirectional=True,)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, tg_len),
            nn.Softmax(dim=-1)  # Correct usage of Softmax
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        
        # Encoder
        x, _ = self.encoder_gru1(x)
        x, _ = self.encoder_gru2(self.dropout(x))
        
        # Decoder
        x, _ = self.decoder_gru(self.dropout(x))
        
        # Fully connected layers
        x = self.fc(self.dropout(x))
        
        return x