import torch
import torch.nn as nn


class _BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        y = self.dropout(y)
        return y[:, -1, :]


class FeatureLevelFusion(nn.Module):
    def __init__(
        self,
        semg_input_dim: int = 8,
        pressure_input_dim: int = 5,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_gestures: int = 14,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.semg_enc = _BiGRUEncoder(semg_input_dim, hidden_dim=hidden_dim, num_layers=num_gru_layers, dropout=dropout)
        self.press_enc = _BiGRUEncoder(
            pressure_input_dim, hidden_dim=hidden_dim, num_layers=num_gru_layers, dropout=dropout
        )
        fused_dim = self.semg_enc.out_dim + self.press_enc.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_gestures),
        )

    def forward(self, semg: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        semg_feat = self.semg_enc(semg)
        press_feat = self.press_enc(pressure)
        fused = torch.cat([semg_feat, press_feat], dim=-1)
        return self.classifier(fused)


class FeatFusion(FeatureLevelFusion):
    pass
