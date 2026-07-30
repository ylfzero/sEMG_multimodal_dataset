import torch
import torch.nn as nn
import torch.nn.functional as F


# data_manager 中 window_size∈{0.25,0.5,0.75} → 时间点数（采样率 500Hz 时为 *1000）
_COMMON_SEMG_LENGTHS = (250, 500, 750)


class _TemporalPool(nn.Module):
    """用 mean || max || last 替代仅取末帧，再投影回原维度。"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        mean = y.mean(dim=1)
        mx = y.max(dim=1).values
        last = y[:, -1, :]
        return self.proj(torch.cat([mean, mx, last], dim=-1))


class _BiGRUEncoder(nn.Module):
    """glove / pressure / flexion 分支：BiGRU + 全时段池化。"""

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
        self.pool = _TemporalPool(hidden_dim * 2, dropout=dropout)
        self.out_dim = self.pool.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        y = self.dropout(y)
        return self.pool(y)


class _EMGNetTrunk(nn.Module):
    """
    与 sEMGNet/EMGNet 完全一致的卷积前端（block_1/2/3）。

    输入兼容：
      - FeatFusion 管线: [B, T, C]
      - sEMGNet 原生:     [B, C, T]
    """

    def __init__(
        self,
        channel: int = 8,
        drop_out: float = 0.1,
        time_point: int = 9,
        n_t: int = 8,
        n_s: int = 16,
    ):
        super().__init__()
        if time_point % 2 == 0 or n_s // 2 == 1:
            raise ValueError("time_point 须为奇数，且 N_s//2 不能为奇数（与 EMGNet 一致）")
        self.channel = int(channel)

        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((time_point // 2, time_point // 2 + 1, 0, 0)),
            nn.Conv2d(1, n_t, (1, time_point), bias=False),
            nn.BatchNorm2d(n_t),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(n_t, n_s, (channel, 1), groups=n_t, bias=False),
            nn.BatchNorm2d(n_s),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop_out),
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((n_s // 2 - 1, n_s // 2, 0, 0)),
            nn.Conv2d(n_s, n_s, (1, n_s), groups=n_s, bias=False),
            nn.Conv2d(n_s, n_s, (1, 1), bias=False),
            nn.BatchNorm2d(n_s),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop_out),
        )

    def _to_bct(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"期望三维输入 [B,T,C] 或 [B,C,T]，得到 {tuple(x.shape)}")
        if x.shape[-1] == self.channel and x.shape[1] != self.channel:
            x = x.transpose(1, 2).contiguous()
        elif x.shape[1] != self.channel:
            raise ValueError(
                f"无法识别 sEMG 布局，期望通道维={self.channel}，得到 shape={tuple(x.shape)}"
            )
        return x

    def forward_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bct(x)
        x = F.pad(x, (0, 6))
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_maps(x).flatten(1)


def _probe_flat_dims(trunk: _EMGNetTrunk, channel: int, lengths=_COMMON_SEMG_LENGTHS) -> list[int]:
    """预计算常见窗口下 EMGNet flatten 维，便于在 __init__ 注册 Linear（可被优化器更新）。"""
    dims: list[int] = []
    trunk.eval()
    with torch.no_grad():
        for t in lengths:
            flat = trunk.forward_flat(torch.zeros(1, t, channel))
            d = int(flat.shape[1])
            if d not in dims:
                dims.append(d)
    trunk.train()
    return dims


def _make_classifier(in_dim: int, mlp_hidden: int, num_gestures: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, num_gestures),
    )


class FeatureLevelFusion(nn.Module):
    """特征级融合：sEMG = EMGNet 前端；glove = BiGRU。"""

    def __init__(
        self,
        semg_input_dim: int = 8,
        pressure_input_dim: int = 5,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_gestures: int = 14,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
        time_point: int = 9,
        n_t: int = 8,
        n_s: int = 16,
    ):
        super().__init__()
        self.semg_trunk = _EMGNetTrunk(
            channel=semg_input_dim,
            drop_out=dropout,
            time_point=time_point,
            n_t=n_t,
            n_s=n_s,
        )
        self.press_enc = _BiGRUEncoder(
            pressure_input_dim, hidden_dim=hidden_dim, num_layers=num_gru_layers, dropout=dropout
        )
        self._semg_proj_dim = int(self.press_enc.out_dim)
        self.semg_proj = nn.ModuleDict(
            {
                str(d): nn.Linear(d, self._semg_proj_dim)
                for d in _probe_flat_dims(self.semg_trunk, semg_input_dim)
            }
        )
        fused_dim = self._semg_proj_dim + self.press_enc.out_dim
        self.classifier = _make_classifier(fused_dim, mlp_hidden, num_gestures, dropout)

    def _project_semg(self, flat: torch.Tensor) -> torch.Tensor:
        key = str(int(flat.shape[1]))
        if key not in self.semg_proj:
            self.semg_proj[key] = nn.Linear(int(flat.shape[1]), self._semg_proj_dim).to(
                device=flat.device, dtype=flat.dtype
            )
        return self.semg_proj[key](flat)

    def forward(self, semg: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        semg_feat = self._project_semg(self.semg_trunk.forward_flat(semg))
        press_feat = self.press_enc(pressure)
        return self.classifier(torch.cat([semg_feat, press_feat], dim=-1))


class FeatFusion(FeatureLevelFusion):
    """sEMG + glove（mode0=flexion，mode1=pressure）双模态融合。"""


class FeatureLevelFusionSEMG(nn.Module):
    """FeatFusion 的 sEMG-only 消融：EMGNet 前端 + Linear 分类头。"""

    def __init__(
        self,
        semg_input_dim: int = 8,
        pressure_input_dim: int = 5,
        hidden_dim: int = 64,
        num_gru_layers: int = 1,
        num_gestures: int = 14,
        dropout: float = 0.1,
        mlp_hidden: int = 128,
        time_point: int = 9,
        n_t: int = 8,
        n_s: int = 16,
    ):
        super().__init__()
        _ = (pressure_input_dim, hidden_dim, num_gru_layers, mlp_hidden)
        self.num_gestures = int(num_gestures)
        self.semg_trunk = _EMGNetTrunk(
            channel=semg_input_dim,
            drop_out=dropout,
            time_point=time_point,
            n_t=n_t,
            n_s=n_s,
        )
        self.fc1 = nn.ModuleDict(
            {
                str(d): nn.Linear(d, self.num_gestures)
                for d in _probe_flat_dims(self.semg_trunk, semg_input_dim)
            }
        )

    def forward(self, semg: torch.Tensor, pressure: torch.Tensor | None = None) -> torch.Tensor:
        _ = pressure
        feat = self.semg_trunk.forward_flat(semg)
        key = str(int(feat.shape[1]))
        if key not in self.fc1:
            self.fc1[key] = nn.Linear(int(feat.shape[1]), self.num_gestures).to(
                device=feat.device, dtype=feat.dtype
            )
        return self.fc1[key](feat)


class FeatFusion_sEMG(FeatureLevelFusionSEMG):
    """仅 sEMG 分支。"""


class FeatureLevelFusionPress(nn.Module):
    """FeatFusion 的 glove-only 消融（mode0=flexion，mode1=pressure）。"""

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
        _ = semg_input_dim
        self.press_enc = _BiGRUEncoder(
            pressure_input_dim, hidden_dim=hidden_dim, num_layers=num_gru_layers, dropout=dropout
        )
        self.classifier = _make_classifier(
            self.press_enc.out_dim, mlp_hidden, num_gestures, dropout
        )

    def forward(self, semg: torch.Tensor | None, pressure: torch.Tensor) -> torch.Tensor:
        _ = semg
        return self.classifier(self.press_enc(pressure))


class FeatFusion_press(FeatureLevelFusionPress):
    """仅 glove 分支（mode0=flexion，mode1=pressure）。"""


__all__ = [
    "FeatFusion",
    "FeatFusion_sEMG",
    "FeatFusion_press",
    "FeatureLevelFusion",
    "FeatureLevelFusionSEMG",
    "FeatureLevelFusionPress",
]
