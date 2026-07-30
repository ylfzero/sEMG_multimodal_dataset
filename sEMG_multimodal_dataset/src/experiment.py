from typing import List, Optional, Tuple

import torch

from data_manager import Data_Manager
from models.FeatFusion import FeatFusion, FeatFusion_press, FeatFusion_sEMG
from train import test_model, train_model

WS = (0.25, 0.5, 0.75)
MODELS = ("FeatFusion", "FeatFusion_sEMG", "FeatFusion_press")


def format_accuracy_context(
    *,
    exp: str,
    model: str,
    mode: int,
    num_gestures: int,
    subject_id: int,
    window_s: float,
    day: Optional[int] = None,
    sub_num: Optional[int] = None,
    single_day_split_seed: Optional[int] = None,
    cross_sub_drop_ratio: float = 0.0,
) -> str:
    gesture = "Static_gesture" if mode == 0 else "Gesture_free"
    parts = [
        f"exp={exp}",
        f"model={model}",
        f"gesture_mode={gesture}",
        f"mode={mode}",
        f"num_classes={num_gestures}",
        f"window_s={window_s}",
    ]
    if exp == "single_day":
        parts.append(f"subject={subject_id}")
        if day is not None:
            parts.append(f"day={day}")
        if single_day_split_seed is not None:
            parts.append(f"split_721_seed={single_day_split_seed}")
    elif exp == "cross_day":
        parts.append(f"subject={subject_id}")
        parts.append("train_day=1")
        parts.append("test_day=2")
    else:
        parts.append(f"held_out_subject={subject_id}")
        if sub_num is not None:
            parts.append(f"total_subjects={sub_num}")
        if cross_sub_drop_ratio > 0:
            parts.append(f"train_drop_ratio={cross_sub_drop_ratio}")
    return "[" + "][".join(parts) + "]"


def split_721(tr_s, tr_p, tr_y, te_s, te_p, te_y, seed: Optional[int]) -> Tuple[torch.Tensor, ...]:
    s = torch.cat([tr_s, te_s], 0)
    p = torch.cat([tr_p, te_p], 0)
    y = torch.cat([tr_y, te_y], 0)
    n = y.shape[0]
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    a, b = n * 7 // 10, n * 9 // 10

    def take(ix):
        return s[ix].clone(), p[ix].clone(), y[ix].clone()

    return (*take(perm[:a]), *take(perm[a:b]), *take(perm[b:]))


def drop_train(ratio: float, seed: Optional[int], semg, press, labels):
    if ratio <= 0:
        return semg, press, labels
    n = labels.shape[0]
    k = max(1, min(n, int(round(n * (1 - ratio)))))
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    g = torch.Generator()
    g.manual_seed(seed)
    ix = torch.randperm(n, generator=g)[:k]
    return semg[ix], press[ix], labels[ix]


class Experiment:
    def __init__(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        lr: float,
        single_day_split_seed=None,
        cross_sub_drop_ratio=0.0,
        cross_sub_drop_seed=None,
    ):
        if model_name not in MODELS:
            raise ValueError(f"不支持的模型: {model_name}，可选 {MODELS}")
        self.m = model_name
        self.ep = epochs
        self.bs = batch_size
        self.lr = lr
        self.sd_seed = single_day_split_seed
        self.dr = cross_sub_drop_ratio
        self.ds = cross_sub_drop_seed

    def _net(self, ng: int):
        common = dict(
            semg_input_dim=8,
            pressure_input_dim=5,
            hidden_dim=64,
            num_gru_layers=1,
            num_gestures=ng,
            dropout=0.1,
            mlp_hidden=128,
        )
        if self.m == "FeatFusion":
            return FeatFusion(**common, time_point=9, n_t=8, n_s=16)
        if self.m == "FeatFusion_sEMG":
            return FeatFusion_sEMG(**common, time_point=9, n_t=8, n_s=16)
        if self.m == "FeatFusion_press":
            return FeatFusion_press(**common)
        raise ValueError(self.m)

    def _train_test(self, ng: int, tr_s, tr_p, tr_y, te_s, te_p, te_y, acc_pf, val_s=None, val_p=None, val_y=None):
        m = train_model(self._net(ng), tr_s, tr_p, tr_y, self.ep, self.bs, self.lr, val_s, val_p, val_y)
        test_model(m, te_s, te_p, te_y, self.bs, acc_prefix=acc_pf)

    def run_single_day(self, mode: int, ng: int, subs: List[int]) -> None:
        for sid in subs:
            for day in (1, 2):
                for w in WS:
                    dm = Data_Manager(sid, day, mode, ng, w)
                    a, b, c, d, e, f = dm.get_single_data()
                    acc_pf = format_accuracy_context(
                        exp="single_day",
                        model=self.m,
                        mode=mode,
                        num_gestures=ng,
                        subject_id=sid,
                        window_s=w,
                        day=day,
                        single_day_split_seed=self.sd_seed,
                    )
                    tr_s, tr_p, tr_y, va_s, va_p, va_y, te_s, te_p, te_y = split_721(a, b, c, d, e, f, self.sd_seed)
                    self._train_test(ng, tr_s, tr_p, tr_y, te_s, te_p, te_y, acc_pf, va_s, va_p, va_y)

    def run_cross_day(self, mode: int, ng: int, subs: List[int]) -> None:
        for sid in subs:
            for w in WS:
                dm = Data_Manager(sid, 1, mode, ng, w)
                tr_s, tr_p, tr_y, te_s, te_p, te_y = dm.get_cross_data()
                acc_pf = format_accuracy_context(
                    exp="cross_day",
                    model=self.m,
                    mode=mode,
                    num_gestures=ng,
                    subject_id=sid,
                    window_s=w,
                )
                self._train_test(ng, tr_s, tr_p, tr_y, te_s, te_p, te_y, acc_pf)

    def run_cross_subject(self, mode: int, ng: int, subs: List[int], sub_num: int) -> None:
        for sid in subs:
            for w in WS:
                dm = Data_Manager(sid, 1, mode, ng, w)
                tr_s, tr_p, tr_y, te_s, te_p, te_y = dm.get_cross_subject_data(sub_num)
                acc_pf = format_accuracy_context(
                    exp="cross_subject",
                    model=self.m,
                    mode=mode,
                    num_gestures=ng,
                    subject_id=sid,
                    window_s=w,
                    sub_num=sub_num,
                    cross_sub_drop_ratio=self.dr,
                )
                tr_s, tr_p, tr_y = drop_train(self.dr, self.ds, tr_s, tr_p, tr_y)
                self._train_test(ng, tr_s, tr_p, tr_y, te_s, te_p, te_y, acc_pf)
