from typing import List, Optional, Tuple

import torch

from data_manager import Data_Manager
from models.FeatFusion import FeatFusion
from models.MSCNN import MSCNN
from train import run_svm, semg_ntc, test_model, test_one_modal, train_model, train_one_modal

WS = (0.25, 0.5, 0.75)

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


def drop_train(ratio: float, seed: Optional[int], semg, labels, press=None):
    if ratio <= 0:
        return (semg, press, labels) if press is not None else (semg, labels)
    n = labels.shape[0]
    k = max(1, min(n, int(round(n * (1 - ratio)))))
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    g = torch.Generator()
    g.manual_seed(seed)
    ix = torch.randperm(n, generator=g)[:k]
    if press is None:
        return semg[ix], labels[ix]
    return semg[ix], press[ix], labels[ix]


class Experiment:
    def __init__(self, model_name: str, epochs: int, batch_size: int, lr: float, single_day_split_seed=None, cross_sub_drop_ratio=0.0, cross_sub_drop_seed=None):
        self.m = model_name
        self.ep = epochs
        self.bs = batch_size
        self.lr = lr
        self.sd_seed = single_day_split_seed
        self.dr = cross_sub_drop_ratio
        self.ds = cross_sub_drop_seed

    def _net(self, ng: int):
        if self.m == "FeatFusion":
            return FeatFusion(8, 5, 64, 1, ng, 0.1, 128)
        if self.m == "MSCNN":
            return MSCNN(ng, 300, 8)
        raise ValueError(self.m)

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
                        single_day_split_seed=self.sd_seed if self.m != "SVM" else None,
                    )
                    if self.m == "SVM":
                        run_svm(ng, a, c, d, f, acc_prefix=acc_pf)
                        continue
                    u = split_721(a, b, c, d, e, f, self.sd_seed)
                    tr_s, tr_p, tr_y, va_s, va_p, va_y, te_s, te_p, te_y = u
                    if self.m == "FeatFusion":
                        m = train_model(self._net(ng), tr_s, tr_p, tr_y, self.ep, self.bs, self.lr, va_s, va_p, va_y)
                        test_model(m, te_s, te_p, te_y, self.bs, acc_prefix=acc_pf)
                    else:
                        m = train_one_modal(self._net(ng), semg_ntc(tr_s), tr_y, self.ep, self.bs, self.lr, 0.1)
                        test_one_modal(m, semg_ntc(te_s), te_y, self.bs, acc_prefix=acc_pf)

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
                if self.m == "SVM":
                    run_svm(ng, tr_s, tr_y, te_s, te_y, acc_prefix=acc_pf)
                    continue
                if self.m == "FeatFusion":
                    m = train_model(self._net(ng), tr_s, tr_p, tr_y, self.ep, self.bs, self.lr)
                    test_model(m, te_s, te_p, te_y, self.bs, acc_prefix=acc_pf)
                else:
                    m = train_one_modal(self._net(ng), semg_ntc(tr_s), tr_y, self.ep, self.bs, self.lr, 0.0)
                    test_one_modal(m, semg_ntc(te_s), te_y, self.bs, acc_prefix=acc_pf)

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
                if self.m == "SVM":
                    tr_s, tr_y = drop_train(self.dr, self.ds, tr_s, tr_y)
                    run_svm(ng, tr_s, tr_y, te_s, te_y, acc_prefix=acc_pf)
                    continue
                tr_s, tr_p, tr_y = drop_train(self.dr, self.ds, tr_s, tr_y, tr_p)
                if self.m == "FeatFusion":
                    m = train_model(self._net(ng), tr_s, tr_p, tr_y, self.ep, self.bs, self.lr)
                    test_model(m, te_s, te_p, te_y, self.bs, acc_prefix=acc_pf)
                else:
                    m = train_one_modal(self._net(ng), semg_ntc(tr_s), tr_y, self.ep, self.bs, self.lr, 0.0)
                    test_one_modal(m, semg_ntc(te_s), te_y, self.bs, acc_prefix=acc_pf)
