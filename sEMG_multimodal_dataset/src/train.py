import copy
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _flatten_labels(y):
    y = y.contiguous().clone()
    return y.view(-1).long()


def train_model(model, train_semg, train_pressure, train_labels, epochs, batch_size, lr, val_semg=None, val_pressure=None, val_labels=None, wd=1e-4):
    device = _device()
    model.to(device)
    ts, tp, tl = train_semg.contiguous().clone(), train_pressure.contiguous().clone(), _flatten_labels(train_labels)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    tr_ld = DataLoader(TensorDataset(ts, tp, tl), batch_size=batch_size, shuffle=True)
    va_ld = None
    if val_semg is not None:
        vs, vp, vl = val_semg.contiguous().clone(), val_pressure.contiguous().clone(), _flatten_labels(val_labels)
        va_ld = DataLoader(TensorDataset(vs, vp, vl), batch_size=batch_size, shuffle=False)
    best, state = (-1.0, None) if va_ld else (0.0, None)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        ok = tot = 0
        for sb, pb, lb in tr_ld:
            sb, pb, lb = sb.to(device), pb.to(device), lb.to(device)
            opt.zero_grad()
            logits = model(sb, pb)
            ce(logits, lb).backward()
            opt.step()
            ok += (logits.argmax(1) == lb).sum().item()
            tot += lb.size(0)
        tr_acc = ok / max(1, tot)
        sc = tr_acc
        if va_ld:
            model.eval()
            ok = tot = 0
            with torch.no_grad():
                for sb, pb, lb in va_ld:
                    sb, pb, lb = sb.to(device), pb.to(device), lb.to(device)
                    ok += (model(sb, pb).argmax(1) == lb).sum().item()
                    tot += lb.size(0)
            sc = ok / max(1, tot)
        if va_ld:
            if sc > best:
                best, state = sc, copy.deepcopy(model.state_dict())
        elif tr_acc > best:
            best, state = tr_acc, copy.deepcopy(model.state_dict())
    if state:
        model.load_state_dict(state)
    print(f"train done {time.time() - t0:.0f}s")
    return model


def _print_acc(acc: float, acc_prefix: Optional[str]) -> None:
    if acc_prefix:
        print(f"{acc_prefix} acc={acc:.4f}")
    else:
        print(f"acc={acc:.4f}")


def test_model(model, test_semg, test_pressure, test_labels, batch_size, acc_prefix: Optional[str] = None):
    device = _device()
    model.to(device).eval()
    ld = DataLoader(TensorDataset(test_semg, test_pressure, _flatten_labels(test_labels)), batch_size=batch_size, shuffle=False)
    ok = tot = 0
    with torch.no_grad():
        for sb, pb, lb in ld:
            sb, pb, lb = sb.to(device), pb.to(device), lb.to(device)
            ok += (model(sb, pb).argmax(1) == lb).sum().item()
            tot += lb.size(0)
    acc = ok / max(1, tot)
    _print_acc(acc, acc_prefix)
    return acc
