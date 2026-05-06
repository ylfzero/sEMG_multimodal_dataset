from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_ROOT = Path(__file__).resolve().parent.parent


def _raw_path(mode: int, sub_id: int, day_id: int) -> str:
    d = "Static_gesture" if mode == 0 else "Gesture_free"
    return str(_ROOT / d / f"Subject_{sub_id:02d}" / f"Day_{day_id}" / "raw.dat")


def _bandpass(ch, fs=500, low=20, high=150, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band", analog=False)
    return signal.filtfilt(b, a, ch)


def _filter_emg(emg):
    out = emg.copy()
    for i in range(out.shape[1]):
        out[:, i] = _bandpass(out[:, i])
    return out


class Data_Manager:
    def __init__(self, sub_id, day_id, mode, num_gestures, window_size):
        self.sub_id = sub_id
        self.day_id = day_id
        self.mode = mode
        self.num_gestures = num_gestures
        self.window_size = window_size

    def read_data_file(self, path):
        with open(path, "rb") as f:
            h = np.fromfile(f, dtype=np.int32, count=7)
            meta = {
                "dtype": np.float32 if h[2] == 2 else np.float64,
                "emg_chs": h[4],
                "acc_chs": h[5],
                "glove_chs": h[6],
            }
            fs = meta["emg_chs"] + meta["acc_chs"] + meta["glove_chs"] + 2
            raw = np.fromfile(f, dtype=meta["dtype"])
            n = len(raw) // fs
            return meta, raw[: n * fs].reshape(n, fs)

    def _modal(self, meta, data):
        e, a, g = meta["emg_chs"], meta["acc_chs"], meta["glove_chs"]
        semg = _filter_emg(data[:, :e])
        gs = e + a
        gl = data[:, gs : gs + g]
        press = gl[:, 0:5] if self.mode == 1 else gl[:, 5:10]
        trig = data[:, -1]
        m = trig < self.num_gestures
        return semg[m], press[m], trig[m]

    def get_single_data(self):
        meta, data = self.read_data_file(_raw_path(self.mode, self.sub_id, self.day_id))
        sw, pw, y = self.preprocess_data(*self._modal(meta, data))
        return self.divide_data(torch.tensor(sw, dtype=torch.float32), torch.tensor(pw, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    def get_cross_data(self):
        m1, d1 = self.read_data_file(_raw_path(self.mode, self.sub_id, 1))
        s1, p1, t1 = self._modal(m1, d1)
        w1s, w1p, y1, sc, pc = self.preprocess_data(
            s1, p1, t1, return_scaler=True, semg_scaler_mode="standard", semg_z_clip=3.0, pressure_z_clip=3.0
        )
        m2, d2 = self.read_data_file(_raw_path(self.mode, self.sub_id, 2))
        s2, p2, t2 = self._modal(m2, d2)
        w2s, w2p, y2, _, _ = self.preprocess_data(
            s2, p2, t2, semg_scaler=sc, pressure_scaler=pc, semg_z_clip=3.0, pressure_z_clip=3.0, return_scaler=True
        )
        return (
            torch.tensor(w1s, dtype=torch.float32),
            torch.tensor(w1p, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.long),
            torch.tensor(w2s, dtype=torch.float32),
            torch.tensor(w2p, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.long),
        )

    def get_cross_subject_data(self, sub_num):
        tr_s = tr_p = tr_y = te_s = te_p = te_y = None
        for sid in range(1, sub_num + 1):
            for did in (1, 2):
                meta, data = self.read_data_file(_raw_path(self.mode, sid, did))
                sw, pw, y = self.preprocess_data(*self._modal(meta, data))
                if sid != self.sub_id:
                    if tr_s is None:
                        tr_s, tr_p, tr_y = sw, pw, y
                    else:
                        tr_s = np.concatenate([tr_s, sw], 0)
                        tr_p = np.concatenate([tr_p, pw], 0)
                        tr_y = np.concatenate([tr_y, y], 0)
                elif te_s is None:
                    te_s, te_p, te_y = sw, pw, y
                else:
                    te_s = np.concatenate([te_s, sw], 0)
                    te_p = np.concatenate([te_p, pw], 0)
                    te_y = np.concatenate([te_y, y], 0)
        return (
            torch.as_tensor(tr_s, dtype=torch.float32),
            torch.as_tensor(tr_p, dtype=torch.float32),
            torch.as_tensor(tr_y, dtype=torch.long),
            torch.as_tensor(te_s, dtype=torch.float32),
            torch.as_tensor(te_p, dtype=torch.float32),
            torch.as_tensor(te_y, dtype=torch.long),
        )

    def divide_data(self, semg_t, press_t, y_t, train_ratio=0.8, random_seed=42):
        y = y_t.numpy()
        np.random.seed(random_seed)
        tri, tei = [], []
        for lab in np.unique(y):
            idx = np.where(y == lab)[0]
            np.random.shuffle(idx)
            k = int(round(len(idx) * train_ratio))
            tri.extend(idx[:k])
            tei.extend(idx[k:])
        np.random.shuffle(tri)
        np.random.shuffle(tei)
        return semg_t[tri], press_t[tri], y_t[tri], semg_t[tei], press_t[tei], y_t[tei]

    def preprocess_data(
        self,
        semg_data,
        pressure_data,
        gesture_labels,
        step_size=250,
        semg_scaler=None,
        pressure_scaler=None,
        semg_scaler_mode="minmax",
        semg_z_clip=3.0,
        pressure_z_clip=10.0,
        return_scaler=False,
    ):
        window_size = int(self.window_size * 1000)
        fit_s = semg_scaler is None
        fit_p = pressure_scaler is None
        if semg_scaler is None:
            semg_scaler = StandardScaler() if semg_scaler_mode.lower() == "standard" else MinMaxScaler()
        if pressure_scaler is None:
            pressure_scaler = StandardScaler()
        semg_n = semg_scaler.fit_transform(semg_data) if fit_s else semg_scaler.transform(semg_data)
        if semg_z_clip is not None and isinstance(semg_scaler, StandardScaler):
            semg_n = np.clip(semg_n, -semg_z_clip, semg_z_clip)
        press_n = pressure_scaler.fit_transform(pressure_data) if fit_p else pressure_scaler.transform(pressure_data)
        if pressure_z_clip is not None:
            press_n = np.clip(press_n, -pressure_z_clip, pressure_z_clip)
        sx, px, gy = [], [], []
        for lab in np.unique(gesture_labels):
            sel = gesture_labels == lab
            sg, pg = semg_n[sel], press_n[sel]
            if sg.shape[0] < window_size:
                continue
            nw = (sg.shape[0] - window_size) // step_size + 1
            for i in range(nw):
                a, b = i * step_size, i * step_size + window_size
                sx.append(sg[a:b])
                px.append(pg[a:b])
                gy.append(lab)
        sx, px, gy = np.array(sx), np.array(px), np.array(gy)
        if return_scaler:
            return sx, px, gy, semg_scaler, pressure_scaler
        return sx, px, gy
