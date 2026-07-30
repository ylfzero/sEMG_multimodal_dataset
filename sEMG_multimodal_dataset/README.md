# sEMG Multimodal Gesture Dataset

Multimodal sEMG + glove dataset with reference experiment code (FeatFusion / FeatFusion_sEMG / FeatFusion_press).

## Data layout
The Static_gesture and Gesture_free folders are located in https://doi.org/10.5281/zenodo.21704999

Each recording is stored as `Subject_X/Day_Y/raw.dat`, where **X** is the participant ID (`01`–`30`) and **Y** is the recording day (`1` or `2`):

- `data/Static_gesture/Subject_XX/Day_Y/raw.dat` → `--mode 0` static gesture (default 14 classes; glove channels 5–9 = flexion)
- `data/Gesture_free/Subject_XX/Day_Y/raw.dat` → `--mode 1` gesture-free (often 6 or 12 classes via `--num_gestures`; glove channels 0–4 = pressure)

30 participants × 2 days × 2 gesture modes = **120** `raw.dat` files (~11.6 GB total).

### Binary format (`raw.dat`)

7×`int32` header: `[header_len, file_version, data_type, sample_rate, emg_chs, acc_chs, glove_chs]`.

- IMU is **not** included: `acc_chs = 0`
- Per-frame layout (float32/float64): `[ EMG (8) | GLOVE (10) | 2 trailing cols ]`
- The **last column** is the gesture trigger label

## Models

| Model | Description |
|-------|-------------|
| `FeatFusion` | sEMG (EMGNet) + glove (BiGRU) feature-level fusion |
| `FeatFusion_sEMG` | sEMG-only ablation |
| `FeatFusion_press` | glove-only ablation (mode 0 = flexion, mode 1 = pressure) |

## Quick start

```bash
cd sEMG_multimodal_dataset
pip install -r requirements.txt
python src/main.py --help
```

Run all commands from the **`sEMG_multimodal_dataset`** folder (the folder that contains `data/`, `models/`, and `src/`). Experiments print accuracy to stdout and do not write checkpoints.

## Example commands

```bash
# Single-day + full fusion + static gesture + subjects 1–3
python src/main.py --exp single_day --model FeatFusion --mode 0 --subjects 1-3 --epochs 2

# Cross-day + sEMG-only + gesture-free + 12 classes
python src/main.py --exp cross_day --model FeatFusion_sEMG --mode 1 --num_gestures 12 --epochs 2

# Cross-subject + glove-only + drop 80% of training samples at random
python src/main.py --exp cross_subject --model FeatFusion_press --mode 1 --num_gestures 6 --sub_num 30 --cross_sub_drop_ratio 0.8 --epochs 2
```

Common flags: `--epochs` (default 50), `--batch_size` (default 256), `--lr` (default 1e-4), `--sub_num` (default 30), `--single_day_split_seed`.

## Package contents

```
sEMG_multimodal_dataset/
├── data/
│   ├── Static_gesture/Subject_XX/Day_Y/raw.dat
│   └── Gesture_free/Subject_XX/Day_Y/raw.dat
├── models/FeatFusion.py
├── src/
│   ├── main.py          # CLI entry
│   ├── experiment.py
│   ├── data_manager.py
│   └── train.py
├── requirements.txt
└── README.md
```

When publishing, upload this folder as-is; exclude `.vscode/` and `__pycache__/` (see `.gitignore`).
