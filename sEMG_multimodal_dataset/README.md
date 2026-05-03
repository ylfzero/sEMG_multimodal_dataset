
10.5281/zenodo.20004279
## Data layout (same level as the unpacked archive)

- `Static_gesture/Subject_XX/Day_X/raw.dat` → `--mode 0` (default 14 classes)
- `Gesture_free/Subject_XX/Day_X/raw.dat` → `--mode 1` (often 6 or 12 classes via `--num_gestures`)

Code and dependency notes live in this folder; model code is under `models/` (`FeatFusion.py`, `MSCNN.py`, `SVM.py`). Experiments do not write checkpoints to disk.

## Environment

```bash
cd Submit
pip install -r requirements.txt
```

Experiment entry point and loaders live under `src/` (`main.py`, `experiment.py`, `data_manger.py`, `train.py`). Run from the repository root so `models/` and dataset folders resolve correctly.

## Example commands

```bash
# Single-day + FeatFusion + static gesture + subjects 1–3
python src/main.py --exp single_day --model FeatFusion --mode 0 --subjects 1-3

# Cross-day + MSCNN + gesture-free + 12 classes
python src/main.py --exp cross_day --model MSCNN --mode 1 --num_gestures 12

# Cross-subject + SVM + drop 80% of training samples at random
python src/main.py --exp cross_subject --model SVM --mode 1 --num_gestures 6 --cross_sub_drop_ratio 0.8
```

Common flags: `--epochs`, `--batch_size`, `--lr`, `--sub_num` (total subjects for cross-subject), `--single_day_split_seed`.

