from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PKG_ROOT))

from experiment import Experiment, MODELS  # noqa: E402


def parse_subjects(s: str) -> list[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gesture classification experiments (FeatFusion / FeatFusion_sEMG / FeatFusion_press)"
    )
    ap.add_argument("--exp", choices=["single_day", "cross_day", "cross_subject"], required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--mode", type=int, choices=[0, 1], required=True, help="0=Static_gesture (flexion), 1=Gesture_free (pressure)")
    ap.add_argument("--num_gestures", type=int, default=None, help="Defaults: mode 0 -> 14, mode 1 -> 12")
    ap.add_argument("--subjects", type=str, default="1-30", help="e.g. 1-30 or 1,2,5")
    ap.add_argument("--sub_num", type=int, default=30, help="Cross-subject: total number of subjects in the dataset")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cross_sub_drop_ratio", type=float, default=0.0, help="Cross-subject: randomly drop this fraction of training samples")
    ap.add_argument("--cross_sub_drop_seed", type=int, default=None)
    ap.add_argument("--single_day_split_seed", type=int, default=None, help="Single-day: RNG seed for 7:2:1 train/val/test split")
    args = ap.parse_args()

    num_gestures = args.num_gestures if args.num_gestures is not None else (14 if args.mode == 0 else 12)
    sub_list = parse_subjects(args.subjects)
    runner = Experiment(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        single_day_split_seed=args.single_day_split_seed,
        cross_sub_drop_ratio=args.cross_sub_drop_ratio,
        cross_sub_drop_seed=args.cross_sub_drop_seed,
    )

    if args.exp == "single_day":
        runner.run_single_day(args.mode, num_gestures, sub_list)
    elif args.exp == "cross_day":
        runner.run_cross_day(args.mode, num_gestures, sub_list)
    else:
        runner.run_cross_subject(args.mode, num_gestures, sub_list, args.sub_num)


if __name__ == "__main__":
    main()
