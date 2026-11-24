"""화면 확인용 합성 데모: 가짜 특징값 CSV를 만들고 데모 RF를 학습한다.

실제 배관 측정값이 아니다. 판정 검증·시각화 화면이 동작하는지 볼 때만 쓴다.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_model import train
from utils.model_loader import ROOT

FEATURES = ("median_rms", "median_power_4_40", "median_power_40_100",
            "median_power_100_300", "median_power_300_800", "median_power_800_1600")
# sample_event.json의 값을 정상 중심으로 쓴다 (log10 값)
NORMAL = np.array([-1.99, -5.32, -4.07, -5.48, -5.75, -5.74])
# 이상 갈래별로 오르는 대역 (합성 규칙)
SHIFTS = {
    "누수 후보": np.array([0.25, 0.0, 0.1, 0.6, 0.5, 0.2]),
    "충격 후보": np.array([0.6, 0.7, 0.4, 0.2, 0.1, 0.0]),
    "균열 후보": np.array([0.2, 0.0, 0.0, 0.2, 0.5, 0.7]),
}


def sessions(rng, count, start, anomaly_share):
    rows = []
    kinds = list(SHIFTS)
    for number in range(start, start + count):
        anomaly = rng.random() < anomaly_share
        kind = kinds[number % len(kinds)] if anomaly else ""
        offset = rng.normal(0, .06, len(FEATURES))
        for _ in range(int(rng.integers(10, 18))):
            centre = NORMAL + offset
            if anomaly:
                centre = centre + SHIFTS[kind] * rng.uniform(.35, 1.1)
            values = centre + rng.normal(0, .12, len(FEATURES))
            rows.append({"source_file": f"SYNTHETIC_DEMO_{number:03d}", **dict(zip(FEATURES, values.round(5))),
                         "label": "이상" if anomaly else "정상", "demo_type": kind or "정상"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--force", action="store_true", help="models/에 있는 기존 모델을 덮어쓴다")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if (args.output / "rf_model.joblib").exists() and not args.force:
        parser.exit(1, "models/에 이미 모델이 있습니다. 덮어쓰려면 --force, 다른 곳에 두려면 --output을 쓰세요.\n")
    rng = np.random.default_rng(args.seed)
    args.data.mkdir(parents=True, exist_ok=True)

    training_rows = sessions(rng, 48, 1, .4)
    training_csv = args.data / "synthetic_train.csv"
    training_rows.to_csv(training_csv, index=False, encoding="utf-8-sig")

    check = sessions(rng, 12, 101, .45)
    # 학습 범위 밖 입력이 어떻게 표시되는지 보기 위한 행
    extreme = check.sample(4, random_state=args.seed).copy()
    extreme["median_power_100_300"] += 1.6
    extreme["source_file"] = "SYNTHETIC_DEMO_OUT_OF_RANGE"
    check = pd.concat([check, extreme], ignore_index=True)
    check.to_csv(args.data / "synthetic_check_labeled.csv", index=False, encoding="utf-8-sig")

    # log10이 아니라 원래 값(linear)으로 잘못 넣은 경우
    unit_error = check.head(30).copy()
    for name in FEATURES:
        unit_error[name] = (10 ** unit_error[name]).round(9)
    unit_error.to_csv(args.data / "synthetic_check_unit_error.csv", index=False, encoding="utf-8-sig")

    options = SimpleNamespace(
        csv=training_csv, features=list(FEATURES), units=["log10"] * len(FEATURES), label="label",
        normal="정상", anomaly="이상", group="source_file", independent_rows=False,
        manual_frequency=False, output=args.output,
    )
    train(options, {"model_name": "Random Forest (합성 데모)", "synthetic_demo": True})
    print(f"합성 데모 데이터 · {args.data}")
    print("웹에서 ‘판정 검증’에 synthetic_check_labeled.csv를 올려 보세요. 실제 배관 측정값이 아닙니다.")


if __name__ == "__main__":
    main()
