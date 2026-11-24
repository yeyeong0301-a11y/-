import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation import feature_ranges, training_hashes
from utils.model_loader import ROOT
from utils.preprocessing import numeric_features, read_csv


def split(frame, grouped):
    if not grouped:
        return train_test_split(
            np.arange(len(frame)), test_size=.25, random_state=42, stratify=frame["_label"]
        )
    splitter = GroupShuffleSplit(n_splits=128, test_size=.25, random_state=42)
    for train, test in splitter.split(frame, groups=frame["_group"]):
        if frame.iloc[train]["_label"].nunique() == frame.iloc[test]["_label"].nunique() == 2:
            return train, test
    raise ValueError("정상과 이상이 모두 포함되는 그룹 분할을 찾지 못했습니다. 독립 측정 파일을 추가해 주세요.")


def train(args, extra_metadata=None):
    features = args.features
    units = args.units
    if len(features) != len(units) or len(set(features)) != len(features):
        raise ValueError("특징마다 학습 단위를 한 개씩 지정하고 중복 이름을 제거해 주세요.")
    if set(features) & {args.label, args.group, "_label", "_group"}:
        raise ValueError("라벨·측정 그룹·예약 열 이름을 특징으로 사용할 수 없습니다.")
    if args.manual_frequency and (len(features) != 1 or units[0] not in ("Hz", "kHz")):
        raise ValueError("수동 모드에는 단일 주파수 특징과 Hz 또는 kHz 단위가 필요합니다.")
    if args.normal == args.anomaly:
        raise ValueError("정상 라벨과 이상 라벨을 다른 값으로 설정해 주세요.")
    raw = read_csv(args.csv.read_bytes())
    labels = raw[args.label].str.strip()
    if set(labels) != {args.normal, args.anomaly}:
        raise ValueError("라벨 열에 결측 없이 지정한 정상·이상 라벨 두 종류가 있어야 합니다.")
    frame = numeric_features(raw, features)
    frame["_label"] = labels.map({args.normal: 0, args.anomaly: 1})
    if args.group:
        groups = raw[args.group].str.strip()
        if groups.isna().any() or groups.eq("").any():
            raise ValueError("측정 그룹 열에 빠진 값이 있습니다.")
        frame["_group"] = groups
    frame = frame.drop_duplicates()
    if frame.duplicated(subset=features).any():
        raise ValueError("동일 특징값이 서로 다른 라벨 또는 측정 그룹에 있습니다. 중복 기록을 확인해 주세요.")
    train_idx, test_idx = split(frame, bool(args.group))
    train_rows, test_rows = frame.iloc[train_idx], frame.iloc[test_idx]
    if train_rows["_label"].nunique() != 2 or test_rows["_label"].nunique() != 2:
        raise ValueError("학습·테스트 각각에 정상과 이상이 필요합니다.")
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(train_rows[features], train_rows["_label"])
    actual = test_rows["_label"]
    prediction = model.predict(test_rows[features])
    precision, recall, f1, _ = precision_recall_fscore_support(actual, prediction, average="binary", zero_division=0)
    print(f"학습 {len(train_rows)}행 · 테스트 {len(test_rows)}행")
    print(f"Accuracy {accuracy_score(actual, prediction):.4f} | Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f}")
    print("혼동행렬 (행=실제, 열=예측, 순서=정상·이상)")
    matrix = confusion_matrix(actual, prediction, labels=[0, 1])
    print(matrix)
    test_metrics = {
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "accuracy": round(float(accuracy_score(actual, prediction)), 4),
        "precision": round(float(precision), 4), "recall": round(float(recall), 4), "f1": round(float(f1), 4),
        "confusion": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
        "holdout_file": "holdout_labeled.csv",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "Random Forest",
        "model_version": uuid4().hex[:12],
        "normal_label": 0,
        "anomaly_label": 1,
        "features": [{"name": name, "unit": unit} for name, unit in zip(features, units, strict=True)],
        "manual_frequency_feature": features[0] if args.manual_frequency else None,
        "input_type": "dataframe",
        "external_preprocessor": False,
        "feature_ranges": feature_ranges(train_rows[features]),
        "train_row_hashes": training_hashes(train_rows[features]),
        "test_metrics": test_metrics,
        **(extra_metadata or {}),
    }
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        model_file = Path(temporary) / "rf_model.joblib"
        config_file = Path(temporary) / "model_metadata.json"
        joblib.dump(model, model_file, compress=3)
        metadata["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
        config_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        holdout_file = Path(temporary) / "holdout_labeled.csv"
        raw.loc[test_rows.index].to_csv(holdout_file, index=False, encoding="utf-8-sig")
        model_file.replace(args.output / model_file.name)
        config_file.replace(args.output / config_file.name)
        holdout_file.replace(args.output / holdout_file.name)
    print("테스트 행은 holdout_labeled.csv로 저장했습니다. 웹의 ‘판정 검증’에 올리면 같은 결과를 확인할 수 있습니다.")
    print(f"모델과 메타데이터 저장 완료 · {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="라벨된 배관 특징값 CSV로 Random Forest 학습")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--label", default="label")
    parser.add_argument("--normal", default="정상")
    parser.add_argument("--anomaly", default="이상")
    split_mode = parser.add_mutually_exclusive_group(required=True)
    split_mode.add_argument("--group", help="동일한 측정 파일 또는 세션을 식별하는 열")
    split_mode.add_argument("--independent-rows", action="store_true", help="각 행이 독립 측정인 경우")
    parser.add_argument("--manual-frequency", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    arguments = parser.parse_args()
    try:
        train(arguments)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"학습 실패 · {exc}\n")
