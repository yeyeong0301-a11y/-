"""노트북의 17단계 OCSVM 학습을 특징값 CSV와 기존 폴드 표로 재현."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from utils.capstone_ocsvm import transform_features
from utils.model_loader import ROOT
from utils.preprocessing import read_csv

CANDIDATES = ("v1_raw28", "v2_core24", "v2_plus28")


def fit(frame, roles, columns):
    if frame["path"].isna().any() or frame["path"].duplicated().any():
        raise ValueError("특징 CSV의 path는 고유해야 합니다.")
    if roles["path"].isna().any() or roles["path"].duplicated().any():
        raise ValueError("폴드 CSV의 path는 고유해야 합니다.")
    assigned = frame["path"].map(roles.set_index("path")["role"])
    if not assigned.isin(["train", "validation", "test"]).all():
        raise ValueError("모든 특징 행을 폴드의 train, validation 또는 test에 매핑해야 합니다.")
    if frame["repeat"].isna().any():
        raise ValueError("repeat 값이 누락되었습니다.")
    if assigned.groupby(frame["repeat"]).nunique().gt(1).any():
        raise ValueError("같은 반복 측정이 여러 분할에 포함됩니다.")
    train = assigned.eq("train").to_numpy()
    if not train.any():
        raise ValueError("학습 행이 없습니다.")
    values = transform_features(frame, columns)
    scaler = RobustScaler().fit(values[train])
    model = OneClassSVM(nu=.05, gamma="scale").fit(scaler.transform(values[train]).astype("float32"))
    return scaler, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("roles", type=Path)
    parser.add_argument("--candidate", required=True, choices=CANDIDATES)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "capstone_ocsvm")
    args = parser.parse_args()
    try:
        frame = read_csv(args.features.read_bytes())
        roles = read_csv(args.roles.read_bytes())
        candidate = args.candidate
        if candidate == "v1_raw28":
            columns = [name for name in frame if name.startswith("ch")]
        else:
            channels = range(4)
            bands = ("4-40", "40-100", "100-300", "300-800", "800-1600")
            columns = [f"ch{channel}_rms" for channel in channels]
            columns += sorted(f"ch{channel}_power_{band}" for channel in channels for band in bands)
            if candidate == "v2_plus28":
                columns += [f"ch{channel}_concentration" for channel in channels]
        expected_count = 24 if candidate == "v2_core24" else 28
        if len(columns) != expected_count:
            raise ValueError(f"{candidate}에는 특징 {expected_count}개가 필요합니다.")
        scaler, model = fit(frame, roles, columns)
        args.output.mkdir(parents=True, exist_ok=True)
        config = args.output / "config.json"
        metadata = json.loads(config.read_text(encoding="utf-8-sig")) if config.exists() else {"input_candidates": {}}
        candidates = metadata["input_candidates"]
        if candidate in candidates and candidates[candidate] != columns:
            raise ValueError("기존 config.json의 특징 순서와 다릅니다. 새 출력 폴더를 지정하세요.")
        prefix = f"{candidate}_fold{args.fold}"
        joblib.dump(scaler, args.output / f"{prefix}_scaler.joblib")
        joblib.dump(model, args.output / f"{prefix}_OCSVM.joblib")
        candidates[candidate] = columns
        config.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"학습 실패 · {exc}\n")
    print(f"탐색용 OCSVM · {prefix} 저장 완료 · {args.output}")


if __name__ == "__main__":
    main()
