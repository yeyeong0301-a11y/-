from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import numeric_features

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Model:
    estimator: object
    preprocessor: object
    features: dict
    name: str
    version: str
    normal: object
    anomaly: object
    manual: bool
    array_input: bool
    score_label: str
    ranges: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    train_hashes: list = field(default_factory=list)
    synthetic: bool = False

    def predict(self, frame):
        values = numeric_features(frame, list(self.features))
        model_input = values.to_numpy() if self.array_input else values
        if self.preprocessor is not None:
            model_input = self.preprocessor.transform(model_input)
        labels = np.asarray(self.estimator.predict(model_input))
        if labels.shape != (len(values),) or not np.isin(labels, [self.normal, self.anomaly]).all():
            raise ValueError("모델 출력이 설정된 정상·이상 라벨과 다릅니다.")
        probabilities = np.full(len(values), np.nan)
        if hasattr(self.estimator, "predict_proba"):
            index = list(self.estimator.classes_).index(self.anomaly)
            probabilities = np.asarray(self.estimator.predict_proba(model_input))[:, index]
            if not (np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all():
                raise ValueError("모델이 유효한 이상 클래스 예측 확률을 반환하지 않았습니다.")
        scores = np.full(len(values), np.nan)
        if self.score_label and hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(model_input)).reshape(-1)
        return pd.DataFrame({
            "판정": np.where(labels == self.anomaly, "이상", "정상"),
            "이상 클래스 예측 확률": probabilities,
            "결정 점수": scores,
            "모델 출력 라벨": labels,
        })


def load_model():
    model_dir = ROOT / "models"
    model_path = model_dir / "rf_model.joblib"
    config_path = model_dir / "model_metadata.json"
    if not model_path.is_file():
        return None, "models 폴더에 rf_model.joblib 파일을 넣어 주세요."
    if not config_path.is_file():
        return None, "학습 당시 입력 순서와 단위를 담은 model_metadata.json이 필요합니다."
    try:
        meta = json.loads(config_path.read_text(encoding="utf-8-sig"))
        feature_list = meta["features"]
        features = {item["name"]: item["unit"] for item in feature_list}
        if not features or len(features) != len(feature_list):
            raise ValueError("특징 이름은 한 개 이상이며 서로 달라야 합니다.")
        if any(not isinstance(unit, str) or not unit.strip() for unit in features.values()):
            raise ValueError("모든 특징에 학습 단위를 지정해 주세요. 단위가 없으면 ‘무차원’을 사용하세요.")
        normal, anomaly = meta["normal_label"], meta["anomaly_label"]
        if normal == anomaly:
            raise ValueError("정상 라벨과 이상 라벨이 같을 수 없습니다.")
        expected_hash = meta.get("model_sha256")
        if expected_hash:
            with model_path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError("모델 파일과 메타데이터 버전이 다릅니다.")
                stream.seek(0)
                estimator = joblib.load(stream)
        else:
            estimator = joblib.load(model_path)
        preprocessor = joblib.load(model_dir / "preprocessor.joblib") if meta.get("external_preprocessor") else None
        first = estimator if preprocessor is None else preprocessor
        expected = getattr(first, "feature_names_in_", None)
        if expected is not None and list(expected) != list(features):
            raise ValueError("메타데이터 특징 순서와 모델의 학습 순서가 다릅니다.")
        if getattr(first, "n_features_in_", len(features)) != len(features):
            raise ValueError("메타데이터 특징 수와 학습 입력 수가 다릅니다.")
        if not callable(getattr(estimator, "predict", None)):
            raise ValueError("모델의 predict 메서드를 찾을 수 없습니다.")
        classes = getattr(estimator, "classes_", None)
        if classes is not None and set(classes) != {normal, anomaly}:
            raise ValueError("메타데이터의 정상·이상 라벨과 모델 클래스가 다릅니다.")
        mode = meta.get("input_type", "dataframe")
        if mode not in ("array", "dataframe") or (mode == "array" and expected is not None):
            raise ValueError("메타데이터 입력 형식을 학습 과정과 동일하게 설정해 주세요.")
        manual = (
            len(features) == 1
            and meta.get("manual_frequency_feature") == next(iter(features))
            and next(iter(features.values())) in ("Hz", "kHz")
        )
        ranges = meta.get("feature_ranges") or {}
        required = {"min", "p01", "p99", "max"}
        if not isinstance(ranges, dict) or any(
            name not in features or not isinstance(item, dict) or not required <= set(item)
            for name, item in ranges.items()
        ):
            raise ValueError("feature_ranges는 특징 이름별 min·p01·p99·max를 담아야 합니다.")
        return Model(estimator, preprocessor, features, meta["model_name"], meta["model_version"],
                     normal, anomaly, manual, mode == "array", meta.get("decision_score_label", ""),
                     ranges, meta.get("test_metrics") or {}, list(meta.get("train_row_hashes") or []),
                     bool(meta.get("synthetic_demo"))), ""
    except Exception as exc:
        return None, f"모델 연결 실패 · {exc}"
