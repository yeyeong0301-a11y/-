"""캡스톤 노트북의 v2 특징 변환 및 OCSVM 희소도 추론."""
import numpy as np

from utils.preprocessing import numeric_features


def transform_features(frame, columns):
    if len(columns) not in (24, 28) or len(set(columns)) != len(columns):
        raise ValueError("서로 다른 입력 특징 24개 또는 28개가 필요합니다.")
    values = numeric_features(frame, columns).to_numpy()
    logarithmic = [
        index for index, name in enumerate(columns)
        if not any(key in name for key in ("ratio4_40", "top100ms_fraction", "concentration"))
    ]
    if (values[:, logarithmic] <= 0).any():
        raise ValueError("로그 변환 대상 특징은 양수여야 합니다.")
    values[:, logarithmic] = np.log10(np.maximum(values[:, logarithmic], 1e-12))
    return values


def score(frame, columns, scaler, model):
    if any(getattr(obj, "n_features_in_", None) != len(columns) for obj in (scaler, model)):
        raise ValueError("특징 목록·스케일러·모델의 입력 차원이 일치해야 합니다.")
    values = transform_features(frame, columns)
    transformed = scaler.transform(values).astype("float32")
    scores = -model.score_samples(transformed)
    if not np.isfinite(scores).all():
        raise ValueError("모델이 유한한 희소도 점수를 반환하지 않았습니다.")
    return scores
