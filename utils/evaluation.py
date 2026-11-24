"""정답 라벨과 판정 비교, 임계값 검토, 학습 범위 점검."""
import hashlib
import math

import numpy as np
import pandas as pd

LABEL_GUESSES = (
    ("정상", "이상"),
    ("normal", "anomaly"),
    ("normal", "abnormal"),
    ("0", "1"),
    ("ok", "ng"),
)
SMALL_TOTAL, SMALL_CLASS = 30, 5


def guess_labels(values):
    """라벨 열의 고유값에서 정상·이상 표기를 추정한다. 못 찾으면 (None, None)."""
    lowered = {str(value).strip().lower(): value for value in values}
    for normal, anomaly in LABEL_GUESSES:
        if normal in lowered and anomaly in lowered:
            return lowered[normal], lowered[anomaly]
    return None, None


def encode_labels(column, normal, anomaly):
    """정답 라벨 열을 0(정상)·1(이상)로 바꾼다. 비었거나 지정하지 않은 값이 있으면 오류."""
    original = column.astype("string").str.strip()
    blank = original.isna() | original.eq("")
    if blank.any():
        rows = (np.flatnonzero(blank.to_numpy())[:5] + 1).tolist()
        raise ValueError(f"정답 라벨이 비어 있는 행이 있습니다 (데이터 행 {rows})")
    # "Normal "과 "normal"처럼 공백·대소문자만 다른 표기는 같은 값으로 본다.
    text = original.str.casefold()
    mapping = {str(value).strip().casefold(): code for value, code in ((normal, 0), (anomaly, 1)) if value is not None}
    encoded = text.map(mapping)
    unknown = sorted(set(original[encoded.isna()]))
    if unknown:
        raise ValueError("정상·이상으로 지정하지 않은 라벨 값: " + ", ".join(unknown[:8]))
    return encoded.astype(int).to_numpy()


def input_issues(frame, features, actual):
    """검증 전에 알아야 할 입력 문제를 짧은 문장으로 돌려준다."""
    notes = []
    values = frame[features]
    constant = [name for name in features if values[name].nunique() <= 1]
    if constant and len(frame) > 1:
        notes.append("값이 변하지 않는 열: " + ", ".join(constant) + " · 센서 고정이나 열 연결을 확인하세요.")
    duplicated = int(values.duplicated().sum())
    if duplicated:
        labelled = values.assign(_actual=actual)
        conflict = int(labelled.groupby(features)["_actual"].nunique().gt(1).sum())
        text = f"특징값이 완전히 같은 중복 행 {duplicated:,}개"
        if conflict:
            text += f", 그중 같은 입력에 정답이 엇갈린 묶음 {conflict:,}개"
        notes.append(text + " · 같은 측정이 여러 번 들어갔는지 확인하세요.")
    return notes


def confusion(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)
    return {
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
    }


def metrics(actual, predicted):
    """이상(1)을 찾는 관점의 지표. 분모가 0이면 계산하지 않고 nan으로 둔다."""
    c = confusion(actual, predicted)
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]

    def part(numerator, denominator):
        return {"값": numerator / denominator if denominator else math.nan,
                "분자": numerator, "분모": denominator}

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        **c,
        "행 수": tp + fn + fp + tn,
        "실제 이상": tp + fn,
        "실제 정상": tn + fp,
        "재현율": part(tp, tp + fn),          # 실제 이상 중 이상으로 잡은 비율
        "정밀도": part(tp, tp + fp),          # 이상 판정 중 실제 이상 비율
        "오경보율": part(fp, fp + tn),        # 실제 정상 중 이상으로 잘못 판정한 비율
        "F1": part(2 * tp, 2 * tp + fp + fn) if tp + fn else part(0, 0),
        "MCC": (tp * tn - fp * fn) / denominator if denominator else math.nan,
    }


def sample_warnings(result):
    notes = []
    if result["실제 이상"] == 0:
        notes.append("실제 이상 행이 없어 재현율·F1·MCC를 계산하지 않습니다. 오경보만 확인할 수 있습니다.")
    if result["실제 정상"] == 0:
        notes.append("실제 정상 행이 없어 오경보율·MCC를 계산하지 않습니다. 탐지 여부만 확인할 수 있습니다.")
    if result["행 수"] < SMALL_TOTAL or 0 < min(result["실제 이상"], result["실제 정상"]) < SMALL_CLASS:
        notes.append(f"표본이 적습니다 (전체 {SMALL_TOTAL}행 미만이거나 한쪽 라벨이 {SMALL_CLASS}행 미만). "
                     "비율보다 건수를 보세요.")
    return notes


def anomaly_score(prediction):
    """클수록 이상에 가까운 점수, 기본 임계값, 점수 이름. 점수가 없으면 None."""
    probability = prediction["이상 클래스 예측 확률"].to_numpy(dtype=float)
    if np.isfinite(probability).all():
        return probability, 0.5, "이상 클래스 예측 확률"
    decision = prediction["결정 점수"].to_numpy(dtype=float)
    if np.isfinite(decision).all():
        return -decision, 0.0, "이상 점수 (결정 점수 부호 반전)"
    return None


def pr_curve(actual, score):
    """임계값을 점수의 고유값마다 두고 정밀도·재현율을 계산한다. 점수 >= 임계값이면 이상."""
    actual = np.asarray(actual, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = actual.sum()
    if positives == 0 or positives == len(actual):
        return None, math.nan
    order = np.argsort(-score, kind="mergesort")
    sorted_score, sorted_actual = score[order], actual[order]
    last = np.r_[np.flatnonzero(np.diff(sorted_score)), len(score) - 1]
    tp = np.cumsum(sorted_actual)[last]
    flagged = last + 1
    precision, recall = tp / flagged, tp / positives
    average_precision = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    curve = pd.DataFrame({"임계값": sorted_score[last], "정밀도": precision, "재현율": recall})
    return curve, average_precision


def row_hash(values):
    text = ",".join(format(float(value), ".10g") for value in values)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def training_hashes(frame):
    return sorted({row_hash(row) for row in frame.to_numpy(dtype=float)})


def overlap_with_training(inputs, hashes):
    """학습에 쓴 행과 특징값이 같은 행. 해시 목록이 없으면 None."""
    if not hashes:
        return None
    known = set(hashes)
    return pd.Series([row_hash(row) in known for row in inputs.to_numpy(dtype=float)], index=inputs.index)


def feature_ranges(frame):
    """학습 행(정상과 이상 모두)에서 특징별 분포 요약을 만든다."""
    values = frame.to_numpy(dtype=float)
    quantiles = np.quantile(values, [0, .01, .5, .99, 1], axis=0)
    keys = ("min", "p01", "p50", "p99", "max")
    return {name: dict(zip(keys, map(float, quantiles[:, index]))) for index, name in enumerate(frame)}


def range_check(inputs, ranges):
    """학습 최소·최대를 벗어난 특징 수(행별)와 특징별 요약. 범위 정보가 없으면 None."""
    names = [name for name in inputs if name in ranges]
    if not names:
        return None, None
    outside = pd.DataFrame({
        name: (inputs[name] < ranges[name]["min"]) | (inputs[name] > ranges[name]["max"]) for name in names
    })
    summary = pd.DataFrame([{
        "특징": name,
        "학습 최소": ranges[name]["min"], "학습 중앙값": ranges[name].get("p50", math.nan),
        "학습 최대": ranges[name]["max"], "입력 중앙값": float(inputs[name].median()),
        "범위 밖 행": int(outside[name].sum()), "확인할 점": unit_hint(inputs[name], ranges[name]),
    } for name in names])
    return outside.sum(axis=1), summary


def unit_hint(values, item):
    """입력 중앙값이 학습과 크게 다를 때 흔한 원인을 짧게 알려준다."""
    median, reference = float(values.median()), item.get("p50")
    if reference is None or not math.isfinite(median):
        return ""
    if item["max"] < 0 and values.min() > 0:
        return "학습은 음수(로그값), 입력은 양수 · log10 변환 여부 확인"
    if reference and median and np.sign(reference) == np.sign(median):
        ratio = abs(median / reference)
        if 300 <= ratio <= 3000:
            return "입력이 약 1000배 큼 · Hz/kHz 등 단위 확인"
        if 1 / 3000 <= ratio <= 1 / 300:
            return "입력이 약 1/1000 · Hz/kHz 등 단위 확인"
    if not item["min"] <= median <= item["max"]:
        return "입력 중앙값이 학습 범위 밖"
    return ""


def row_notes(count, outside=None, score=None, threshold=None, margin=None, overlap=None):
    """행마다 판정을 그대로 믿기 전에 볼 점. 문제가 없으면 빈 문자열."""
    notes = []
    for position in range(count):
        reasons = []
        if outside is not None and outside.iloc[position]:
            reasons.append(f"학습 범위 밖 특징 {int(outside.iloc[position])}개")
        if score is not None and margin and abs(score[position] - threshold) < margin:
            reasons.append("임계값 근처")
        if overlap is not None and overlap.iloc[position]:
            reasons.append("학습에 쓴 행과 같음")
        notes.append(" · ".join(reasons))
    return notes
