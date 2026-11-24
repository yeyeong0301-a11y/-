# PIPEGUARD AI 파일별 전체 코드 (이전 스냅숏)

> 이 문서는 이전 버전 코드를 글로 옮긴 기록입니다. 지금 실행되는 코드는 저장소의 `pages/`, `utils/`, `training/`, `tests/` 폴더와 `.streamlit/config.toml`에 있습니다. 내용이 다르면 폴더의 파일을 기준으로 하세요.

## app.py

```python
import streamlit as st

from utils.model_loader import ROOT, load_model


@st.cache_resource
def current_model(signature):
    return load_model()


def file_signature(path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


st.set_page_config(page_title="PIPEGUARD AI", page_icon=":material/monitoring:", layout="wide")
paths = [ROOT / "models" / name for name in ("rf_model.joblib", "model_metadata.json", "preprocessor.joblib")]
signature = tuple(file_signature(path) for path in paths)
st.session_state.model, st.session_state.model_detail = current_model(signature)

page = st.navigation([
    st.Page("pages/dashboard.py", title="대시보드", icon=":material/dashboard:", default=True),
    st.Page("pages/manual_diagnosis.py", title="수동 진단", icon=":material/sensors:"),
    st.Page("pages/csv_analysis.py", title="CSV 분석 및 신호", icon=":material/monitoring:"),
    st.Page("pages/capstone_ocsvm.py", title="캡스톤 OCSVM", icon=":material/analytics:"),
    st.Page("pages/history.py", title="진단 기록", icon=":material/history:"),
])
st.sidebar.divider()
st.sidebar.markdown("**PIPEGUARD AI**")
if st.session_state.model is None:
    st.sidebar.warning("AI 모델 미연결")
else:
    model = st.session_state.model
    st.sidebar.success(f"{model.name} · {model.version}")
st.title("PIPEGUARD AI — 지하배관 지능형 이상 진단 시스템")
page.run()
```

## pages/capstone_ocsvm.py

```python
import json

import joblib
import plotly.express as px
import streamlit as st

from utils.capstone_ocsvm import score
from utils.model_loader import ROOT
from utils.preprocessing import read_csv
from utils.visualization import chart

st.header("캡스톤 OCSVM 희소도 분석")
st.caption("라벨이 없는 혼합 데이터로 학습한 탐색 모델입니다. 점수가 클수록 모델 기준으로 드문 패턴입니다. 정상·이상 판정에는 검증된 임계값이 필요합니다.")
models = ROOT / "models" / "capstone_ocsvm"
candidate = st.selectbox("입력 특징 세트", ("v1_raw28", "v2_core24", "v2_plus28"))
fold = st.selectbox("학습 폴드", range(5))
prefix = f"{candidate}_fold{fold}"
paths = [models / name for name in ("config.json", f"{prefix}_scaler.joblib", f"{prefix}_OCSVM.joblib")]
missing = [path.name for path in paths if not path.is_file()]
if missing:
    st.warning("캡스톤 OCSVM 미연결 · models/capstone_ocsvm/에 필요한 파일: " + ", ".join(missing))
    st.stop()
try:
    columns = json.loads(paths[0].read_text(encoding="utf-8-sig"))["input_candidates"][candidate]
    if not isinstance(columns, list) or len(set(columns)) != len(columns):
        raise ValueError("config.json의 특징 목록을 확인해 주세요.")
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"입력 설정을 읽을 수 없습니다. {exc}")
    st.stop()
st.caption("학습 입력 순서 · " + ", ".join(columns))
upload = st.file_uploader("노트북에서 생성한 특징값 CSV", type="csv")
if upload is not None:
    try:
        frame = read_csv(upload.getvalue())
        values = score(frame, columns, joblib.load(paths[1]), joblib.load(paths[2]))
    except Exception as exc:
        st.error(f"희소도 분석 실패 · {exc}")
    else:
        result = frame.copy()
        result.insert(len(result.columns), "OCSVM 희소도", values)
        st.dataframe(result.head(100), hide_index=True)
        chart(px.line(x=range(1, len(values) + 1), y=values, title="OCSVM 희소도",
                      labels={"x": "CSV 행", "y": "희소도 점수"}))
        st.download_button("희소도 결과 CSV 다운로드", result.to_csv(index=False).encode("utf-8-sig"),
                           "capstone_ocsvm_scores.csv", "text/csv")
```

## pages/csv_analysis.py

```python
import hashlib

import streamlit as st

from utils import database, visualization
from utils.preprocessing import numeric_features, read_csv

st.header("CSV 분석 및 센서 신호")
model = st.session_state.model
if model:
    st.caption("학습 입력 순서 · " + " → ".join(f"{name} [{unit}]" for name, unit in model.features.items()))
else:
    st.warning(f"AI 모델 미연결 · {st.session_state.model_detail}")
tab1, tab2 = st.tabs(["특징값 CSV 진단", "원시 시계열 시각화"])
with tab1:
    upload = st.file_uploader("특징값마다 열을, 진단 대상마다 행을 사용한 CSV", type="csv", key="features")
    if upload:
        raw = upload.getvalue()
        try:
            frame = read_csv(raw)
        except Exception as exc:
            st.error(f"CSV 읽기 실패 · {exc}")
        else:
            st.dataframe(frame.head(10), hide_index=True)
            st.caption(f"{len(frame):,}행 × {len(frame.columns):,}열")
            with st.expander("열별 결측치"):
                st.dataframe(frame.isna().sum().rename("결측 개수"))
            if model:
                token = (upload.name, hashlib.sha256(raw).hexdigest(), model.name, model.version, id(model))
                previous = st.session_state.get("csv_token")
                retry = st.button("다시 분석", type="primary", disabled=previous != token)
                if previous != token or retry:
                    st.session_state.csv_token = token
                    st.session_state.pop("csv_result", None)
                    st.session_state.pop("csv_error", None)
                    try:
                        inputs = numeric_features(frame, list(model.features))
                        prediction = model.predict(inputs)
                        timestamp = database.save(f"CSV · {upload.name}", inputs, prediction, model)
                        st.session_state.csv_result = (prediction, timestamp)
                    except Exception as exc:
                        st.session_state.csv_error = f"분석 또는 저장 실패 · {exc}"
                if st.session_state.get("csv_error"):
                    st.error(st.session_state.csv_error)
                if st.session_state.get("csv_result"):
                    prediction, timestamp = st.session_state.csv_result
                    st.success(f"{len(prediction):,}행 진단 완료 · {model.name} {model.version} · {timestamp}")
                    result = frame.copy()
                    for name, column in prediction.items():
                        output_name = f"AI {name}"
                        while output_name in result:
                            output_name += " (진단)"
                        result[output_name] = column
                    st.dataframe(result.head(100), hide_index=True)
                    st.caption("표에는 처음 100행을 표시합니다. 다운로드에는 전체 행이 포함됩니다.")
                    st.download_button("예측 결과 CSV 다운로드", result.to_csv(index=False).encode("utf-8-sig"),
                                       "pipeguard_predictions.csv", "text/csv")
                    st.caption(f"판정 근거 · 모델 출력 라벨 {model.normal}은 정상, {model.anomaly}는 이상입니다.")
                    if model.score_label:
                        st.caption(model.score_label)
                    if prediction["이상 클래스 예측 확률"].isna().all():
                        st.info("이 모델은 이상 클래스 예측 확률을 제공하지 않습니다.")
                    feature = st.selectbox("분포에 표시할 특징", list(model.features))
                    visualization.results(prediction, numeric_features(frame, [feature])[feature], feature, model.features[feature])
with tab2:
    raw_signal = st.file_uploader("원시 시계열 CSV · 행은 샘플 시간 순서", type="csv", key="signal")
    if raw_signal:
        try:
            signal_frame = read_csv(raw_signal.getvalue())
        except Exception as exc:
            st.error(f"원시 신호 읽기 실패 · {exc}")
        else:
            channel = st.selectbox("센서 신호 열", signal_frame.columns)
            fs = st.number_input("샘플링 주파수 (Hz)", min_value=0.001, value=None, step=1.0)
            unit = st.text_input("센서 진폭 단위", placeholder="예: V, g")
            if st.button("파형과 FFT 그리기", disabled=fs is None):
                try:
                    samples = numeric_features(signal_frame, [channel])[channel].to_numpy()
                    if not 4 <= len(samples) <= 250_000:
                        raise ValueError("4~250,000개 샘플을 선택해 주세요.")
                    label = f"{channel} ({unit})" if unit.strip() else channel
                    st.session_state.last_signal = samples, fs, label
                except Exception as exc:
                    st.error(f"신호 처리 실패 · {exc}")
    if st.session_state.get("last_signal") is not None:
        st.caption("마지막으로 생성한 원시 신호 그래프")
        visualization.signal_charts(*st.session_state.last_signal)
```

## pages/dashboard.py

```python
import json
import sqlite3

import pandas as pd
import streamlit as st

from utils import database, visualization

st.header("운영 대시보드")
model = st.session_state.model
if model is None:
    st.warning(f"AI 모델 미연결 · {st.session_state.model_detail}")
try:
    normal, abnormal = database.counts()
    recent = database.history(100)
except (OSError, sqlite3.Error) as exc:
    st.error(f"기록 저장소에 연결할 수 없습니다. {exc}")
    st.stop()
total = normal + abnormal
first = st.columns(2)
first[0].metric("시스템 상태", "모델 연결" if model else "모델 미연결", border=True)
first[1].metric("분석한 데이터", f"{total:,}건", border=True)
second = st.columns(3)
second[0].metric("정상", f"{normal:,}건", border=True)
second[1].metric("이상", f"{abnormal:,}건", border=True)
second[2].metric("이상 발생 비율", f"{abnormal / total:.1%}" if total else "—", border=True)
st.subheader("최근 진단 기록")
if recent.empty:
    st.info("진단을 실행하면 최근 기록과 입력값 분포가 표시됩니다.")
else:
    st.dataframe(recent.head(10), hide_index=True)
    feature = None
    unit = None
    values = None
    plot_frame = recent.iloc[::-1]
    if model:
        matching = plot_frame[(plot_frame["모델명"] == model.name) & (plot_frame["모델 버전"] == model.version)]
        if not matching.empty:
            plot_frame = matching
            feature = st.selectbox("최근 기록에 표시할 센서 특징", list(model.features))
            unit = model.features[feature]
            values = pd.to_numeric(plot_frame["입력 데이터"].map(lambda row: json.loads(row).get(feature)), errors="coerce")
    visualization.results(plot_frame, values, feature, unit)

st.subheader("최근 원시 신호")
signal = st.session_state.get("last_signal")
if signal is None:
    st.info("‘CSV 분석 및 신호’에서 원시 시계열과 샘플링 주파수를 선택하면 이곳에 파형과 FFT가 표시됩니다.")
else:
    visualization.signal_charts(*signal)
```

## pages/history.py

```python
import sqlite3

import streamlit as st

from utils import database

st.header("진단 기록")
label = st.radio("판정 필터", ("전체", "정상", "이상"), horizontal=True)
try:
    records = database.history(verdict=None if label == "전체" else label)
except (OSError, sqlite3.Error) as exc:
    st.error(f"기록을 읽을 수 없습니다. {exc}")
    st.stop()
if records.empty:
    st.info("선택한 조건에 해당하는 기록이 없습니다.")
else:
    st.caption(f"{len(records):,}건 · 최신 500건 표시")
    st.dataframe(records.head(500), hide_index=True)
    st.download_button("진단 기록 CSV 다운로드", records.to_csv(index=False).encode("utf-8-sig"),
                       "pipeguard_history.csv", "text/csv")
```

## pages/manual_diagnosis.py

```python
import pandas as pd
import streamlit as st

from utils import database

st.header("수동 주파수 진단")
model = st.session_state.model
available = model is not None and model.manual
if not available:
    if model is None:
        st.warning(f"AI 모델 미연결 · {st.session_state.model_detail}")
    else:
        st.info("주파수 한 개로 학습하고 수동 진단을 허용한 모델을 연결해 주세요. 현재 모델은 CSV에서 사용할 수 있습니다.")
frequency = st.number_input("주파수 (kHz)", min_value=0.0, value=35.5, format="%.4f", disabled=not available)
if st.button("AI 진단하기", type="primary", disabled=not available):
    st.session_state.pop("last_manual", None)
    feature, unit = next(iter(model.features.items()))
    inputs = pd.DataFrame({feature: [frequency * 1000 if unit == "Hz" else frequency]})
    try:
        predictions = model.predict(inputs)
        timestamp = database.save("수동 주파수", inputs, predictions, model)
        st.session_state.last_manual = (model.name, model.version, frequency, predictions.iloc[0], timestamp)
    except Exception as exc:
        st.error(f"진단 또는 저장 실패 · {exc}")
last = st.session_state.get("last_manual")
if available and last and last[:3] == (model.name, model.version, frequency):
    _, _, _, row, timestamp = last
    alert = st.error if row["판정"] == "이상" else st.success
    alert(f"AI 예측 결과 · {row['판정']}")
    st.write(f"입력 주파수 · {frequency:g} kHz")
    st.write(f"사용 모델 · {model.name} {model.version}")
    st.write(f"진단 시각 · {timestamp}")
    probability = row["이상 클래스 예측 확률"]
    st.write(f"이상 클래스 예측 확률 · {probability:.1%}" if pd.notna(probability) else "이상 클래스 예측 확률 · 모델에서 제공하지 않음")
    if pd.notna(row["결정 점수"]):
        st.write(f"{model.score_label} · {row['결정 점수']:.5g}")
    st.caption(f"판정 근거 · 모델 출력 라벨 {row['모델 출력 라벨']}을 '{row['판정']}'에 매핑했습니다.")
```

## training/train_capstone_ocsvm.py

```python
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
```

## training/train_model.py

```python
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


def train(args):
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
    print(confusion_matrix(actual, prediction, labels=[0, 1]))

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
    }
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        model_file = Path(temporary) / "rf_model.joblib"
        config_file = Path(temporary) / "model_metadata.json"
        joblib.dump(model, model_file, compress=3)
        metadata["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
        config_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        model_file.replace(args.output / model_file.name)
        config_file.replace(args.output / config_file.name)
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
```

## utils/capstone_ocsvm.py

```python
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
```

## utils/database.py

```python
from contextlib import closing
from datetime import datetime
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd

from utils.model_loader import ROOT

DB_PATH = ROOT / "data" / "diagnoses.sqlite3"


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY, analyzed_at TEXT NOT NULL,
                source TEXT NOT NULL, input_json TEXT NOT NULL,
                verdict TEXT NOT NULL, probability REAL, decision_score REAL,
                model_name TEXT NOT NULL, model_version TEXT NOT NULL
            )
        """)
    return conn


def save(source, inputs, predictions, model):
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    inputs_json = (row.to_json(force_ascii=False, double_precision=15) for _, row in inputs.iterrows())
    rows = (
        (timestamp, source, payload, verdict, probability, score, model.name, model.version)
        for payload, (verdict, probability, score, _) in zip(
            inputs_json, predictions.itertuples(index=False, name=None), strict=True
        )
    )
    with closing(connect()) as conn, conn:
        conn.executemany("""
            INSERT INTO diagnoses
            (analyzed_at, source, input_json, verdict, probability, decision_score, model_name, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    return timestamp


def history(limit=-1, verdict=None):
    with closing(connect()) as conn:
        return pd.read_sql_query("""
            SELECT analyzed_at AS '분석 일시', source AS '출처', input_json AS '입력 데이터',
                   verdict AS '판정', probability AS '이상 클래스 예측 확률',
                   decision_score AS '결정 점수', model_name AS '모델명', model_version AS '모델 버전'
            FROM diagnoses WHERE (? IS NULL OR verdict = ?) ORDER BY id DESC LIMIT ?
        """, conn, params=(verdict, verdict, limit))


def counts():
    with closing(connect()) as conn:
        rows = dict(conn.execute("SELECT verdict, count(*) FROM diagnoses GROUP BY verdict"))
    return rows.get("정상", 0), rows.get("이상", 0)
```

## utils/model_loader.py

```python
from dataclasses import dataclass
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
        return Model(estimator, preprocessor, features, meta["model_name"], meta["model_version"],
                     normal, anomaly, manual, mode == "array", meta.get("decision_score_label", "")), ""
    except Exception as exc:
        return None, f"모델 연결 실패 · {exc}"
```

## utils/preprocessing.py

```python
import csv
import io

import numpy as np
import pandas as pd


def read_csv(raw):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV를 UTF-8 또는 CP949 인코딩으로 저장해 주세요.")

    reader = csv.reader(io.StringIO(decoded), strict=True)
    header = next(reader, [])
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV 첫 행에 모든 열의 이름을 입력해 주세요.")
    if len(set(header)) != len(header):
        raise ValueError("중복된 열 이름을 정리해 주세요.")
    for row in reader:
        if row and len(row) != len(header):
            raise ValueError(f"CSV {reader.line_num}번째 줄의 열 개수가 헤더와 다릅니다.")
    frame = pd.read_csv(io.StringIO(decoded), index_col=False, dtype=str)
    if frame.empty:
        raise ValueError("CSV에 분석할 데이터 행이 없습니다.")
    return frame


def numeric_features(frame, names):
    absent = [name for name in names if name not in frame]
    if absent:
        raise ValueError("누락된 특징값: " + ", ".join(absent))
    result = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").astype(float)
    valid = np.isfinite(result.to_numpy())
    if not valid.all():
        bad = result.columns[~valid.all(axis=0)].tolist()
        rows = np.flatnonzero(~valid.all(axis=1))[:5] + 1
        raise ValueError(f"숫자 형식·결측·무한대 확인 필요: {', '.join(bad)} (데이터 행 {rows.tolist()})")
    return result
```

## utils/visualization.py

```python
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import streamlit as st

COLORS = {"정상": "#42D392", "이상": "#FF6577"}


def chart(fig):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0A111D", plot_bgcolor="#101B2B",
        margin=dict(l=12, r=15, t=55, b=12), height=320, legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def results(frame, input_values=None, feature=None, unit=None):
    counts = frame["판정"].value_counts().reindex(["정상", "이상"], fill_value=0)
    distribution = pd.DataFrame({"판정": counts.index, "개수": counts.values})
    chart(px.bar(distribution, x="판정", y="개수", color="판정",
                 color_discrete_map=COLORS, title="진단 결과별 개수"))
    if input_values is not None:
        data = pd.DataFrame({feature: input_values, "판정": frame["판정"].to_numpy()}).dropna()
        if not data.empty:
            chart(px.histogram(data, x=feature, color="판정", barmode="overlay", opacity=.7,
                               labels={feature: f"{feature} ({unit})"},
                               color_discrete_map=COLORS, title="정상·이상 입력값 분포"))
    if frame["이상 클래스 예측 확률"].notna().any():
        probability = frame.reset_index(drop=True).assign(순번=lambda table: table.index + 1)
        chart(px.scatter(probability, x="순번", y="이상 클래스 예측 확률",
                         color="판정", color_discrete_map=COLORS,
                         range_y=[-.03, 1.03], title="이상 클래스 예측 확률"))


def signal_charts(samples, fs, label):
    time = np.arange(len(samples)) / fs
    step = max(1, int(np.ceil(len(samples) / 15_000)))
    chart(px.line(x=time[::step], y=samples[::step], title="시간 영역 파형",
                  labels={"x": "시간 (s)", "y": label}))
    window = hann(len(samples), sym=False)
    amplitude = np.abs(rfft((samples - samples.mean()) * window)) / window.sum()
    amplitude[1:] *= 2
    if len(samples) % 2 == 0:
        amplitude[-1] /= 2
    frequency = rfftfreq(len(samples), d=1 / fs) / 1000
    chart(px.line(x=frequency, y=amplitude, title="FFT 주파수 스펙트럼",
                  labels={"x": "주파수 (kHz)", "y": f"진폭 · {label}"}))
    st.caption("FFT는 평균을 제거하고 Hann 윈도를 적용한 단측 진폭입니다. 원시 신호는 등간격 샘플이어야 합니다.")
```

## requirements.txt

```text
streamlit>=1.52,<2
pandas>=2.2,<4
numpy>=2.1,<3
scipy>=1.14.1,<2
scikit-learn>=1.6,<2
plotly>=6,<7
joblib>=1.4,<2
tzdata>=2025.1
```

## .streamlit/config.toml

```toml
[theme]
base = "dark"
primaryColor = "#30DBC5"
backgroundColor = "#0A111D"
secondaryBackgroundColor = "#152338"
textColor = "#E6F0FC"
showWidgetBorder = true

[server]
address = "localhost"
maxUploadSize = 25

[browser]
gatherUsageStats = false
```
