import hashlib

import numpy as np
import streamlit as st

from utils import database, evaluation, visualization
from utils.preprocessing import numeric_features, read_csv

st.header("CSV 분석 및 신호")
model = st.session_state.model
if model:
    st.caption("학습 입력 순서 · " + " → ".join(f"{name} [{unit}]" for name, unit in model.features.items()))
    if model.synthetic:
        st.warning("합성 데모 모델입니다. 화면 확인용이며 실제 배관 상태와 관계없습니다.")
else:
    st.warning(f"모델 미연결 · {st.session_state.model_detail}")
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
            missing = frame.isna().sum()
            st.caption(f"{len(frame):,}행 × {len(frame.columns):,}열 · 빈 칸 {int(missing.sum()):,}개")
            if missing.any():
                with st.expander("열별 빈 칸"):
                    st.dataframe(missing[missing > 0].rename("빈 칸 수"))
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
                        source = f"CSV · {upload.name}" + (" · 합성 데모" if model.synthetic else "")
                        timestamp = database.save(source, inputs, prediction, model)
                        st.session_state.csv_result = (inputs.reset_index(drop=True), prediction, timestamp)
                    except Exception as exc:
                        st.session_state.csv_error = f"분석 또는 저장 실패 · {exc}"
                if st.session_state.get("csv_error"):
                    st.error(st.session_state.csv_error)
                if st.session_state.get("csv_result"):
                    inputs, prediction, timestamp = st.session_state.csv_result
                    normal, abnormal = visualization.verdict_counts(prediction)
                    outside, range_summary = (evaluation.range_check(inputs, model.ranges)
                                              if model.ranges else (None, None))
                    probability = prediction["이상 클래스 예측 확률"].to_numpy(dtype=float)
                    has_probability = np.isfinite(probability).all()
                    notes = evaluation.row_notes(len(inputs), outside,
                                                 probability if has_probability else None, .5, .1)
                    attention = sum(bool(note) for note in notes)

                    stats = st.columns(4)
                    stats[0].metric("진단한 행", f"{len(prediction):,}")
                    stats[1].metric("정상", f"{normal:,}")
                    stats[2].metric("이상", f"{abnormal:,}")
                    stats[2].caption(f"전체의 {abnormal / len(prediction):.1%}")
                    stats[3].metric("확인할 행", f"{attention:,}",
                                    help="학습 범위 밖 특징이 있거나 이상 확률이 0.4~0.6인 행")
                    st.caption(f"{model.name} {model.version} · {timestamp} 저장 · "
                               f"모델 출력 {model.normal}은 정상, {model.anomaly}는 이상")
                    if range_summary is not None and range_summary["확인할 점"].ne("").any():
                        st.warning("단위나 변환이 학습 때와 다를 수 있습니다. 판정을 쓰기 전에 아래 표를 확인하세요.")
                        st.dataframe(range_summary[range_summary["확인할 점"].ne("")], hide_index=True)
                    elif not model.ranges:
                        st.caption("이 모델에는 학습 범위 정보가 없어 입력 범위를 점검하지 않습니다.")

                    # 판정과 확인할 점을 앞에 두고 원래 열을 뒤에 붙인다.
                    result = frame.copy()
                    front = {}
                    for name, column in [*prediction.items(), ("확인할 점", notes)]:
                        output_name = name
                        while output_name in result or output_name in front:
                            output_name += " (진단)"
                        front[output_name] = np.asarray(column)
                    for position, (name, column) in enumerate(front.items()):
                        result.insert(position, name, column)
                    only_attention = st.toggle("확인할 행만 보기", value=False, disabled=attention == 0)
                    shown = result[result[list(front)[-1]].ne("")] if only_attention else result
                    st.dataframe(shown.head(200), hide_index=True)
                    st.caption("표에는 처음 200행을 표시합니다. 다운로드에는 전체 행이 포함됩니다.")
                    st.download_button("예측 결과 CSV 다운로드", result.to_csv(index=False).encode("utf-8-sig"),
                                       "pipeguard_predictions.csv", "text/csv")
                    if model.score_label:
                        st.caption(model.score_label)

                    if has_probability:
                        visualization.probability_strip(prediction, .5, key="csv-probability")
                    else:
                        st.info("이 모델은 이상 클래스 예측 확률을 제공하지 않습니다.")
                    feature = st.selectbox("분포에 표시할 특징", list(model.features))
                    visualization.feature_distribution(inputs[feature], prediction["판정"], feature,
                                                       model.features[feature], model.ranges.get(feature),
                                                       key="csv-feature")
                    st.caption("정답 라벨이 있으면 ‘판정 검증’에서 판정이 맞았는지 확인할 수 있습니다.")
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
