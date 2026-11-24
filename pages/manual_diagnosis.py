import pandas as pd
import streamlit as st

from utils import database

st.header("수동 주파수 진단")
model = st.session_state.model
available = model is not None and model.manual
if not available:
    if model is None:
        st.warning(f"모델 미연결 · {st.session_state.model_detail}")
    else:
        st.info("주파수 한 개로 학습하고 수동 진단을 허용한 모델을 연결해 주세요. 현재 모델은 CSV에서 사용할 수 있습니다.")
frequency = st.number_input("주파수 (kHz)", min_value=0.0, value=35.5, format="%.4f", disabled=not available)
if st.button("진단", type="primary", disabled=not available):
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
    alert(f"판정 · {row['판정']}")
    st.write(f"입력 주파수 · {frequency:g} kHz")
    st.write(f"사용 모델 · {model.name} {model.version}")
    st.write(f"진단 시각 · {timestamp}")
    probability = row["이상 클래스 예측 확률"]
    st.write(f"이상 클래스 예측 확률 · {probability:.1%}" if pd.notna(probability) else "이상 클래스 예측 확률 · 모델에서 제공하지 않음")
    if pd.notna(row["결정 점수"]):
        st.write(f"{model.score_label} · {row['결정 점수']:.5g}")
    st.caption(f"판정 근거 · 모델 출력 라벨 {row['모델 출력 라벨']}을 '{row['판정']}'에 매핑했습니다.")
