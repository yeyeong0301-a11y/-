import json
import sqlite3

import pandas as pd
import streamlit as st

from utils import database, visualization

st.header("운영 대시보드")
model = st.session_state.model
if model is None:
    st.warning(f"모델 미연결 · {st.session_state.model_detail}")
elif model.synthetic:
    st.warning("합성 데모 모델이 연결되어 있습니다. 화면 확인용이며 실제 배관 상태와 관계없습니다.")
try:
    normal, abnormal = database.counts()
    recent = database.history(500)
except (OSError, sqlite3.Error) as exc:
    st.error(f"기록 저장소에 연결할 수 없습니다. {exc}")
    st.stop()
total = normal + abnormal
stats = st.columns(4)
stats[0].metric("분석한 행", f"{total:,}")
stats[1].metric("정상", f"{normal:,}")
stats[2].metric("이상", f"{abnormal:,}")
stats[3].metric("이상 비율", f"{abnormal / total:.1%}" if total else "—")
st.caption("전체 기록 기준입니다. 모델 버전과 출처가 섞여 있을 수 있습니다. "
           + (f"현재 모델 · {model.name} {model.version}" if model else "현재 연결된 모델 없음"))

st.subheader("최근 진단 기록")
if recent.empty:
    st.info("진단을 실행하면 최근 기록과 입력값 분포가 표시됩니다.")
else:
    order = ["분석 일시", "판정", "이상 클래스 예측 확률", "결정 점수", "출처", "모델명", "모델 버전", "입력 데이터"]
    st.dataframe(recent.head(10)[order], hide_index=True, column_config={
        "이상 클래스 예측 확률": st.column_config.NumberColumn(format="%.3f"),
        "결정 점수": st.column_config.NumberColumn(format="%.4g"),
    })
    if model:
        matching = recent[(recent["모델명"] == model.name) & (recent["모델 버전"] == model.version)].iloc[::-1]
        if matching.empty:
            st.caption("현재 모델로 진단한 기록이 없어 그래프를 그리지 않습니다.")
        else:
            st.caption(f"아래 그래프는 현재 모델로 진단한 최근 {len(matching):,}행만 사용합니다.")
            feature = st.selectbox("최근 기록에 표시할 센서 특징", list(model.features))
            values = pd.to_numeric(matching["입력 데이터"].map(lambda row: json.loads(row).get(feature)),
                                   errors="coerce")
            visualization.results(matching, values, feature, model.features[feature], model.ranges.get(feature))
    else:
        st.caption("모델을 연결하면 현재 모델의 기록만 골라 그래프로 보여줍니다.")

st.subheader("최근 원시 신호")
signal = st.session_state.get("last_signal")
if signal is None:
    st.info("‘CSV 분석 및 신호’에서 원시 시계열과 샘플링 주파수를 선택하면 이곳에 파형과 FFT가 표시됩니다.")
else:
    visualization.signal_charts(*signal)
