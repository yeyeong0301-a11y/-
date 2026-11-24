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
