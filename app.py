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

# 실시간 진동 번들 화면은 해당 파일이 있을 때만 메뉴에 넣는다.
optional = [
    ("pages/mat_playback.py", "MAT 순차 재생·알림", ":material/play_circle:"),
    ("pages/live_monitor.py", "실시간 진동·알림", ":material/notifications_active:"),
]
pages = [st.Page("pages/dashboard.py", title="대시보드", icon=":material/dashboard:", default=True)]
pages += [st.Page(path, title=title, icon=icon) for path, title, icon in optional if (ROOT / path).is_file()]
pages += [
    st.Page("pages/manual_diagnosis.py", title="수동 진단", icon=":material/sensors:"),
    st.Page("pages/csv_analysis.py", title="CSV 분석 및 신호", icon=":material/monitoring:"),
    st.Page("pages/verification.py", title="판정 검증", icon=":material/fact_check:"),
    st.Page("pages/capstone_ocsvm.py", title="캡스톤 OCSVM", icon=":material/analytics:"),
    st.Page("pages/history.py", title="진단 기록", icon=":material/history:"),
]
page = st.navigation(pages)

st.sidebar.divider()
st.sidebar.markdown("**PIPEGUARD AI**")
st.sidebar.caption("지하배관 진동 특징값 판정과 검증")
model = st.session_state.model
if model is None:
    st.sidebar.caption("● 모델 미연결 · 수동·CSV 진단과 판정 검증에는 models/rf_model.joblib이 필요합니다")
else:
    st.sidebar.caption(f"● {model.name} · {model.version}")
    if model.synthetic:
        st.sidebar.caption("합성 데모 모델 · 실제 판정에 쓰지 마세요")
page.run()
