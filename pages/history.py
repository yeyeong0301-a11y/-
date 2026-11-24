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
