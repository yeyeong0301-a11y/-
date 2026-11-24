import math

import numpy as np
import pandas as pd
import streamlit as st

from utils import evaluation, visualization
from utils.preprocessing import numeric_features, read_csv

st.header("판정 검증")
st.caption("정답 라벨이 있는 CSV로 모델 판정이 맞는지 확인합니다. 이 화면의 결과는 진단 기록에 저장하지 않습니다.")
model = st.session_state.model
if model is None:
    st.warning(f"모델 미연결 · {st.session_state.model_detail}")
    st.caption("모델 없이 화면만 확인하려면 `python -m training.make_demo`로 합성 데모 모델을 만드세요.")
    st.stop()
if model.synthetic:
    st.warning("합성 데모 모델입니다. 화면 확인용 가짜 데이터로 학습했으며 실제 배관 상태와 관계없습니다.")

if model.test_metrics:
    tm = model.test_metrics
    c = tm.get("confusion", {})
    with st.expander("학습 때 떼어 둔 테스트 결과"):
        st.write(f"학습 {tm.get('train_rows', 0):,}행 · 테스트 {tm.get('test_rows', 0):,}행 · "
                 f"재현율 {tm.get('recall', math.nan):.3f} · 정밀도 {tm.get('precision', math.nan):.3f} · "
                 f"F1 {tm.get('f1', math.nan):.3f}")
        st.caption(f"TP {c.get('tp', '-')} · FN {c.get('fn', '-')} · FP {c.get('fp', '-')} · TN {c.get('tn', '-')}. "
                   f"`models/{tm.get('holdout_file', 'holdout_labeled.csv')}`을 올리면 같은 행을 이 화면에서 다시 확인할 수 있습니다.")

upload = st.file_uploader("정답 라벨 열이 있는 특징값 CSV", type="csv", key="verify")
if upload is None:
    st.info("특징값 열과 정답 라벨 열(예: label = 정상/이상)이 있는 CSV를 올리세요. "
            "학습에 쓰지 않은 독립 측정 데이터여야 실제 성능에 가깝습니다.")
    st.stop()

try:
    frame = read_csv(upload.getvalue())
except Exception as exc:
    st.error(f"CSV 읽기 실패 · {exc}")
    st.stop()

features = list(model.features)
candidates = [name for name in frame.columns if name not in features]
if not candidates:
    st.error("특징값 외에 정답 라벨 열이 없습니다.")
    st.stop()
default = candidates.index("label") if "label" in candidates else 0
left, middle, right = st.columns(3)
label_column = left.selectbox("정답 라벨 열", candidates, index=default)
values = sorted(frame[label_column].dropna().astype(str).str.strip().unique())
if len(values) > 20:
    st.error(f"‘{label_column}’ 열의 값이 {len(values)}종류입니다. 정답 라벨 열을 다시 고르세요.")
    st.stop()
NONE = "(없음)"
guess_normal, guess_anomaly = evaluation.guess_labels(values)
if guess_normal is None and len(values) >= 2:
    guess_normal, guess_anomaly = values[0], values[1]
normal_options = values + [NONE]
normal = middle.selectbox("정상을 뜻하는 값", normal_options,
                          index=normal_options.index(guess_normal) if guess_normal in values else 0)
anomaly_options = [value for value in values if value != normal] + [NONE]
anomaly = right.selectbox("이상을 뜻하는 값", anomaly_options,
                          index=anomaly_options.index(guess_anomaly) if guess_anomaly in anomaly_options else 0)
if normal == NONE and anomaly == NONE:
    st.error("정상과 이상 중 하나 이상은 라벨 값을 골라야 합니다.")
    st.stop()
normal, anomaly = (None if normal == NONE else normal), (None if anomaly == NONE else anomaly)

try:
    actual_all = evaluation.encode_labels(frame[label_column], normal, anomaly)
    inputs_all = numeric_features(frame, features)
    prediction_all = model.predict(inputs_all)
except Exception as exc:
    st.error(f"검증 실패 · {exc}")
    st.stop()

# 1. 입력 점검
st.subheader("입력 점검")
overlap = evaluation.overlap_with_training(inputs_all, model.train_hashes)
keep = np.ones(len(frame), dtype=bool)
if overlap is not None and overlap.any():
    exclude = st.checkbox(f"학습에 쓴 행과 같은 {int(overlap.sum()):,}행 빼고 계산", value=True)
    st.caption("학습에 쓴 행이 섞이면 성능이 실제보다 좋게 나옵니다.")
    if exclude:
        keep = ~overlap.to_numpy()
elif overlap is None:
    st.caption("이 모델에는 학습 행 기록이 없어 학습 데이터가 섞였는지 확인하지 못합니다. 독립 측정 데이터만 쓰세요.")
if not keep.any():
    st.error("남은 행이 없습니다. 학습에 쓰지 않은 데이터로 검증하세요.")
    st.stop()

rows = np.flatnonzero(keep) + 1  # CSV 데이터 행 번호 (헤더 제외, 1부터)
inputs = inputs_all[keep].reset_index(drop=True)
prediction = prediction_all[keep].reset_index(drop=True)
actual = actual_all[keep]
outside, range_summary = evaluation.range_check(inputs, model.ranges) if model.ranges else (None, None)

checks = st.columns(4)
checks[0].metric("검증 행", f"{len(inputs):,}")
checks[1].metric("정답 정상 · 이상", f"{int((actual == 0).sum()):,} · {int((actual == 1).sum()):,}")
if outside is not None:
    checks[2].metric("학습 범위 밖 행", f"{int((outside > 0).sum()):,}")
else:
    checks[2].metric("학습 범위 밖 행", "—", help="범위 정보가 없는 모델입니다. 새로 학습하면 저장됩니다.")
if overlap is None:
    checks[3].metric("학습 행과 같은 행", "—", help="학습 행 기록이 없는 모델입니다.")
else:
    checks[3].metric("학습 행과 같은 행", f"{int(overlap.sum()):,}",
                     help="특징값이 학습에 쓴 행과 완전히 같은 행. 업로드 전체 기준")
for note in evaluation.input_issues(inputs, features, actual):
    st.warning(note)
if range_summary is not None:
    flagged = range_summary[(range_summary["범위 밖 행"] > 0) | range_summary["확인할 점"].ne("")]
    if not flagged.empty:
        st.dataframe(flagged, hide_index=True, column_config={
            name: st.column_config.NumberColumn(format="%.4g") for name in ("학습 최소", "학습 중앙값", "학습 최대", "입력 중앙값")
        })
        if flagged["확인할 점"].ne("").any():
            st.warning("단위나 변환이 학습 때와 다를 수 있습니다. 판정 결과를 보기 전에 표의 ‘확인할 점’을 먼저 확인하세요.")

# 2. 임계값과 지표
st.subheader("판정 결과")
scored = evaluation.anomaly_score(prediction)
if scored is None:
    predicted = (prediction["판정"] == "이상").to_numpy().astype(int)
    threshold = margin = None
    score = None
    st.caption("이 모델은 확률이나 점수를 제공하지 않아 모델 판정을 그대로 비교합니다.")
else:
    score, default_threshold, score_name = scored
    if score_name == "이상 클래스 예측 확률":
        threshold = st.slider("임계값 · 확률이 이 값 이상이면 이상", 0.0, 1.0, default_threshold, .01)
        margin = .1
        st.caption("RF 확률은 나무들의 투표 비율이며 실제 고장 확률로 보정한 값이 아닙니다. "
                   "모델 기본 판정은 0.5이고, 정확히 0.5인 행은 기본 판정에서 정상으로 처리됩니다.")
    else:
        low, high = float(score.min()), float(score.max())
        if high <= low:
            high = low + 1e-6
        threshold = st.slider("임계값 · 점수가 이 값 이상이면 이상", min(low, default_threshold),
                              max(high, default_threshold), default_threshold)
        margin = (high - low) * .05
    predicted = (score >= threshold).astype(int)
    changed = int((predicted != (prediction["판정"] == "이상").to_numpy()).sum())
    if changed:
        st.caption(f"기본 판정과 다른 행 {changed:,}개 · 아래 지표는 슬라이더 임계값 기준입니다.")

result = evaluation.metrics(actual, predicted)


def share(item):
    value, numerator, denominator = item["값"], item["분자"], item["분모"]
    return "—" if not denominator else f"{value:.1%}"


def share_help(item):
    return f"{item['분자']:,} / {item['분모']:,}건" if item["분모"] else "분모가 0이라 계산하지 않음"


cards = st.columns(4)
cards[0].metric("재현율", share(result["재현율"]), help="실제 이상 중 이상으로 잡은 비율")
cards[0].caption(share_help(result["재현율"]))
cards[1].metric("오경보율", share(result["오경보율"]), help="실제 정상 중 이상으로 잘못 판정한 비율")
cards[1].caption(share_help(result["오경보율"]))
cards[2].metric("정밀도", share(result["정밀도"]), help="이상 판정 중 실제 이상 비율")
cards[2].caption(share_help(result["정밀도"]))
cards[3].metric("MCC", "—" if math.isnan(result["MCC"]) else f"{result['MCC']:.3f}",
                help="네 칸을 모두 반영한 요약값. 1은 완전 일치, 0은 무작위 수준")
cards[3].caption(f"F1 {share(result['F1'])}")
for note in evaluation.sample_warnings(result):
    st.warning(note)

matrix = pd.DataFrame(
    {"판정 정상": [result["TN"], result["FN"]], "판정 이상": [result["FP"], result["TP"]]},
    index=["정답 정상", "정답 이상"],
)
left, right = st.columns([1, 2])
with left:
    st.caption("혼동행렬 (행 수)")
    st.dataframe(matrix)
with right:
    st.caption("읽는 법")
    st.markdown(f"- 놓친 이상 **{result['FN']:,}행** · 실제 이상인데 정상으로 판정\n"
                f"- 잘못 경보 **{result['FP']:,}행** · 실제 정상인데 이상으로 판정")

if score is not None:
    left, right = st.columns([2, 1])
    with left:
        visualization.score_distribution(score, actual, threshold, score_name)
    with right:
        curve, average_precision = evaluation.pr_curve(actual, score)
        if curve is None:
            st.caption("정답에 정상과 이상이 모두 있어야 정밀도·재현율 곡선을 그립니다.")
        else:
            visualization.pr_chart(curve, result["정밀도"]["값"], result["재현율"]["값"], average_precision)

# 3. 틀린 행
outcome = np.select(
    [(actual == 1) & (predicted == 1), (actual == 1) & (predicted == 0), (actual == 0) & (predicted == 1)],
    ["맞춘 이상", "놓친 이상", "잘못 경보"], "맞춘 정상",
)
notes = evaluation.row_notes(len(inputs), outside, score, threshold, margin)
table = pd.DataFrame({
    "행": rows, "정답": np.where(actual == 1, "이상", "정상"), "판정": np.where(predicted == 1, "이상", "정상"),
    "결과": outcome, "점수": score if score is not None else np.nan, "확인할 점": notes,
})
table = pd.concat([table, inputs], axis=1)
wrong = table[table["결과"].isin(["놓친 이상", "잘못 경보"])]
st.subheader(f"틀린 행 {len(wrong):,}개")
if wrong.empty:
    st.caption("현재 임계값에서 틀린 행이 없습니다.")
else:
    if score is not None:
        wrong = wrong.assign(_gap=(wrong["점수"] - threshold).abs()).sort_values(["결과", "_gap"]).drop(columns="_gap")
    st.dataframe(wrong, hide_index=True, column_config={
        "행": st.column_config.NumberColumn(help="CSV 데이터 행 번호 (헤더 제외)"),
        "점수": st.column_config.NumberColumn(format="%.3f"),
    })
    near = int(wrong["확인할 점"].str.contains("임계값 근처").sum())
    far = int(wrong["확인할 점"].str.contains("학습 범위 밖").sum())
    st.caption(f"틀린 행 중 임계값 근처 {near:,}개, 학습 범위 밖 특징이 있는 행 {far:,}개입니다. "
               "임계값 근처는 기준을 옮기면 바뀌고, 범위 밖은 학습 데이터에 없던 입력입니다.")
st.download_button("검증 결과 CSV 다운로드", table.to_csv(index=False).encode("utf-8-sig"),
                   "pipeguard_verification.csv", "text/csv")

# 4. 특징별로 어디서 틀리는지
st.subheader("특징별 확인")
feature = st.selectbox("특징", features)
visualization.outcome_strip(inputs[feature], outcome, feature, model.features[feature], model.ranges.get(feature))
