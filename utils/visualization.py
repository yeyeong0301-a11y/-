import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
import streamlit as st

# 계측 기록지 같은 회색 위에 이상만 적주황. 학습 범위 밖은 황토색.
INK = "#1c2127"
SECONDARY = "#57606a"
MUTED = "#6e7781"
NEUTRAL = "#8c959f"
ACCENT = "#c63d2f"
CAUTION = "#b87a1e"
STEEL = "#2d3e50"
GRID = "#e6ebf0"
AXIS = "#b6c0ca"
BAND = "rgba(45, 62, 80, 0.07)"
COLORS = {"정상": NEUTRAL, "이상": ACCENT}
OUTCOME_COLORS = {"맞춘 정상": "#c3c9cf", "맞춘 이상": NEUTRAL, "놓친 이상": ACCENT, "잘못 경보": CAUTION}
FONT = "IBM Plex Sans KR, Pretendard, sans-serif"
MONO = "IBM Plex Mono, ui-monospace, monospace"


def chart(fig, height=300, key=None):
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family=FONT, size=12, color=SECONDARY),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=.97),
        margin=dict(l=48, r=16, t=44, b=40), height=height,
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(color=SECONDARY)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        bargap=.08,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=AXIS, zeroline=False, ticks="outside", ticklen=4,
                     tickcolor=AXIS, tickfont=dict(family=MONO, size=11, color=SECONDARY), title_font=dict(color=SECONDARY))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, zeroline=False, ticks="outside", ticklen=4,
                     tickcolor=AXIS, tickfont=dict(family=MONO, size=11, color=SECONDARY), title_font=dict(color=SECONDARY))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=key)


def training_band(fig, item, horizontal=True):
    """학습 데이터의 1~99% 구간을 옅은 띠로, 최소·최대를 점선으로 그린다."""
    if not item:
        return
    add_rect, add_line = (fig.add_vrect, fig.add_vline) if horizontal else (fig.add_hrect, fig.add_hline)
    add_rect(item["p01"], item["p99"], fillcolor=BAND, line_width=0, layer="below")
    for edge in ("min", "max"):
        add_line(item[edge], line=dict(color=MUTED, width=1, dash="dot"))


def verdict_counts(frame):
    counts = frame["판정"].value_counts()
    return int(counts.get("정상", 0)), int(counts.get("이상", 0))


def probability_strip(frame, threshold=.5, rows=None, title="이상 클래스 예측 확률", key=None):
    """행 순서대로 확률을 찍고 판정 기준선을 긋는다. 호버에 원본 행 번호를 보여준다."""
    data = frame.reset_index(drop=True)
    data["행"] = rows if rows is not None else np.arange(1, len(data) + 1)
    fig = px.scatter(data, x="행", y="이상 클래스 예측 확률", color="판정", color_discrete_map=COLORS,
                     category_orders={"판정": ["정상", "이상"]}, range_y=[-.03, 1.03], title=title)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.add_hline(threshold, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text=f"기준 {threshold:g}", annotation_position="top left",
                  annotation_font=dict(color=MUTED, family=MONO, size=11))
    chart(fig, key=key)


def feature_distribution(values, verdicts, feature, unit, item=None, key=None):
    """판정별 특징 분포. 비율로 그려 한쪽이 적어도 모양이 보이게 하고 학습 범위를 겹친다."""
    data = pd.DataFrame({feature: np.asarray(values, dtype=float), "판정": np.asarray(verdicts)})
    dropped = int(data[feature].isna().sum())
    data = data.dropna()
    if data.empty:
        st.caption("표시할 숫자 값이 없습니다.")
        return
    if dropped:
        st.caption(f"숫자로 읽지 못한 {dropped:,}행은 그래프에서 빠졌습니다.")
    fig = px.histogram(data, x=feature, color="판정", barmode="overlay", opacity=.75, histnorm="percent",
                       color_discrete_map=COLORS, category_orders={"판정": ["정상", "이상"]},
                       labels={feature: f"{feature} ({unit})" if unit else feature},
                       title="판정별 입력값 분포")
    fig.update_yaxes(title_text="판정 안에서의 비율 (%)")
    training_band(fig, item)
    chart(fig, key=key)
    if item:
        st.caption("옅은 띠는 학습 데이터의 1~99% 구간, 점선은 학습 최소·최대입니다.")


def score_distribution(score, actual, threshold, name):
    """실제 라벨별 점수 분포와 현재 임계값. 두 무리가 갈라지는지 보는 주 차트."""
    data = pd.DataFrame({name: score, "정답": np.where(np.asarray(actual) == 1, "이상", "정상")})
    fig = px.histogram(data, x=name, color="정답", barmode="overlay", opacity=.8, nbins=40,
                       color_discrete_map=COLORS, category_orders={"정답": ["정상", "이상"]},
                       title="정답별 점수 분포")
    fig.update_yaxes(title_text="행 수")
    fig.add_vline(threshold, line=dict(color=INK, width=1, dash="dash"),
                  annotation_text=f"임계값 {threshold:.3g}", annotation_position="top right",
                  annotation_font=dict(color=INK, family=MONO, size=11))
    chart(fig, height=340)


def pr_chart(curve, precision, recall, average_precision):
    fig = go.Figure(go.Scatter(x=curve["재현율"], y=curve["정밀도"], mode="lines", line=dict(color=NEUTRAL, width=2),
                               line_shape="hv", hovertemplate="재현율 %{x:.2f}<br>정밀도 %{y:.2f}<extra></extra>"))
    if np.isfinite(precision) and np.isfinite(recall):
        fig.add_trace(go.Scatter(x=[recall], y=[precision], mode="markers", marker=dict(color=INK, size=9, symbol="diamond"),
                                 hovertemplate="현재 임계값<br>재현율 %{x:.2f}<br>정밀도 %{y:.2f}<extra></extra>"))
    fig.update_layout(showlegend=False, title=f"정밀도·재현율 곡선 · AP {average_precision:.3f}")
    fig.update_xaxes(title_text="재현율", range=[0, 1.02])
    fig.update_yaxes(title_text="정밀도", range=[0, 1.02])
    chart(fig, height=340)


def outcome_strip(values, outcomes, feature, unit, item=None):
    """특징값 위에 맞춘 행과 틀린 행을 나눠 찍는다. 틀린 행이 학습 범위 어디에 있는지 본다."""
    order = ["맞춘 정상", "잘못 경보", "놓친 이상", "맞춘 이상"]
    data = pd.DataFrame({feature: np.asarray(values, dtype=float), "결과": outcomes})
    fig = px.strip(data, x=feature, y="결과", color="결과", color_discrete_map=OUTCOME_COLORS,
                   category_orders={"결과": order}, labels={feature: f"{feature} ({unit})" if unit else feature},
                   title="특징값별 맞춘 행과 틀린 행")
    fig.update_traces(marker=dict(size=6, opacity=.85), jitter=.5)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="")
    training_band(fig, item)
    chart(fig, height=280)


def results(frame, input_values=None, feature=None, unit=None, item=None):
    """대시보드용: 판정별 입력값 분포와 확률."""
    if input_values is not None:
        feature_distribution(input_values, frame["판정"], feature, unit, item, key="dashboard-feature")
    if frame["이상 클래스 예측 확률"].notna().any():
        probability_strip(frame.dropna(subset=["이상 클래스 예측 확률"]),
                          title="최근 기록의 이상 클래스 예측 확률", key="dashboard-probability")


def signal_charts(samples, fs, label):
    time = np.arange(len(samples)) / fs
    step = max(1, int(np.ceil(len(samples) / 15_000)))
    fig = px.line(x=time[::step], y=samples[::step], title="시간 영역 파형", labels={"x": "시간 (s)", "y": label})
    fig.update_traces(line=dict(color=STEEL, width=1))
    chart(fig)
    window = hann(len(samples), sym=False)
    amplitude = np.abs(rfft((samples - samples.mean()) * window)) / window.sum()
    amplitude[1:] *= 2
    if len(samples) % 2 == 0:
        amplitude[-1] /= 2
    frequency = rfftfreq(len(samples), d=1 / fs) / 1000
    fig = px.line(x=frequency, y=amplitude, title="FFT 주파수 스펙트럼", labels={"x": "주파수 (kHz)", "y": f"진폭 · {label}"})
    fig.update_traces(line=dict(color=STEEL, width=1))
    if len(amplitude) > 1:
        peak = int(np.argmax(amplitude[1:]) + 1)
        fig.add_trace(go.Scatter(x=[frequency[peak]], y=[amplitude[peak]], mode="markers+text",
                                 marker=dict(color=STEEL, size=7), text=[f"최대 {frequency[peak]:.4g} kHz"],
                                 textposition="top right", textfont=dict(color=INK, family=MONO, size=11),
                                 showlegend=False, hoverinfo="skip"))
    chart(fig)
    st.caption("FFT는 평균을 제거하고 Hann 윈도를 적용한 단측 진폭입니다. 원시 신호는 등간격 샘플이어야 합니다.")
