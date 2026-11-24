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
