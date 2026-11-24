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
    header = [name.strip() for name in next(reader, [])]
    if not header or any(not name for name in header):
        raise ValueError("CSV 첫 행에 모든 열의 이름을 입력해 주세요.")
    if len(set(header)) != len(header):
        raise ValueError("중복된 열 이름을 정리해 주세요 (앞뒤 공백은 무시합니다).")
    if len(header) == 1 and any(mark in header[0] for mark in (";", "\t")):
        raise ValueError("쉼표(,)로 구분한 CSV로 저장해 주세요.")
    for row in reader:
        if row and len(row) != len(header):
            raise ValueError(f"CSV {reader.line_num}번째 줄의 열 개수가 헤더와 다릅니다.")
    # "NA"·"null" 같은 글자를 조용히 빈 칸으로 바꾸지 않는다. 빈 칸만 결측으로 본다.
    frame = pd.read_csv(io.StringIO(decoded), index_col=False, dtype=str, keep_default_na=False, na_values=[""])
    frame.columns = header
    if frame.empty:
        raise ValueError("CSV에 분석할 데이터 행이 없습니다.")
    return frame


def numeric_features(frame, names):
    absent = [name for name in names if name not in frame]
    if absent:
        raise ValueError("누락된 특징값: " + ", ".join(absent))
    source = frame.loc[:, names]
    result = source.apply(lambda column: pd.to_numeric(column.str.strip() if column.dtype == object else column,
                                                       errors="coerce")).astype(float)
    valid = np.isfinite(result.to_numpy())
    if not valid.all():
        bad = result.columns[~valid.all(axis=0)].tolist()
        rows = np.flatnonzero(~valid.all(axis=1))[:5] + 1
        row, column = np.argwhere(~valid)[0]
        example = source.iat[row, column]
        shown = "빈 칸" if pd.isna(example) else f"'{example}'"
        raise ValueError(f"숫자로 읽을 수 없는 값: {', '.join(bad)} (데이터 행 {rows.tolist()}, 예: {shown}). "
                         "천 단위 쉼표·단위 글자·빈 칸·무한대를 확인하세요.")
    return result
