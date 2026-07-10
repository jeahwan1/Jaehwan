"""과거 시즌 전체를 재수집해 고장났던 투수 피처(x14,x15,x16,x17,x27~x34)를 복구한다.

_is_day_complete()가 결과만 채워지면 그 날짜를 다시 수집하지 않으므로,
kbo_client.py 버그 수정 이후 저장된 기본값(4.5, 1.40 등)이 영구히 남아있었다.
이 스크립트는 완료 여부와 상관없이 시즌 시작일부터 전체를 강제로 재수집한다.
"""
from __future__ import annotations

import datetime as dt
import sys

sys.path.insert(0, ".")

from kbo_analyzer.predictor import GamePredictor  # noqa: E402


def main() -> None:
    p = GamePredictor(db_path="kbo_predictor.db")
    with p._connect() as con:
        dates = [r[0] for r in con.execute("SELECT DISTINCT game_date FROM features ORDER BY game_date").fetchall()]

    print(f"재수집 대상: {len(dates)}일 ({dates[0]} ~ {dates[-1]})", flush=True)
    for i, d in enumerate(dates, 1):
        day = dt.date.fromisoformat(d)
        try:
            p._ingest_day(day, include_label=True)
            print(f"[{i}/{len(dates)}] {d} 완료", flush=True)
        except Exception as e:
            print(f"[{i}/{len(dates)}] {d} 실패: {e}", flush=True)


if __name__ == "__main__":
    main()
