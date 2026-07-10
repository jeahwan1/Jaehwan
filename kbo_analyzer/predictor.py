from __future__ import annotations

import datetime as dt
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from .kbo_client import KBORecordClient
from .naver_client import NaverSportsClient
import requests

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression as SKLearnLR
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# 승패 예측 모델에 실제로 넣는 피처(0-indexed, x1..x38 배열 기준).
# 38개 전부 넣으면 학습표본(~300경기)에 비해 피처가 너무 많아 과적합됨 —
# walk-forward 백테스트(전 시즌 대상, 날짜별로 그 이전 경기만으로 학습)로
# 전진선택(forward selection)해 찾은 조합. 정확도 53.3%->63.8%, Brier 0.255->0.236 확인.
# (x2,x3=타선점수, x14,x15=선발시즌ERA, x16=홈선발최근폼ERA, x18,x19=홈/원정전용승률,
#  x20=상대전적, x22=원정투수진최근2경기실점, x23=홈타선최고매치업점수)
_MODEL_FEATURE_IDX = [0, 1, 2, 3, 4, 13, 14, 17, 18, 19, 21, 15, 22]


@dataclass
class FeatureRow:
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    x1_home_adv: float
    x2_home_attack: float
    x3_away_attack: float
    x4_recent_home_winrate: float
    x5_recent_away_winrate: float
    x6_home_recent_runs: float
    x7_away_recent_runs: float
    x8_home_recent_ra: float
    x9_away_recent_ra: float
    x10_park_factor: float
    x11_temp_c: float
    x12_wind_kph: float
    x13_precip_mm: float
    x14_home_starter_era: float
    x15_away_starter_era: float
    x16_home_starter_recent_era: float
    x17_away_starter_recent_era: float
    x18_home_team_home_wr: float
    x19_away_team_away_wr: float
    x20_h2h_home_winrate: float
    x21_home_recent_ra2: float
    x22_away_recent_ra2: float
    x23_home_attack_max: float
    x24_away_attack_max: float
    x25_home_high_count: float
    x26_away_high_count: float
    x27_home_pitcher_whip: float
    x28_away_pitcher_whip: float
    x29_home_pitcher_avg_ip: float
    x30_away_pitcher_avg_ip: float
    x31_home_pitcher_k9: float
    x32_away_pitcher_k9: float
    x33_home_pitcher_bb9: float
    x34_away_pitcher_bb9: float
    x35_away_consec_away: float
    x36_home_consec_home: float
    x37_home_b2b: float
    x38_away_b2b: float
    lineup_collected_at: str
    lineup_confirmed: int
    lineup_source_note: str
    y_home_win: int | None
    home_score: int | None
    away_score: int | None


class GamePredictor:
    def __init__(self, db_path: str = "kbo_predictor.db") -> None:
        self.db_path = db_path
        self.naver = NaverSportsClient()
        self.kbo = KBORecordClient()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    game_id TEXT PRIMARY KEY,
                    game_date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    x1_home_adv REAL NOT NULL,
                    x2_home_attack REAL NOT NULL,
                    x3_away_attack REAL NOT NULL,
                    x4_recent_home_winrate REAL NOT NULL,
                    x5_recent_away_winrate REAL NOT NULL,
                    x6_home_recent_runs REAL NOT NULL DEFAULT 0.0,
                    x7_away_recent_runs REAL NOT NULL DEFAULT 0.0,
                    x8_home_recent_ra REAL NOT NULL DEFAULT 0.0,
                    x9_away_recent_ra REAL NOT NULL DEFAULT 0.0,
                    x10_park_factor REAL NOT NULL DEFAULT 1.0,
                    x11_temp_c REAL NOT NULL DEFAULT 18.0,
                    x12_wind_kph REAL NOT NULL DEFAULT 5.0,
                    x13_precip_mm REAL NOT NULL DEFAULT 0.0,
                    x14_home_starter_era REAL NOT NULL DEFAULT 4.5,
                    x15_away_starter_era REAL NOT NULL DEFAULT 4.5,
                    x16_home_starter_recent_era REAL NOT NULL DEFAULT 4.5,
                    x17_away_starter_recent_era REAL NOT NULL DEFAULT 4.5,
                    x18_home_team_home_wr REAL NOT NULL DEFAULT 0.5,
                    x19_away_team_away_wr REAL NOT NULL DEFAULT 0.5,
                    x20_h2h_home_winrate REAL NOT NULL DEFAULT 0.5,
                    x21_home_recent_ra2 REAL NOT NULL DEFAULT 4.5,
                    x22_away_recent_ra2 REAL NOT NULL DEFAULT 4.5,
                    x23_home_attack_max REAL NOT NULL DEFAULT 0.26,
                    x24_away_attack_max REAL NOT NULL DEFAULT 0.26,
                    x25_home_high_count REAL NOT NULL DEFAULT 0.0,
                    x26_away_high_count REAL NOT NULL DEFAULT 0.0,
                    x27_home_pitcher_whip REAL NOT NULL DEFAULT 1.40,
                    x28_away_pitcher_whip REAL NOT NULL DEFAULT 1.40,
                    x29_home_pitcher_avg_ip REAL NOT NULL DEFAULT 5.5,
                    x30_away_pitcher_avg_ip REAL NOT NULL DEFAULT 5.5,
                    x31_home_pitcher_k9 REAL NOT NULL DEFAULT 7.0,
                    x32_away_pitcher_k9 REAL NOT NULL DEFAULT 7.0,
                    x33_home_pitcher_bb9 REAL NOT NULL DEFAULT 3.5,
                    x34_away_pitcher_bb9 REAL NOT NULL DEFAULT 3.5,
                    x35_away_consec_away REAL NOT NULL DEFAULT 0.0,
                    x36_home_consec_home REAL NOT NULL DEFAULT 0.0,
                    x37_home_b2b REAL NOT NULL DEFAULT 0.0,
                    x38_away_b2b REAL NOT NULL DEFAULT 0.0,
                    lineup_collected_at TEXT,
                    lineup_confirmed INTEGER NOT NULL DEFAULT 0,
                    lineup_source_note TEXT NOT NULL DEFAULT '최근 라인업 기준',
                    y_home_win INTEGER
                    ,
                    home_score INTEGER,
                    away_score INTEGER
                )
                """
            )
            self._migrate_columns(con)
            con.commit()

    def _migrate_columns(self, con: sqlite3.Connection) -> None:
        cols = {r[1] for r in con.execute("PRAGMA table_info(features)").fetchall()}
        wanted = {
            "x6_home_recent_runs": "REAL NOT NULL DEFAULT 0.0",
            "x7_away_recent_runs": "REAL NOT NULL DEFAULT 0.0",
            "x8_home_recent_ra": "REAL NOT NULL DEFAULT 0.0",
            "x9_away_recent_ra": "REAL NOT NULL DEFAULT 0.0",
            "x10_park_factor": "REAL NOT NULL DEFAULT 1.0",
            "x11_temp_c": "REAL NOT NULL DEFAULT 18.0",
            "x12_wind_kph": "REAL NOT NULL DEFAULT 5.0",
            "x13_precip_mm": "REAL NOT NULL DEFAULT 0.0",
            "x14_home_starter_era": "REAL NOT NULL DEFAULT 4.5",
            "x15_away_starter_era": "REAL NOT NULL DEFAULT 4.5",
            "x16_home_starter_recent_era": "REAL NOT NULL DEFAULT 4.5",
            "x17_away_starter_recent_era": "REAL NOT NULL DEFAULT 4.5",
            "x18_home_team_home_wr": "REAL NOT NULL DEFAULT 0.5",
            "x19_away_team_away_wr": "REAL NOT NULL DEFAULT 0.5",
            "x20_h2h_home_winrate": "REAL NOT NULL DEFAULT 0.5",
            "x21_home_recent_ra2": "REAL NOT NULL DEFAULT 4.5",
            "x22_away_recent_ra2": "REAL NOT NULL DEFAULT 4.5",
            "x23_home_attack_max": "REAL NOT NULL DEFAULT 0.26",
            "x24_away_attack_max": "REAL NOT NULL DEFAULT 0.26",
            "x25_home_high_count": "REAL NOT NULL DEFAULT 0.0",
            "x26_away_high_count": "REAL NOT NULL DEFAULT 0.0",
            "x27_home_pitcher_whip": "REAL NOT NULL DEFAULT 1.40",
            "x28_away_pitcher_whip": "REAL NOT NULL DEFAULT 1.40",
            "x29_home_pitcher_avg_ip": "REAL NOT NULL DEFAULT 5.5",
            "x30_away_pitcher_avg_ip": "REAL NOT NULL DEFAULT 5.5",
            "x31_home_pitcher_k9": "REAL NOT NULL DEFAULT 7.0",
            "x32_away_pitcher_k9": "REAL NOT NULL DEFAULT 7.0",
            "x33_home_pitcher_bb9": "REAL NOT NULL DEFAULT 3.5",
            "x34_away_pitcher_bb9": "REAL NOT NULL DEFAULT 3.5",
            "x35_away_consec_away": "REAL NOT NULL DEFAULT 0.0",
            "x36_home_consec_home": "REAL NOT NULL DEFAULT 0.0",
            "x37_home_b2b": "REAL NOT NULL DEFAULT 0.0",
            "x38_away_b2b": "REAL NOT NULL DEFAULT 0.0",
            "lineup_collected_at": "TEXT",
            "lineup_confirmed": "INTEGER NOT NULL DEFAULT 0",
            "lineup_source_note": "TEXT NOT NULL DEFAULT '최근 라인업 기준'",
            "home_score": "INTEGER",
            "away_score": "INTEGER",
        }
        for col, ddl in wanted.items():
            if col not in cols:
                con.execute(f"ALTER TABLE features ADD COLUMN {col} {ddl}")

    def backfill(self, days: int = 30, end_date: dt.date | None = None) -> None:
        # 오늘은 ingest_today_for_predict 에서만 처리 — backfill은 어제까지
        end = end_date or (dt.date.today() - dt.timedelta(days=1))
        start = end - dt.timedelta(days=days)
        for i in range((end - start).days + 1):
            day = start + dt.timedelta(days=i)
            # 3일 이상 지난 날짜이고 결과가 완전히 채워진 경우 스킵
            if (dt.date.today() - day).days > 3 and self._is_day_complete(day):
                continue
            self._ingest_day(day, include_label=True)

    def _is_day_complete(self, date: dt.date) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*), SUM(CASE WHEN y_home_win IS NOT NULL THEN 1 ELSE 0 END) "
                "FROM features WHERE game_date = ?",
                (date.isoformat(),),
            ).fetchone()
        total, filled = row[0], (row[1] or 0)
        return total > 0 and total == filled

    def ingest_today_for_predict(self, date: dt.date | None = None) -> None:
        self._ingest_day(date or dt.date.today(), include_label=False, confirmed_only=True)

    def _ingest_day(self, date: dt.date, include_label: bool, confirmed_only: bool = False) -> None:
        try:
            game_urls = self.naver.fetch_today_game_urls(date)
        except Exception as e:
            print(f"[WARN] Skip {date.isoformat()} ingest: {e}")
            return
        games = {str(g["G_ID"]): g for g in self.naver.get_cached_games()}
        rows: list[FeatureRow] = []
        wx_cache: dict[str, dict[str, float]] = {}

        for game_url in game_urls:
            gid = game_url.split("/")[-1]
            g = games.get(gid)
            if not g:
                continue
            lineup_confirmed_flag = bool(g.get("LINEUP_CK"))
            if confirmed_only and not lineup_confirmed_flag:
                home_nm = str(g.get("HOME_NM", "")).strip()
                away_nm = str(g.get("AWAY_NM", "")).strip()
                print(f"[SKIP] {away_nm} @ {home_nm} — 라인업 미확정, 건너뜀")
                continue
            lineup = self.naver.fetch_game_lineup(game_url)
            home_attack, home_attack_max, home_high_count = self._team_attack_features(
                pitcher_name=lineup.away.starter_pitcher,
                pitcher_team=lineup.away.team,
                batter_team=lineup.home.team,
                batters=lineup.home.batters,
            )
            away_attack, away_attack_max, away_high_count = self._team_attack_features(
                pitcher_name=lineup.home.starter_pitcher,
                pitcher_team=lineup.home.team,
                batter_team=lineup.away.team,
                batters=lineup.away.batters,
            )
            # 라인업 미확정 시 타선 점수를 리그 평균(0.26) 쪽으로 50% 보정
            # → API가 반환하는 "최근 라인업"이 실제 출전 선수와 다를 수 있음
            if not lineup_confirmed_flag:
                home_attack = home_attack * 0.5 + 0.26 * 0.5
                away_attack = away_attack * 0.5 + 0.26 * 0.5
                home_attack_max = home_attack_max * 0.5 + 0.26 * 0.5
                away_attack_max = away_attack_max * 0.5 + 0.26 * 0.5
                home_high_count *= 0.5
                away_high_count *= 0.5

            home_p_season = self.kbo.fetch_pitcher_season_stats(lineup.home.starter_pitcher, lineup.home.team)
            away_p_season = self.kbo.fetch_pitcher_season_stats(lineup.away.starter_pitcher, lineup.away.team)
            home_p_form = self.kbo.fetch_pitcher_recent_form(lineup.home.starter_pitcher, lineup.home.team)
            away_p_form = self.kbo.fetch_pitcher_recent_form(lineup.away.starter_pitcher, lineup.away.team)

            home_recent = self._recent_winrate(lineup.home.team, date, n=5)
            away_recent = self._recent_winrate(lineup.away.team, date, n=5)
            home_recent_runs, home_recent_ra = self._recent_runs(lineup.home.team, date, n=5)
            away_recent_runs, away_recent_ra = self._recent_runs(lineup.away.team, date, n=5)
            home_home_wr = self._venue_winrate(lineup.home.team, date, venue="home", n=10)
            away_away_wr = self._venue_winrate(lineup.away.team, date, venue="away", n=10)
            h2h_wr = self._h2h_winrate(lineup.home.team, lineup.away.team, date)
            _, home_recent_ra2 = self._recent_runs(lineup.home.team, date, n=2)
            _, away_recent_ra2 = self._recent_runs(lineup.away.team, date, n=2)
            away_consec = self._consec_venue(lineup.away.team, date, "away")
            home_consec = self._consec_venue(lineup.home.team, date, "home")
            home_b2b = self._back_to_back(lineup.home.team, date)
            away_b2b = self._back_to_back(lineup.away.team, date)
            stadium = str(g.get("S_NM", ""))
            park_factor = _park_factor(stadium)
            if stadium not in wx_cache:
                wx_cache[stadium] = _fetch_weather_for_stadium(stadium)
            wx = wx_cache[stadium]

            y = None
            hs: int | None = None
            aw: int | None = None
            if include_label:
                try:
                    hs = int(str(g.get("B_SCORE_CN", "0")).strip())
                    aw = int(str(g.get("T_SCORE_CN", "0")).strip())
                    if str(g.get("GAME_RESULT_CK", "0")) == "1":
                        y = 1 if hs > aw else 0
                except ValueError:
                    y = None

            lineup_confirmed = 1 if bool(g.get("LINEUP_CK")) else 0
            lineup_note = "확정 라인업" if lineup_confirmed == 1 else "최근 라인업 기준"
            rows.append(
                FeatureRow(
                    game_id=gid,
                    game_date=date.isoformat(),
                    home_team=lineup.home.team,
                    away_team=lineup.away.team,
                    x1_home_adv=1.0,
                    x2_home_attack=home_attack,
                    x3_away_attack=away_attack,
                    x4_recent_home_winrate=home_recent,
                    x5_recent_away_winrate=away_recent,
                    x6_home_recent_runs=home_recent_runs,
                    x7_away_recent_runs=away_recent_runs,
                    x8_home_recent_ra=home_recent_ra,
                    x9_away_recent_ra=away_recent_ra,
                    x10_park_factor=park_factor,
                    x11_temp_c=wx["temp_c"],
                    x12_wind_kph=wx["wind_kph"],
                    x13_precip_mm=wx["precip_mm"],
                    x14_home_starter_era=home_p_season.era,
                    x15_away_starter_era=away_p_season.era,
                    x16_home_starter_recent_era=home_p_form.era,
                    x17_away_starter_recent_era=away_p_form.era,
                    x18_home_team_home_wr=home_home_wr,
                    x19_away_team_away_wr=away_away_wr,
                    x20_h2h_home_winrate=h2h_wr,
                    x21_home_recent_ra2=home_recent_ra2,
                    x22_away_recent_ra2=away_recent_ra2,
                    x23_home_attack_max=home_attack_max,
                    x24_away_attack_max=away_attack_max,
                    x25_home_high_count=home_high_count,
                    x26_away_high_count=away_high_count,
                    x27_home_pitcher_whip=home_p_season.whip,
                    x28_away_pitcher_whip=away_p_season.whip,
                    x29_home_pitcher_avg_ip=_avg_ip(home_p_season.ip, home_p_season.gs),
                    x30_away_pitcher_avg_ip=_avg_ip(away_p_season.ip, away_p_season.gs),
                    x31_home_pitcher_k9=home_p_season.k9,
                    x32_away_pitcher_k9=away_p_season.k9,
                    x33_home_pitcher_bb9=home_p_season.bb9,
                    x34_away_pitcher_bb9=away_p_season.bb9,
                    x35_away_consec_away=away_consec,
                    x36_home_consec_home=home_consec,
                    x37_home_b2b=home_b2b,
                    x38_away_b2b=away_b2b,
                    lineup_collected_at=dt.datetime.now().isoformat(),
                    lineup_confirmed=lineup_confirmed,
                    lineup_source_note=lineup_note,
                    y_home_win=y,
                    home_score=hs,
                    away_score=aw,
                )
            )

        self._warn_on_stale_pitcher_stats(date, rows)

        with self._connect() as con:
            for r in rows:
                con.execute(
                    """
                    INSERT INTO features (
                      game_id, game_date, home_team, away_team,
                      x1_home_adv, x2_home_attack, x3_away_attack,
                      x4_recent_home_winrate, x5_recent_away_winrate,
                      x6_home_recent_runs, x7_away_recent_runs, x8_home_recent_ra, x9_away_recent_ra,
                      x10_park_factor, x11_temp_c, x12_wind_kph, x13_precip_mm,
                      x14_home_starter_era, x15_away_starter_era,
                      x16_home_starter_recent_era, x17_away_starter_recent_era,
                      x18_home_team_home_wr, x19_away_team_away_wr, x20_h2h_home_winrate,
                      x21_home_recent_ra2, x22_away_recent_ra2,
                      x23_home_attack_max, x24_away_attack_max,
                      x25_home_high_count, x26_away_high_count,
                      x27_home_pitcher_whip, x28_away_pitcher_whip,
                      x29_home_pitcher_avg_ip, x30_away_pitcher_avg_ip,
                      x31_home_pitcher_k9, x32_away_pitcher_k9,
                      x33_home_pitcher_bb9, x34_away_pitcher_bb9,
                      x35_away_consec_away, x36_home_consec_home,
                      x37_home_b2b, x38_away_b2b,
                      lineup_collected_at, lineup_confirmed, lineup_source_note,
                      y_home_win, home_score, away_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                      game_date=excluded.game_date,
                      home_team=excluded.home_team,
                      away_team=excluded.away_team,
                      x1_home_adv=excluded.x1_home_adv,
                      x2_home_attack=excluded.x2_home_attack,
                      x3_away_attack=excluded.x3_away_attack,
                      x4_recent_home_winrate=excluded.x4_recent_home_winrate,
                      x5_recent_away_winrate=excluded.x5_recent_away_winrate,
                      x6_home_recent_runs=excluded.x6_home_recent_runs,
                      x7_away_recent_runs=excluded.x7_away_recent_runs,
                      x8_home_recent_ra=excluded.x8_home_recent_ra,
                      x9_away_recent_ra=excluded.x9_away_recent_ra,
                      x10_park_factor=excluded.x10_park_factor,
                      x11_temp_c=excluded.x11_temp_c,
                      x12_wind_kph=excluded.x12_wind_kph,
                      x13_precip_mm=excluded.x13_precip_mm,
                      x14_home_starter_era=excluded.x14_home_starter_era,
                      x15_away_starter_era=excluded.x15_away_starter_era,
                      x16_home_starter_recent_era=excluded.x16_home_starter_recent_era,
                      x17_away_starter_recent_era=excluded.x17_away_starter_recent_era,
                      x18_home_team_home_wr=excluded.x18_home_team_home_wr,
                      x19_away_team_away_wr=excluded.x19_away_team_away_wr,
                      x20_h2h_home_winrate=excluded.x20_h2h_home_winrate,
                      x21_home_recent_ra2=excluded.x21_home_recent_ra2,
                      x22_away_recent_ra2=excluded.x22_away_recent_ra2,
                      x23_home_attack_max=excluded.x23_home_attack_max,
                      x24_away_attack_max=excluded.x24_away_attack_max,
                      x25_home_high_count=excluded.x25_home_high_count,
                      x26_away_high_count=excluded.x26_away_high_count,
                      x27_home_pitcher_whip=excluded.x27_home_pitcher_whip,
                      x28_away_pitcher_whip=excluded.x28_away_pitcher_whip,
                      x29_home_pitcher_avg_ip=excluded.x29_home_pitcher_avg_ip,
                      x30_away_pitcher_avg_ip=excluded.x30_away_pitcher_avg_ip,
                      x31_home_pitcher_k9=excluded.x31_home_pitcher_k9,
                      x32_away_pitcher_k9=excluded.x32_away_pitcher_k9,
                      x33_home_pitcher_bb9=excluded.x33_home_pitcher_bb9,
                      x34_away_pitcher_bb9=excluded.x34_away_pitcher_bb9,
                      x35_away_consec_away=excluded.x35_away_consec_away,
                      x36_home_consec_home=excluded.x36_home_consec_home,
                      x37_home_b2b=excluded.x37_home_b2b,
                      x38_away_b2b=excluded.x38_away_b2b,
                      lineup_collected_at=excluded.lineup_collected_at,
                      lineup_confirmed=excluded.lineup_confirmed,
                      lineup_source_note=excluded.lineup_source_note,
                      y_home_win=COALESCE(excluded.y_home_win, features.y_home_win),
                      home_score=COALESCE(excluded.home_score, features.home_score),
                      away_score=COALESCE(excluded.away_score, features.away_score)
                    """,
                    (
                        r.game_id,
                        r.game_date,
                        r.home_team,
                        r.away_team,
                        r.x1_home_adv,
                        r.x2_home_attack,
                        r.x3_away_attack,
                        r.x4_recent_home_winrate,
                        r.x5_recent_away_winrate,
                        r.x6_home_recent_runs,
                        r.x7_away_recent_runs,
                        r.x8_home_recent_ra,
                        r.x9_away_recent_ra,
                        r.x10_park_factor,
                        r.x11_temp_c,
                        r.x12_wind_kph,
                        r.x13_precip_mm,
                        r.x14_home_starter_era,
                        r.x15_away_starter_era,
                        r.x16_home_starter_recent_era,
                        r.x17_away_starter_recent_era,
                        r.x18_home_team_home_wr,
                        r.x19_away_team_away_wr,
                        r.x20_h2h_home_winrate,
                        r.x21_home_recent_ra2,
                        r.x22_away_recent_ra2,
                        r.x23_home_attack_max,
                        r.x24_away_attack_max,
                        r.x25_home_high_count,
                        r.x26_away_high_count,
                        r.x27_home_pitcher_whip,
                        r.x28_away_pitcher_whip,
                        r.x29_home_pitcher_avg_ip,
                        r.x30_away_pitcher_avg_ip,
                        r.x31_home_pitcher_k9,
                        r.x32_away_pitcher_k9,
                        r.x33_home_pitcher_bb9,
                        r.x34_away_pitcher_bb9,
                        r.x35_away_consec_away,
                        r.x36_home_consec_home,
                        r.x37_home_b2b,
                        r.x38_away_b2b,
                        r.lineup_collected_at,
                        r.lineup_confirmed,
                        r.lineup_source_note,
                        r.y_home_win,
                        r.home_score,
                        r.away_score,
                    ),
                )
            con.commit()

    def _warn_on_stale_pitcher_stats(self, date: dt.date, rows: list["FeatureRow"]) -> None:
        """선발투수 스탯 조회가 조용히 리그평균 기본값으로 떨어지는 걸 감지해 로그로 남긴다.
        과거에 이 실패가 눈에 안 띄어서 시즌 절반 가까이 x14/x27 등이 기본값으로 고정된 적 있음."""
        if not rows:
            return
        defaulted = sum(
            1 for r in rows
            if math.isclose(r.x14_home_starter_era, 4.5) or math.isclose(r.x15_away_starter_era, 4.5)
        )
        if defaulted / len(rows) >= 0.3:
            print(
                f"[WARN] {date.isoformat()}: 선발 시즌 ERA가 리그평균 기본값으로 떨어진 경기 "
                f"{defaulted}/{len(rows)}건 — KBO 선수 스탯 조회 실패 가능성, kbo_client.py 점검 필요"
            )

    def _team_attack_features(
        self, pitcher_name: str, pitcher_team: str, batter_team: str, batters: list
    ) -> tuple[float, float, float]:
        """Returns (avg_score, max_score, high_count) based on pitcher-batter hand matchups."""
        rows = self._batter_split_rows(pitcher_name, pitcher_team, batter_team, batters)
        scores = [r["score"] for r in rows]
        if not scores:
            return 0.25, 0.25, 0.0
        avg = sum(scores) / len(scores)
        mx = max(scores)
        high = float(sum(1 for s in scores if s >= 0.300))
        return avg, mx, high

    def _batter_split_rows(
        self, pitcher_name: str, pitcher_team: str, batter_team: str, batters: list
    ) -> list[dict]:
        p_split = self.kbo.fetch_pitcher_split(pitcher_name, pitcher_team)
        rows: list[dict] = []
        for b in batters[:9]:
            h_split = self.kbo.fetch_hitter_split(b.name, batter_team)
            batter_vs_hand = h_split.vs_left_avg if p_split.hand == "L" else h_split.vs_right_avg
            pitcher_vs_hand = (
                p_split.vs_left_avg_allowed if h_split.hand == "L" else p_split.vs_right_avg_allowed
            )
            score = 0.6 * batter_vs_hand + 0.4 * pitcher_vs_hand
            rows.append({
                "order": b.order,
                "name": b.name,
                "batter_hand": h_split.hand,
                "pitcher_hand": p_split.hand,
                "batter_vs_pitcher_hand": round(batter_vs_hand, 3),
                "pitcher_vs_batter_hand": round(pitcher_vs_hand, 3),
                "score": round(score, 3),
                "band": "HIGH" if score >= 0.300 else ("MID" if score >= 0.260 else "LOW"),
            })
        return rows

    def _attach_h2h(
        self, rows: list[dict], pitcher_name: str, pitcher_team: str, batter_team: str
    ) -> None:
        for row in rows:
            stat = self.kbo.fetch_matchup(
                pitcher_name=pitcher_name, batter_name=row["name"],
                pitcher_team=pitcher_team, batter_team=batter_team,
            )
            row["h2h_ab"] = stat.ab
            row["h2h_hits"] = stat.hits
            row["h2h_hr"] = stat.hr
            row["h2h_avg"] = round(stat.avg, 3) if stat.ab > 0 else None

    def build_today_matchup_report(self, date: dt.date | None = None) -> list[dict]:
        """오늘 확정 경기의 선발투수 vs 타자 H2H + 스플릿 데이터 반환."""
        target = date or dt.date.today()
        try:
            game_urls = self.naver.fetch_today_game_urls(target)
        except Exception:
            return []
        games_map = {str(g["G_ID"]): g for g in self.naver.get_cached_games()}
        result: list[dict] = []

        for game_url in game_urls:
            gid = game_url.split("/")[-1]
            g = games_map.get(gid)
            if not g or not bool(g.get("LINEUP_CK")):
                continue
            try:
                lineup = self.naver.fetch_game_lineup(game_url)
            except Exception:
                continue

            print(f"[MATCHUP] {lineup.away.team} @ {lineup.home.team} 상대전적 조회 중...")
            home_rows = self._batter_split_rows(
                lineup.away.starter_pitcher, lineup.away.team,
                lineup.home.team, lineup.home.batters,
            )
            away_rows = self._batter_split_rows(
                lineup.home.starter_pitcher, lineup.home.team,
                lineup.away.team, lineup.away.batters,
            )
            self._attach_h2h(home_rows, lineup.away.starter_pitcher, lineup.away.team, lineup.home.team)
            self._attach_h2h(away_rows, lineup.home.starter_pitcher, lineup.home.team, lineup.away.team)

            result.append({
                "game_id": gid,
                "away_team": lineup.away.team,
                "home_team": lineup.home.team,
                "away_pitcher": lineup.away.starter_pitcher,
                "home_pitcher": lineup.home.starter_pitcher,
                "home_batters_vs_away_pitcher": home_rows,
                "away_batters_vs_home_pitcher": away_rows,
            })
        return result

    def _recent_winrate(self, team: str, before_date: dt.date, n: int = 5) -> float:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT home_team, away_team, y_home_win
                FROM features
                WHERE game_date < ?
                  AND y_home_win IS NOT NULL
                  AND (home_team = ? OR away_team = ?)
                ORDER BY game_date DESC
                LIMIT ?
                """,
                (before_date.isoformat(), team, team, n),
            ).fetchall()
        wins = 0
        for home_team, away_team, y_home_win in rows:
            if home_team == team and y_home_win == 1:
                wins += 1
            elif away_team == team and y_home_win == 0:
                wins += 1
        if not rows:
            return 0.5
        return wins / len(rows)

    def _recent_runs(self, team: str, before_date: dt.date, n: int = 5) -> tuple[float, float]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT home_team, away_team, home_score, away_score
                FROM features
                WHERE game_date < ?
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                  AND (home_team = ? OR away_team = ?)
                ORDER BY game_date DESC
                LIMIT ?
                """,
                (before_date.isoformat(), team, team, n),
            ).fetchall()
        rs: list[int] = []
        ra: list[int] = []
        for home_team, away_team, home_score, away_score in rows:
            if home_team == team:
                rs.append(int(home_score))
                ra.append(int(away_score))
            else:
                rs.append(int(away_score))
                ra.append(int(home_score))
        if not rs:
            return 4.5, 4.5
        return sum(rs) / len(rs), sum(ra) / len(ra)

    def _venue_winrate(self, team: str, before_date: dt.date, venue: str, n: int = 10) -> float:
        """venue='home' → 홈 경기만, venue='away' → 원정 경기만 승률"""
        col = "home_team" if venue == "home" else "away_team"
        win_expr = "y_home_win = 1" if venue == "home" else "y_home_win = 0"
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT COUNT(*), SUM(CASE WHEN {win_expr} THEN 1 ELSE 0 END)
                FROM features
                WHERE game_date < ? AND {col} = ? AND y_home_win IS NOT NULL
                ORDER BY game_date DESC
                LIMIT ?
                """,
                (before_date.isoformat(), team, n),
            ).fetchone()
        total, wins = rows[0], (rows[1] or 0)
        return wins / total if total > 0 else 0.5

    def _h2h_winrate(self, home_team: str, away_team: str, before_date: dt.date) -> float:
        """올시즌 두 팀 간 상대전적 기반 home_team 승률 (최소 2경기 이상)"""
        season_start = dt.date(before_date.year, 3, 1)
        with self._connect() as con:
            # home_team이 홈일 때
            r1 = con.execute(
                """
                SELECT COUNT(*), SUM(y_home_win) FROM features
                WHERE game_date >= ? AND game_date < ?
                  AND home_team=? AND away_team=? AND y_home_win IS NOT NULL
                """,
                (season_start.isoformat(), before_date.isoformat(), home_team, away_team),
            ).fetchone()
            # home_team이 원정일 때
            r2 = con.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN y_home_win=0 THEN 1 ELSE 0 END) FROM features
                WHERE game_date >= ? AND game_date < ?
                  AND home_team=? AND away_team=? AND y_home_win IS NOT NULL
                """,
                (season_start.isoformat(), before_date.isoformat(), away_team, home_team),
            ).fetchone()
        total = (r1[0] or 0) + (r2[0] or 0)
        wins = (r1[1] or 0) + (r2[1] or 0)
        return wins / total if total >= 2 else 0.5

    def _consec_venue(self, team: str, before_date: dt.date, venue: str) -> float:
        """연속 홈(venue='home') 또는 연속 원정(venue='away') 경기 수."""
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT home_team, away_team FROM features
                WHERE game_date < ?
                  AND (home_team = ? OR away_team = ?)
                ORDER BY game_date DESC
                LIMIT 15
                """,
                (before_date.isoformat(), team, team),
            ).fetchall()
        count = 0.0
        for home_t, away_t in rows:
            if venue == "home" and home_t == team:
                count += 1
            elif venue == "away" and away_t == team:
                count += 1
            else:
                break
        return count

    def _back_to_back(self, team: str, before_date: dt.date) -> float:
        """전날 경기 여부 (1.0=있음, 0.0=없음)."""
        yesterday = (before_date - dt.timedelta(days=1)).isoformat()
        with self._connect() as con:
            count = con.execute(
                "SELECT COUNT(*) FROM features WHERE game_date = ? AND (home_team = ? OR away_team = ?)",
                (yesterday, team, team),
            ).fetchone()[0]
        return 1.0 if count > 0 else 0.0

    def train_and_predict_today(self, date: dt.date | None = None) -> list[dict]:
        d = (date or dt.date.today()).isoformat()
        with self._connect() as con:
            train = con.execute(
                """
                SELECT x1_home_adv, x2_home_attack, x3_away_attack,
                       x4_recent_home_winrate, x5_recent_away_winrate,
                       x6_home_recent_runs, x7_away_recent_runs, x8_home_recent_ra, x9_away_recent_ra,
                       x10_park_factor, x11_temp_c, x12_wind_kph, x13_precip_mm,
                       x14_home_starter_era, x15_away_starter_era,
                       x16_home_starter_recent_era, x17_away_starter_recent_era,
                       x18_home_team_home_wr, x19_away_team_away_wr, x20_h2h_home_winrate,
                       x21_home_recent_ra2, x22_away_recent_ra2,
                       x23_home_attack_max, x24_away_attack_max,
                       x25_home_high_count, x26_away_high_count,
                       x27_home_pitcher_whip, x28_away_pitcher_whip,
                       x29_home_pitcher_avg_ip, x30_away_pitcher_avg_ip,
                       x31_home_pitcher_k9, x32_away_pitcher_k9,
                       x33_home_pitcher_bb9, x34_away_pitcher_bb9,
                       x35_away_consec_away, x36_home_consec_home,
                       x37_home_b2b, x38_away_b2b,
                       y_home_win
                FROM features
                WHERE y_home_win IS NOT NULL
                ORDER BY game_date
                """
            ).fetchall()
            today = con.execute(
                """
                SELECT game_id, home_team, away_team,
                       x1_home_adv, x2_home_attack, x3_away_attack,
                       x4_recent_home_winrate, x5_recent_away_winrate,
                       x6_home_recent_runs, x7_away_recent_runs, x8_home_recent_ra, x9_away_recent_ra,
                       x10_park_factor, x11_temp_c, x12_wind_kph, x13_precip_mm,
                       x14_home_starter_era, x15_away_starter_era,
                       x16_home_starter_recent_era, x17_away_starter_recent_era,
                       x18_home_team_home_wr, x19_away_team_away_wr, x20_h2h_home_winrate,
                       x21_home_recent_ra2, x22_away_recent_ra2,
                       x23_home_attack_max, x24_away_attack_max,
                       x25_home_high_count, x26_away_high_count,
                       x27_home_pitcher_whip, x28_away_pitcher_whip,
                       x29_home_pitcher_avg_ip, x30_away_pitcher_avg_ip,
                       x31_home_pitcher_k9, x32_away_pitcher_k9,
                       x33_home_pitcher_bb9, x34_away_pitcher_bb9,
                       x35_away_consec_away, x36_home_consec_home,
                       x37_home_b2b, x38_away_b2b,
                       lineup_collected_at, lineup_confirmed, lineup_source_note
                FROM features
                WHERE game_date = ?
                ORDER BY game_id
                """,
                (d,),
            ).fetchall()

        if not today:
            return []

        if len(train) < 20:
            raise RuntimeError("학습 데이터가 부족합니다. backfill 기간을 늘리세요.")

        X_full = [list(t[:38]) for t in train]
        X = [[row[i] for i in _MODEL_FEATURE_IDX] for row in X_full]
        y = [int(t[38]) for t in train]

        if _SKLEARN_AVAILABLE and len(train) >= 40:
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
            if len(train) >= 400:
                # 데이터 충분 시 GBM 사용
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=5, random_state=42,
                )
            else:
                # 소규모 데이터: 강하게 정규화된 LR이 가장 안정적
                model = SKLearnLR(C=0.1, max_iter=2000, random_state=42)
            model.fit(X_norm, y)
            _predict_proba = lambda xrow: float(
                model.predict_proba(scaler.transform([xrow]))[0][1]
            )
        else:
            means, stds = _compute_normalize_params(X)
            X_norm_list = _apply_normalize(X, means, stds)
            w, b = _fit_logistic(X_norm_list, y, lr=0.2, epochs=1200, l2=0.05)
            scaler_params = (means, stds)
            _predict_proba = lambda xrow: _sigmoid(
                _dot(w, _apply_normalize([xrow], *scaler_params)[0]) + b
            )

        out: list[dict] = []
        for row in today:
            gid, home, away, *rest = row
            x = [float(v) for v in rest[:38]]
            lineup_collected_at = rest[38]
            lineup_confirmed = int(rest[39] or 0)
            lineup_source_note = str(rest[40] or "")
            x_model = [x[i] for i in _MODEL_FEATURE_IDX]
            raw_p = _predict_proba(x_model)
            # 스포츠 예측 특성상 70% 초과 확신은 과신 — 로그오즈를 ±0.847로 클램핑
            log_odds = math.log(raw_p / (1 - raw_p))
            p = _sigmoid(_clamp(log_odds, -0.847, 0.847))  # sigmoid(0.847) ≈ 0.70
            home_score_pred, away_score_pred = _project_score(
                home_attack=x[1],
                away_attack=x[2],
                home_recent_runs=x[5],
                away_recent_runs=x[6],
                home_recent_ra=x[7],
                away_recent_ra=x[8],
                park_factor=x[9],
                temp_c=x[10],
                wind_kph=x[11],
                home_win_prob=float(p),
                home_starter_era=x[13],
                away_starter_era=x[14],
                home_bullpen_ra=x[20],
                away_bullpen_ra=x[21],
            )
            out.append(
                {
                    "game_id": gid,
                    "home_team": home,
                    "away_team": away,
                    "home_win_prob": round(p, 4),
                    "away_win_prob": round(1.0 - p, 4),
                    "home_score_pred": home_score_pred,
                    "away_score_pred": away_score_pred,
                    "total_score_pred": round(home_score_pred + away_score_pred, 1),
                    "lineup_collected_at": lineup_collected_at,
                    "lineup_confirmed": bool(lineup_confirmed),
                    "lineup_source_note": lineup_source_note,
                    "signals": {
                        "home_attack": round(x[1], 3),
                        "away_attack": round(x[2], 3),
                        "home_attack_max": round(x[22], 3),
                        "away_attack_max": round(x[23], 3),
                        "home_high_count": int(round(x[24])),
                        "away_high_count": int(round(x[25])),
                        "home_pitcher_whip": round(x[26], 2),
                        "away_pitcher_whip": round(x[27], 2),
                        "home_pitcher_avg_ip": round(x[28], 1),
                        "away_pitcher_avg_ip": round(x[29], 1),
                        "home_pitcher_k9": round(x[30], 2),
                        "away_pitcher_k9": round(x[31], 2),
                        "home_pitcher_bb9": round(x[32], 2),
                        "away_pitcher_bb9": round(x[33], 2),
                        "away_consec_away": int(round(x[34])),
                        "home_consec_home": int(round(x[35])),
                        "home_b2b": bool(x[36]),
                        "away_b2b": bool(x[37]),
                        "home_recent_winrate": round(x[3], 3),
                        "away_recent_winrate": round(x[4], 3),
                        "home_recent_runs": round(x[5], 2),
                        "away_recent_runs": round(x[6], 2),
                        "home_recent_ra": round(x[7], 2),
                        "away_recent_ra": round(x[8], 2),
                        "park_factor": round(x[9], 2),
                        "temp_c": round(x[10], 1),
                        "wind_kph": round(x[11], 1),
                        "precip_mm": round(x[12], 1),
                        "home_starter_era": round(x[13], 2),
                        "away_starter_era": round(x[14], 2),
                        "home_starter_recent_era": round(x[15], 2),
                        "away_starter_recent_era": round(x[16], 2),
                        "home_team_home_wr": round(x[17], 3),
                        "away_team_away_wr": round(x[18], 3),
                        "h2h_home_winrate": round(x[19], 3),
                        "home_bullpen_ra2": round(x[20], 2),
                        "away_bullpen_ra2": round(x[21], 2),
                    },
                    "reasons": _generate_reasons(home, away, float(p), x),
                }
            )
        return out


def _fit_logistic(
    X: list[list[float]], y: list[int], lr: float, epochs: int, l2: float = 0.01
) -> tuple[list[float], float]:
    m = len(X)
    n = len(X[0])
    w = [0.0] * n
    b = 0.0
    for _ in range(epochs):
        dw = [0.0] * n
        db = 0.0
        for xi, yi in zip(X, y):
            z = _dot(w, xi) + b
            p = _sigmoid(z)
            err = p - yi
            for j in range(n):
                dw[j] += err * xi[j]
            db += err
        for j in range(n):
            w[j] -= lr * (dw[j] / m + l2 * w[j])
        b -= lr * (db / m)
    return w, b


def _compute_normalize_params(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(n)]
    stds = []
    for j in range(n):
        var = sum((row[j] - means[j]) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    return means, stds


def _apply_normalize(X: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    n = len(X[0])
    return [[(row[j] - means[j]) / stds[j] for j in range(n)] for row in X]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(z: float) -> float:
    z = max(min(z, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-z))


def _project_score(
    home_attack: float,
    away_attack: float,
    home_recent_runs: float,
    away_recent_runs: float,
    home_recent_ra: float,
    away_recent_ra: float,
    park_factor: float,
    temp_c: float,
    wind_kph: float,
    home_win_prob: float,
    home_starter_era: float = 4.5,
    away_starter_era: float = 4.5,
    home_bullpen_ra: float = 4.5,
    away_bullpen_ra: float = 4.5,
) -> tuple[float, float]:
    # Base offense vs opponent recent run-prevention
    home_mu = 0.55 * home_recent_runs + 0.45 * away_recent_ra
    away_mu = 0.55 * away_recent_runs + 0.45 * home_recent_ra

    # Attack quality adjustment (center around ~0.26)
    home_mu *= _clamp(home_attack / 0.26, 0.75, 1.30)
    away_mu *= _clamp(away_attack / 0.26, 0.75, 1.30)

    # ERA adjustment: KBO league avg ~4.5; high opponent ERA → more runs scored
    # home scores against away_starter, away scores against home_starter
    home_mu *= _clamp(away_starter_era / 4.5, 0.78, 1.28)
    away_mu *= _clamp(home_starter_era / 4.5, 0.78, 1.28)

    # 불펜 피로도: 최근 2경기 실점이 높으면 오늘 7회 이후 실점 증가 가능
    # home 투수진이 허용한 실점 = away_bullpen_ra, away 투수진 = home_bullpen_ra
    away_mu *= _clamp(home_bullpen_ra / 4.5, 0.90, 1.15)
    home_mu *= _clamp(away_bullpen_ra / 4.5, 0.90, 1.15)

    # Park + weather adjustment
    weather_mult = 1.0
    if temp_c >= 27:
        weather_mult += 0.04
    if wind_kph >= 11:
        weather_mult += 0.03
    home_mu *= park_factor * weather_mult
    away_mu *= park_factor * weather_mult

    # Align expected margin with win probability
    current_margin = home_mu - away_mu
    target_margin = (home_win_prob - 0.5) * 4.0  # ~[-2, +2]
    delta = (target_margin - current_margin) * 0.5
    home_mu += delta
    away_mu -= delta

    home_mu = _clamp(home_mu, 1.5, 9.5)
    away_mu = _clamp(away_mu, 1.5, 9.5)
    return round(home_mu, 1), round(away_mu, 1)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _generate_reasons(home: str, away: str, p: float, x: list[float]) -> list[str]:
    # x indices: 0=home_adv, 1=home_attack, 2=away_attack,
    #            3=home_wr, 4=away_wr, 5=home_runs, 6=away_runs,
    #            7=home_ra, 8=away_ra, 9=park_factor,
    #            10=temp_c, 11=wind_kph, 12=precip_mm,
    #            13=home_starter_era, 14=away_starter_era,
    #            15=home_starter_recent_era, 16=away_starter_recent_era,
    #            17=home_team_home_wr, 18=away_team_away_wr, 19=h2h_home_winrate,
    #            20=home_recent_ra2, 21=away_recent_ra2
    candidates: list[tuple[float, str]] = []

    atk_gap = x[1] - x[2]
    if abs(atk_gap) >= 0.015:
        stronger = home if atk_gap > 0 else away
        s_atk, w_atk = (x[1], x[2]) if atk_gap > 0 else (x[2], x[1])
        candidates.append((
            abs(atk_gap) * 10,
            f"{stronger} 타선이 상대 선발 투수 대비 우위 (타선 점수 {s_atk:.3f} vs {w_atk:.3f})",
        ))

    wr_gap = x[3] - x[4]
    if abs(wr_gap) >= 0.08:
        stronger = home if wr_gap > 0 else away
        s_wr, w_wr = (x[3], x[4]) if wr_gap > 0 else (x[4], x[3])
        candidates.append((
            abs(wr_gap) * 5,
            f"{stronger} 최근 5경기 승률 우세 ({s_wr*100:.0f}% vs {w_wr*100:.0f}%)",
        ))

    run_gap = x[5] - x[6]
    if abs(run_gap) >= 0.4:
        stronger = home if run_gap > 0 else away
        s_run, w_run = (x[5], x[6]) if run_gap > 0 else (x[6], x[5])
        candidates.append((
            abs(run_gap),
            f"{stronger} 최근 5경기 평균 득점 우위 ({s_run:.1f}점 vs {w_run:.1f}점)",
        ))

    ra_gap = x[8] - x[7]  # away_ra - home_ra: 클수록 홈 투수진 유리
    if abs(ra_gap) >= 0.4:
        stronger = home if ra_gap > 0 else away
        s_ra, w_ra = (x[7], x[8]) if ra_gap > 0 else (x[8], x[7])
        candidates.append((
            abs(ra_gap) * 0.8,
            f"{stronger} 투수진 최근 실점 억제력 우세 (실점 {s_ra:.1f} vs {w_ra:.1f})",
        ))

    if x[12] > 0.5:
        candidates.append((1.5, f"강수 예보({x[12]:.1f}mm) - 타구 억제 · 투수전 가능성"))
    elif x[10] >= 28:
        candidates.append((1.0, f"고온({x[10]:.0f}°C) - 타자 유리 환경, 고득점 경기 가능"))
    elif x[11] >= 12:
        candidates.append((1.0, f"강풍({x[11]:.0f}kph) - 외야 타구 변수 증가"))

    if x[9] >= 1.05:
        candidates.append((0.8, f"타자 친화 구장(팩터 {x[9]:.2f}) - 고득점 경향"))
    elif x[9] <= 0.95:
        candidates.append((0.8, f"투수 친화 구장(팩터 {x[9]:.2f}) - 저득점 경향"))

    # ERA 우위 (13=홈선발ERA, 14=원정선발ERA; 낮을수록 좋은 투수)
    if len(x) > 14:
        era_gap = x[14] - x[13]  # 양수 = 홈선발이 더 좋음
        if abs(era_gap) >= 0.5:
            better = home if era_gap > 0 else away
            better_era = x[13] if era_gap > 0 else x[14]
            worse_era = x[14] if era_gap > 0 else x[13]
            candidates.append((
                abs(era_gap) * 0.7,
                f"{better} 선발투수 시즌 ERA 우위 ({better_era:.2f} vs {worse_era:.2f})",
            ))

    if len(x) > 16:
        recent_gap = x[16] - x[15]  # 양수 = 홈선발 최근폼 우위
        if abs(recent_gap) >= 1.0:
            better = home if recent_gap > 0 else away
            better_era = x[15] if recent_gap > 0 else x[16]
            candidates.append((
                abs(recent_gap) * 0.5,
                f"{better} 선발 최근 3선발 ERA {better_era:.2f} — 현재 폼 우세",
            ))

    # 홈/원정 분리 승률
    if len(x) > 18:
        venue_gap = x[17] - x[18]  # 홈팀 홈승률 - 원정팀 원정승률
        if abs(venue_gap) >= 0.12:
            better = home if venue_gap > 0 else away
            s_wr = x[17] if venue_gap > 0 else x[18]
            candidates.append((
                abs(venue_gap) * 4,
                f"{better} 홈/원정 전용 승률 우세 ({s_wr*100:.0f}%)",
            ))

    # 시즌 상대전적
    if len(x) > 19:
        h2h_gap = x[19] - 0.5
        if abs(h2h_gap) >= 0.12:
            better = home if h2h_gap > 0 else away
            candidates.append((
                abs(h2h_gap) * 3,
                f"{better} 올시즌 상대전적 우세 ({x[19]*100:.0f}%)",
            ))

    # 불펜 피로도: 최근 2경기 실점(높을수록 불펜 피로 혹은 약세)
    if len(x) > 21:
        bullpen_gap = x[21] - x[20]  # 원정 최근RA2 - 홈 최근RA2: 양수 = 홈 투수진 우세
        if abs(bullpen_gap) >= 2.0:
            better = home if bullpen_gap > 0 else away
            s_ra = x[20] if bullpen_gap > 0 else x[21]
            candidates.append((
                abs(bullpen_gap) * 0.4,
                f"{better} 투수진 최근 2경기 실점 우세 ({s_ra:.1f}점)",
            ))

    # WHIP 우위 (낮을수록 좋음, KBO 리그 평균 약 1.40)
    if len(x) > 27:
        whip_gap = x[27] - x[26]  # 원정투수 WHIP - 홈투수 WHIP: 양수 = 홈투수 우세
        if abs(whip_gap) >= 0.15:
            better = home if whip_gap > 0 else away
            better_whip = x[26] if whip_gap > 0 else x[27]
            candidates.append((
                abs(whip_gap) * 2.5,
                f"{better} 선발 WHIP 우위 ({better_whip:.2f}) — 출루 억제력 우세",
            ))

    # K/9 우위 (높을수록 좋음, KBO 평균 약 7.0)
    if len(x) > 31:
        k9_gap = x[30] - x[31]  # 홈 K/9 - 원정 K/9
        if abs(k9_gap) >= 1.5:
            better = home if k9_gap > 0 else away
            better_k9 = x[30] if k9_gap > 0 else x[31]
            candidates.append((
                abs(k9_gap) * 0.5,
                f"{better} 선발 K/9 {better_k9:.1f} — 탈삼진 능력 우세",
            ))

    # 연속 원정 피로 (x34=원정팀 연속원정, x35=홈팀 연속홈)
    if len(x) > 34 and x[34] >= 4:
        candidates.append((
            x[34] * 0.5,
            f"{away} {int(x[34])}연속 원정 — 이동 피로 누적",
        ))
    if len(x) > 35 and x[35] >= 4:
        candidates.append((
            x[35] * 0.4,
            f"{home} {int(x[35])}연속 홈 — 홈 루틴 안정",
        ))

    # 백투백 (전날 경기 여부)
    if len(x) > 37:
        if x[37] and not x[36]:
            candidates.append((1.2, f"{away} 전날 경기 소화 (백투백) — {home} 대비 피로"))
        elif x[36] and not x[37]:
            candidates.append((1.2, f"{home} 전날 경기 소화 (백투백) — {away} 대비 피로"))

    # BB/9 우위 (낮을수록 좋음, KBO 평균 약 3.5)
    if len(x) > 33:
        bb9_gap = x[33] - x[32]  # 원정 BB/9 - 홈 BB/9: 양수 = 홈 유리
        if abs(bb9_gap) >= 1.0:
            better = home if bb9_gap > 0 else away
            better_bb9 = x[32] if bb9_gap > 0 else x[33]
            candidates.append((
                abs(bb9_gap) * 0.7,
                f"{better} 선발 BB/9 {better_bb9:.1f} — 제구력 우세",
            ))

    # 평균 이닝 소화 (높을수록 불펜 의존도 낮음, KBO 평균 약 5.5이닝)
    if len(x) > 29:
        ip_gap = x[28] - x[29]  # 홈 avg_ip - 원정 avg_ip
        if abs(ip_gap) >= 0.7:
            better = home if ip_gap > 0 else away
            better_ip = x[28] if ip_gap > 0 else x[29]
            candidates.append((
                abs(ip_gap) * 0.8,
                f"{better} 선발 평균 {better_ip:.1f}이닝 소화 — 불펜 부담 적음",
            ))

    # 위협 타자 수 (핸드니스 매치업 기반)
    if len(x) > 25:
        high_gap = x[24] - x[25]  # 홈 HIGH타자 수 - 원정 HIGH타자 수
        if abs(high_gap) >= 1.5:
            better = home if high_gap > 0 else away
            s_cnt = x[24] if high_gap > 0 else x[25]
            candidates.append((
                abs(high_gap) * 0.6,
                f"{better} 타선 핸드니스 매치업 우세 — 위협 타자 {int(s_cnt)}명",
            ))
        max_gap = x[22] - x[23]  # 홈 최고 타자 점수 - 원정 최고 타자 점수
        if abs(max_gap) >= 0.025:
            better = home if max_gap > 0 else away
            s_max = x[22] if max_gap > 0 else x[23]
            candidates.append((
                abs(max_gap) * 5,
                f"{better} 클린업 타자 투수 궁합 우세 (점수 {s_max:.3f})",
            ))

    candidates.sort(key=lambda c: c[0], reverse=True)
    reasons = [text for _, text in candidates[:3]]

    if not reasons:
        pick = home if p >= 0.5 else away
        reasons.append(f"뚜렷한 우위 요인 없음 - 홈 어드밴티지({home}) 반영, {pick} 근소 우세")
    return reasons


def _avg_ip(ip: float, gs: int) -> float:
    """등판당 평균 이닝 소화 (이닝 소화 능력 = 불펜 의존도 역수)."""
    if gs > 0:
        return _clamp(ip / gs, 2.0, 9.0)
    return 5.5  # KBO 선발 리그 평균


def _park_factor(stadium_name: str) -> float:
    # 1.00 = neutral, >1.00 hitter-friendly
    mapping = {
        "잠실": 0.93,
        "대구": 1.08,
        "문학": 1.03,
        "고척": 0.98,
        "광주": 1.01,
        "대전": 1.02,
        "사직": 1.00,
        "수원": 1.04,
        "창원": 0.99,
    }
    return mapping.get(stadium_name.strip(), 1.00)


def _fetch_weather_for_stadium(stadium_name: str) -> dict[str, float]:
    coords = {
        "잠실": (37.5121, 127.0719),
        "대구": (35.8419, 128.6811),
        "문학": (37.4369, 126.6933),
        "고척": (37.4982, 126.8671),
        "광주": (35.1681, 126.8891),
        "대전": (36.3172, 127.4292),
        "사직": (35.1940, 129.0615),
        "수원": (37.2998, 127.0098),
        "창원": (35.2226, 128.5828),
    }
    latlon = coords.get(stadium_name.strip())
    if not latlon:
        return {"temp_c": 18.0, "wind_kph": 5.0, "precip_mm": 0.0}
    lat, lon = latlon
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,wind_speed_10m,precipitation"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        c: dict[str, Any] = r.json().get("current", {})
        return {
            "temp_c": float(c.get("temperature_2m", 18.0)),
            "wind_kph": float(c.get("wind_speed_10m", 5.0)),
            "precip_mm": float(c.get("precipitation", 0.0)),
        }
    except Exception:
        return {"temp_c": 18.0, "wind_kph": 5.0, "precip_mm": 0.0}
