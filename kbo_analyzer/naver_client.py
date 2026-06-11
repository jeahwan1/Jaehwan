from __future__ import annotations

import datetime as dt
import json
import time

import requests

from .models import GameLineup, LineupPlayer, TeamLineup


class NaverSportsClient:
    """
    기존 코드 호환을 위해 클래스명은 유지.
    실제 데이터 소스는 KBO 공식 엔드포인트를 사용한다.
    """

    GAME_LIST_URL = "https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList"
    LINEUP_URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetLineUpAnalysis"

    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "Referer": "https://www.koreabaseball.com/",
                "Origin": "https://www.koreabaseball.com",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            }
        )
        self._games_by_id: dict[str, dict] = {}

    def fetch_today_game_urls(self, date: dt.date) -> list[str]:
        payload = {"leId": "1", "srId": "0,1,3,4,5,7,9", "date": date.strftime("%Y%m%d")}
        games: list[dict] = []
        last_err: Exception | None = None
        for i in range(3):
            try:
                res = self.session.post(self.GAME_LIST_URL, data=payload, timeout=self.timeout)
                res.raise_for_status()
                games = res.json().get("game", [])
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (i + 1))
        if last_err and not games:
            raise last_err

        self._games_by_id = {str(g["G_ID"]): g for g in games}
        # 기존 인터페이스를 맞추기 위해 가상 URL 포맷 사용
        return [f"kbo://game/{gid}" for gid in self._games_by_id]

    def get_cached_games(self) -> list[dict]:
        return list(self._games_by_id.values())

    def fetch_game_lineup(self, game_url: str) -> GameLineup:
        game_id = game_url.split("/")[-1] if game_url.startswith("kbo://game/") else game_url
        game = self._games_by_id.get(game_id)
        if not game:
            raise ValueError(f"Unknown game id: {game_id}")

        lineup_payload = {
            "leId": str(game["LE_ID"]),
            "srId": str(game["SR_ID"]),
            "seasonId": str(game["SEASON_ID"]),
            "gameId": str(game["G_ID"]),
        }
        lineup_raw = self.session.post(self.LINEUP_URL, data=lineup_payload, timeout=self.timeout)
        lineup_raw.raise_for_status()
        lineup = lineup_raw.json()

        away = self._build_team_lineup(
            team_name=str(game["AWAY_NM"]).strip(),
            starter_pitcher=str(game.get("T_PIT_P_NM", "")).strip(),
            rows_json=self._safe_json_table(lineup, 4),
        )
        home = self._build_team_lineup(
            team_name=str(game["HOME_NM"]).strip(),
            starter_pitcher=str(game.get("B_PIT_P_NM", "")).strip(),
            rows_json=self._safe_json_table(lineup, 3),
        )

        return GameLineup(
            game_id=str(game["G_ID"]),
            game_time=str(game.get("G_TM", "")),
            away=away,
            home=home,
        )

    def _build_team_lineup(self, team_name: str, starter_pitcher: str, rows_json: dict) -> TeamLineup:
        batters: list[LineupPlayer] = []
        rows = rows_json.get("rows", [])
        for row_obj in rows:
            cells = row_obj.get("row", [])
            # [타순, 포지션, 선수명, WAR]
            if len(cells) < 3:
                continue
            order_text = str(cells[0].get("Text", "")).strip()
            name = str(cells[2].get("Text", "")).strip()
            if not order_text.isdigit() or not name:
                continue
            batters.append(
                LineupPlayer(order=int(order_text), name=name, team=team_name)
            )

        batters.sort(key=lambda x: x.order)
        return TeamLineup(team=team_name, starter_pitcher=starter_pitcher, batters=batters[:9])

    @staticmethod
    def _safe_json_table(raw: list, index: int) -> dict:
        if len(raw) <= index or not raw[index]:
            return {}
        text = raw[index][0]
        if not isinstance(text, str):
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
