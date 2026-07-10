from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from .models import MatchupStat


@dataclass
class HitterSplit:
    vs_left_avg: float
    vs_right_avg: float
    hand: str  # L/R/U


@dataclass
class PitcherSplit:
    vs_left_avg_allowed: float
    vs_right_avg_allowed: float
    hand: str  # L/R/U


@dataclass
class PitcherSeasonStats:
    era: float
    whip: float
    ip: float
    gs: int
    k9: float = 7.0   # KBO 리그 평균
    bb9: float = 3.5  # KBO 리그 평균


@dataclass
class PitcherRecentForm:
    era: float
    n_starts: int


@dataclass
class PlayerRef:
    player_id: int
    team_code: str
    hand: str  # L/R/U


class KBORecordClient:
    VS_URL = "https://www.koreabaseball.com/Record/Etc/HitVsPit.aspx"
    SEARCH_URL = "https://www.koreabaseball.com/ws/Controls.asmx/GetSearchPlayer"
    HITTER_SITUATION_URL = "https://www.koreabaseball.com/Record/Player/HitterDetail/Situation.aspx"
    PITCHER_SITUATION_URL = "https://www.koreabaseball.com/Record/Player/PitcherDetail/Situation.aspx"
    PITCHER_BASIC_URL = "https://www.koreabaseball.com/Record/Player/PitcherDetail/Basic.aspx"
    # KBO 공식 사이트에 개별 경기 로그 페이지는 없음 — "경기별기록" 링크가 실제로는
    # 월별/구장별 등 스플릿 페이지(Game.aspx)로 연결됨. 월별 스플릿의 최근 달을
    # "최근 폼" 근사치로 사용한다. (구 GameList.aspx는 404 — 항상 기본값만 반환했음)
    PITCHER_MONTHLY_URL = "https://www.koreabaseball.com/Record/Player/PitcherDetail/Game.aspx"

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
            }
        )
        self._player_cache: dict[tuple[str, str], PlayerRef] = {}
        self._hitter_split_cache: dict[int, HitterSplit] = {}
        self._pitcher_split_cache: dict[int, PitcherSplit] = {}
        self._matchup_cache: dict[tuple[str, str, str, str], MatchupStat] = {}
        self._pitcher_season_cache: dict[int, PitcherSeasonStats] = {}
        self._pitcher_recent_cache: dict[int, PitcherRecentForm] = {}

    def _get_with_retry(
        self, url: str, params: dict, retries: int = 3, backoff: float = 0.6
    ) -> requests.Response:
        """일시적 네트워크/서버 오류 시 재시도 — 예전엔 1회 실패로 바로 리그 평균값에 영구히 묻혔음."""
        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(retries):
            try:
                res = self.session.get(url, params=params, timeout=self.timeout)
                res.raise_for_status()
                return res
            except Exception as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
        raise last_exc

    def _post_with_retry(
        self, url: str, data: dict, retries: int = 3, backoff: float = 0.6
    ) -> requests.Response:
        last_exc: Exception = RuntimeError("no attempt made")
        for attempt in range(retries):
            try:
                res = self.session.post(url, data=data, timeout=self.timeout)
                res.raise_for_status()
                return res
            except Exception as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
        raise last_exc

    def fetch_matchup(
        self, pitcher_name: str, batter_name: str, pitcher_team: str = "", batter_team: str = ""
    ) -> MatchupStat:
        cache_key = (pitcher_name.strip(), batter_name.strip(), pitcher_team.strip(), batter_team.strip())
        if cache_key in self._matchup_cache:
            return self._matchup_cache[cache_key]

        p_code = _team_name_to_code(pitcher_team)
        h_code = _team_name_to_code(batter_team)
        if not p_code or not h_code:
            return _empty(pitcher_name, batter_name)

        try:
            html = self._get_initial()
            html = self._postback_select_team(html, "pitcher", p_code)
            html = self._postback_select_player(html, "pitcher", pitcher_name)
            html = self._postback_select_team(html, "hitter", h_code)
            html = self._postback_select_player(html, "hitter", batter_name)
            html = self._submit_search(html)
            result = self._parse_matchup_stat(html, pitcher_name, batter_name)
            self._matchup_cache[cache_key] = result
            return result
        except Exception:
            return _empty(pitcher_name, batter_name)

    def fetch_hitter_split(self, hitter_name: str, hitter_team: str) -> HitterSplit:
        pref = self._resolve_player(hitter_name, hitter_team)
        if not pref:
            return HitterSplit(vs_left_avg=0.0, vs_right_avg=0.0, hand="U")
        if pref.player_id in self._hitter_split_cache:
            return self._hitter_split_cache[pref.player_id]

        try:
            res = self._get_with_retry(self.HITTER_SITUATION_URL, {"playerId": pref.player_id})
            soup = BeautifulSoup(res.text, "html.parser")
            tbl = _find_table_after_heading(soup, "투수유형별")
            data = _parse_split_avg_table(tbl)
            split = HitterSplit(
                vs_left_avg=data.get("좌투수", 0.0),
                vs_right_avg=data.get("우투수", 0.0),
                hand=pref.hand,
            )
            self._hitter_split_cache[pref.player_id] = split
            return split
        except Exception:
            return HitterSplit(vs_left_avg=0.0, vs_right_avg=0.0, hand=pref.hand)

    def fetch_pitcher_season_stats(self, pitcher_name: str, pitcher_team: str) -> PitcherSeasonStats:
        pref = self._resolve_player(pitcher_name, pitcher_team)
        if not pref:
            return PitcherSeasonStats(era=4.50, whip=1.40, ip=0.0, gs=0)
        if pref.player_id in self._pitcher_season_cache:
            return self._pitcher_season_cache[pref.player_id]
        try:
            res = self._get_with_retry(self.PITCHER_BASIC_URL, {"playerId": pref.player_id})
            soup = BeautifulSoup(res.text, "html.parser")
            stats = _parse_pitcher_season_table(soup)
            self._pitcher_season_cache[pref.player_id] = stats
            return stats
        except Exception:
            return PitcherSeasonStats(era=4.50, whip=1.40, ip=0.0, gs=0)

    def fetch_pitcher_recent_form(self, pitcher_name: str, pitcher_team: str, n: int = 3) -> PitcherRecentForm:
        pref = self._resolve_player(pitcher_name, pitcher_team)
        if not pref:
            return PitcherRecentForm(era=4.50, n_starts=0)
        if pref.player_id in self._pitcher_recent_cache:
            return self._pitcher_recent_cache[pref.player_id]
        try:
            res = self._get_with_retry(self.PITCHER_MONTHLY_URL, {"playerId": pref.player_id})
            soup = BeautifulSoup(res.text, "html.parser")
            form = _parse_pitcher_recent_form(soup, n=n)
            self._pitcher_recent_cache[pref.player_id] = form
            return form
        except Exception:
            return PitcherRecentForm(era=4.50, n_starts=0)

    def fetch_pitcher_split(self, pitcher_name: str, pitcher_team: str) -> PitcherSplit:
        pref = self._resolve_player(pitcher_name, pitcher_team)
        if not pref:
            return PitcherSplit(vs_left_avg_allowed=0.0, vs_right_avg_allowed=0.0, hand="U")
        if pref.player_id in self._pitcher_split_cache:
            return self._pitcher_split_cache[pref.player_id]

        try:
            res = self._get_with_retry(self.PITCHER_SITUATION_URL, {"playerId": pref.player_id})
            soup = BeautifulSoup(res.text, "html.parser")
            tbl = _find_table_after_heading(soup, "타자유형별")
            data = _parse_split_avg_table(tbl)
            split = PitcherSplit(
                vs_left_avg_allowed=data.get("좌타자", 0.0),
                vs_right_avg_allowed=data.get("우타자", 0.0),
                hand=pref.hand,
            )
            self._pitcher_split_cache[pref.player_id] = split
            return split
        except Exception:
            return PitcherSplit(vs_left_avg_allowed=0.0, vs_right_avg_allowed=0.0, hand=pref.hand)

    def _resolve_player(self, player_name: str, team_name: str) -> PlayerRef | None:
        key = (player_name.strip(), team_name.strip())
        if key in self._player_cache:
            return self._player_cache[key]

        team_code = _team_name_to_code(team_name)
        if not team_code:
            return None

        try:
            res = self._post_with_retry(self.SEARCH_URL, {"name": player_name})
            payload = res.json()
            candidates = payload.get("now", []) + payload.get("retire", [])
            for c in candidates:
                if _norm(c.get("P_NM", "")) != _norm(player_name):
                    continue
                if c.get("T_ID") != team_code:
                    continue
                pref = PlayerRef(
                    player_id=int(c["P_ID"]),
                    team_code=str(c.get("T_ID", "")),
                    hand=_hand_from_type(str(c.get("P_TYPE", ""))),
                )
                self._player_cache[key] = pref
                return pref
        except Exception:
            return None
        return None

    def _get_initial(self) -> str:
        res = self.session.get(self.VS_URL, timeout=self.timeout)
        res.raise_for_status()
        return res.text

    def _postback_select_team(self, html: str, side: str, team_code: str) -> str:
        target = f"{_side_prefix()}$ddl{'Pitcher' if side == 'pitcher' else 'Hitter'}Team"
        data = _build_form_data(html)
        data["__EVENTTARGET"] = target
        data["__EVENTARGUMENT"] = ""
        data[target] = team_code
        res = self.session.post(self.VS_URL, data=data, timeout=self.timeout)
        res.raise_for_status()
        return res.text

    def _postback_select_player(self, html: str, side: str, player_name: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        side_word = "Pitcher" if side == "pitcher" else "Hitter"
        player_key = f"{_side_prefix()}$ddl{side_word}Player"
        team_key = f"{_side_prefix()}$ddl{side_word}Team"
        player_sel_id = f"cphContents_cphContents_cphContents_ddl{side_word}Player"
        team_sel_id = f"cphContents_cphContents_cphContents_ddl{side_word}Team"

        sel = soup.find("select", id=player_sel_id)
        if not sel:
            return html
        player_value = ""
        target = _norm(player_name)
        for opt in sel.select("option"):
            if _norm(opt.get_text(strip=True)) == target:
                player_value = opt.get("value", "")
                break
        if not player_value or player_value == "0":
            return html

        data = _build_form_data(html)
        data["__EVENTTARGET"] = player_key
        data["__EVENTARGUMENT"] = ""
        data[player_key] = player_value
        current_team = _current_selected_value(soup, team_sel_id)
        if current_team:
            data[team_key] = current_team

        res = self.session.post(self.VS_URL, data=data, timeout=self.timeout)
        res.raise_for_status()
        return res.text

    def _submit_search(self, html: str) -> str:
        data = _build_form_data(html)
        data["__EVENTTARGET"] = ""
        data["__EVENTARGUMENT"] = ""
        data[f"{_side_prefix()}$btnSearch"] = "검색"
        res = self.session.post(self.VS_URL, data=data, timeout=self.timeout)
        res.raise_for_status()
        return res.text

    def _parse_matchup_stat(self, html: str, pitcher_name: str, batter_name: str) -> MatchupStat:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.tData.tt")
        if not table:
            return _empty(pitcher_name, batter_name)
        rows = table.select("tr")
        if len(rows) < 2:
            return _empty(pitcher_name, batter_name)
        cols = [c.get_text(strip=True) for c in rows[1].select("td")]
        if len(cols) < 11 or "기록이 없습니다" in "".join(cols):
            return _empty(pitcher_name, batter_name)
        return MatchupStat(
            pitcher=pitcher_name,
            batter=batter_name,
            pa=_to_int(cols[1]),
            ab=_to_int(cols[2]),
            hits=_to_int(cols[3]),
            hr=_to_int(cols[6]),
            bb=_to_int(cols[8]),
            so=_to_int(cols[10]),
        )


def _find_table_after_heading(soup: BeautifulSoup, heading_text: str):
    heading = soup.find(lambda t: t.name in ("h3", "h4", "h5", "strong") and heading_text in t.get_text())
    if not heading:
        return None
    return heading.find_next("table")


def _parse_split_avg_table(table) -> dict[str, float]:
    if not table:
        return {}
    headers = [x.get_text(strip=True) for x in table.select("tr th")]
    if not headers:
        return {}
    try:
        avg_idx = headers.index("AVG")
    except ValueError:
        return {}

    out: dict[str, float] = {}
    for tr in table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) <= avg_idx:
            continue
        kind = tds[0].get_text(strip=True)
        avg = _to_float(tds[avg_idx].get_text(strip=True))
        if kind:
            out[kind] = avg
    return out


def _build_form_data(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", id="mainForm")
    if not form:
        return {}
    data: dict[str, str] = {}
    for inp in form.select("input[name]"):
        itype = (inp.get("type") or "").lower()
        name = inp.get("name")
        if not name:
            continue
        if itype in ("checkbox", "radio") and not inp.has_attr("checked"):
            continue
        data[name] = inp.get("value", "")
    for sel in form.select("select[name]"):
        name = sel.get("name")
        if not name:
            continue
        selected = sel.find("option", selected=True) or sel.find("option")
        data[name] = selected.get("value", "") if selected else ""
    return data


def _current_selected_value(soup: BeautifulSoup, select_id: str) -> str:
    sel = soup.find("select", id=select_id)
    if not sel:
        return ""
    selected = sel.find("option", selected=True) or sel.find("option")
    return selected.get("value", "") if selected else ""


def _to_int(value: str) -> int:
    try:
        return int(value.replace(",", "").strip())
    except ValueError:
        return 0


def _to_float(value: str) -> float:
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return 0.0


def _empty(pitcher_name: str, batter_name: str) -> MatchupStat:
    return MatchupStat(pitcher=pitcher_name, batter=batter_name, pa=0, ab=0, hits=0, hr=0, bb=0, so=0)


def _side_prefix() -> str:
    return "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents"


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name or "").strip()


def _team_name_to_code(name: str) -> str:
    mapping = {
        "KT": "KT",
        "삼성": "SS",
        "LG": "LG",
        "SSG": "SK",
        "KIA": "HT",
        "한화": "HH",
        "두산": "OB",
        "NC": "NC",
        "롯데": "LT",
        "키움": "WO",
    }
    return mapping.get((name or "").strip(), "")


def _hand_from_type(p_type: str) -> str:
    t = (p_type or "").strip()
    if "좌" in t:
        return "L"
    if "우" in t:
        return "R"
    return "U"


def _parse_pitcher_season_table(soup: BeautifulSoup) -> PitcherSeasonStats:
    tables = soup.select("table")
    if not tables:
        return PitcherSeasonStats(era=4.50, whip=1.40, ip=0.0, gs=0)

    def _extract(table, col_name: str) -> str:
        ths = [th.get_text(strip=True) for th in table.select("tr th")]
        try:
            idx = ths.index(col_name)
        except ValueError:
            return ""
        rows = [tr for tr in table.select("tr") if tr.select("td")]
        if not rows:
            return ""
        tds = rows[0].select("td")
        return tds[idx].get_text(strip=True) if idx < len(tds) else ""

    # ERA, IP, GS(선발)은 table[0], BB/SO/WHIP은 table[1]
    era = _to_float(_extract(tables[0], "ERA")) or 4.50
    ip_raw = _extract(tables[0], "IP")
    ip = _parse_ip_season(ip_raw)
    gs_raw = _extract(tables[0], "G")   # 선발 횟수 컬럼명이 없으면 G 사용
    gs = _to_int(_extract(tables[0], "선발") or gs_raw or "0")

    whip = bb = so = 0.0
    if len(tables) > 1:
        whip = _to_float(_extract(tables[1], "WHIP")) or 1.40
        bb = _to_float(_extract(tables[1], "BB"))
        so = _to_float(_extract(tables[1], "SO"))

    k9 = round((so / ip) * 9, 2) if ip > 0 else 7.0
    bb9 = round((bb / ip) * 9, 2) if ip > 0 else 3.5

    return PitcherSeasonStats(
        era=era,
        whip=whip if whip > 0 else 1.40,
        ip=ip,
        gs=gs,
        k9=k9,
        bb9=bb9,
    )


def _parse_pitcher_recent_form(soup: BeautifulSoup, n: int = 3, min_ip: float = 8.0) -> PitcherRecentForm:
    """KBO 사이트에는 선발별 개별 경기 로그 페이지가 없어(경기별기록 링크도 실제로는
    스플릿 페이지로 연결됨), "월별" 스플릿 테이블을 최근 폼 근사치로 사용한다.
    가장 최근 달만 쓰면 등판 1~2회짜리 극단치(ERA 50+)가 나올 수 있어, 최소 min_ip
    이닝을 채울 때까지 최근 달부터 역순으로 누적한 뒤 ER*9/IP로 계산한다."""
    table = _find_table_after_heading(soup, "월별")
    if not table:
        return PitcherRecentForm(era=4.50, n_starts=0)
    headers = [th.get_text(strip=True) for th in table.select("tr th")]

    def _col(name: str) -> int:
        try:
            return headers.index(name)
        except ValueError:
            return -1

    g_idx = _col("G")
    ip_idx = _col("IP")
    er_idx = _col("ER")
    if ip_idx == -1 or er_idx == -1:
        return PitcherRecentForm(era=4.50, n_starts=0)

    rows = [tr for tr in table.select("tr") if tr.select("td")]
    rows = [tr for tr in rows if "기록이 없습니다" not in tr.get_text()]
    if not rows:
        return PitcherRecentForm(era=4.50, n_starts=0)

    total_ip = 0.0
    total_er = 0.0
    total_g = 0
    for tr in reversed(rows):
        tds = tr.select("td")
        if ip_idx >= len(tds) or er_idx >= len(tds):
            continue
        total_ip += _parse_ip_season(tds[ip_idx].get_text(strip=True))
        total_er += _to_float(tds[er_idx].get_text(strip=True))
        if 0 <= g_idx < len(tds):
            total_g += _to_int(tds[g_idx].get_text(strip=True))
        if total_ip >= min_ip:
            break

    if total_ip < 1.0:
        return PitcherRecentForm(era=4.50, n_starts=total_g)
    era = max(0.0, min((total_er / total_ip) * 9, 12.0))
    return PitcherRecentForm(era=round(era, 2), n_starts=total_g)


def _parse_ip_season(ip_str: str) -> float:
    """시즌 누적 IP 변환: '39 1/3' 또는 '39.1' 두 형식 모두 처리."""
    s = ip_str.replace(",", "").strip()
    if not s:
        return 0.0
    # "39 1/3" 분수 형식
    if "/" in s:
        parts = s.split()
        try:
            whole = float(parts[0]) if parts else 0.0
            num, den = parts[1].split("/") if len(parts) > 1 else ("0", "1")
            return whole + int(num) / int(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    # "39.1" KBO 아웃 표기 형식
    try:
        val = float(s)
        whole = int(val)
        outs = round((val - whole) * 10)
        return whole + outs / 3
    except (ValueError, TypeError):
        return 0.0
