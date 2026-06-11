from __future__ import annotations

from .kbo_client import KBORecordClient
from .models import GameAnalysis, GameLineup


class LineupAnalyzer:
    def __init__(self, kbo_client: KBORecordClient) -> None:
        self.kbo = kbo_client

    def analyze_game(self, lineup: GameLineup) -> GameAnalysis:
        notes: list[str] = []

        notes.append(
            f"{lineup.away.team} starter {lineup.away.starter_pitcher} vs "
            f"{lineup.home.team} lineup"
        )
        notes.extend(
            self._analyze_pitcher_vs_lineup(
                pitcher=lineup.away.starter_pitcher,
                pitcher_team=lineup.away.team,
                opponent_team=lineup.home.team,
                batters=lineup.home.batters,
            )
        )

        notes.append(
            f"{lineup.home.team} starter {lineup.home.starter_pitcher} vs "
            f"{lineup.away.team} lineup"
        )
        notes.extend(
            self._analyze_pitcher_vs_lineup(
                pitcher=lineup.home.starter_pitcher,
                pitcher_team=lineup.home.team,
                opponent_team=lineup.away.team,
                batters=lineup.away.batters,
            )
        )

        title = f"{lineup.away.team} @ {lineup.home.team} ({lineup.game_time})"
        return GameAnalysis(game_id=lineup.game_id, title=title, notes=notes)

    def _analyze_pitcher_vs_lineup(
        self, pitcher: str, pitcher_team: str, opponent_team: str, batters: list
    ) -> list[str]:
        lines: list[str] = []
        high_risk = 0

        p_split = self.kbo.fetch_pitcher_split(pitcher, pitcher_team)
        lines.append(
            f"- pitcher split allowed AVG vs L/R: "
            f"{p_split.vs_left_avg_allowed:.3f}/{p_split.vs_right_avg_allowed:.3f}"
        )

        for batter in batters[:9]:
            h_split = self.kbo.fetch_hitter_split(batter.name, opponent_team)
            stat = self.kbo.fetch_matchup(
                pitcher_name=pitcher,
                batter_name=batter.name,
                pitcher_team=pitcher_team,
                batter_team=opponent_team,
            )

            batter_vs_pitcher_hand = (
                h_split.vs_left_avg if p_split.hand == "L" else h_split.vs_right_avg
            )
            pitcher_vs_batter_hand = (
                p_split.vs_left_avg_allowed if h_split.hand == "L" else p_split.vs_right_avg_allowed
            )
            h2h_weight = min(stat.ab / 10.0, 1.0) * 0.2
            split_weight = 1.0 - h2h_weight
            score = split_weight * (
                0.55 * batter_vs_pitcher_hand + 0.45 * pitcher_vs_batter_hand
            ) + h2h_weight * stat.avg
            band = _score_band(score)

            lines.append(
                f"- {batter.order} {batter.name}: "
                f"H2H {stat.ab}AB-{stat.hits}H AVG {stat.avg:.3f}, "
                f"BAT split(vs {p_split.hand}) {batter_vs_pitcher_hand:.3f}, "
                f"PIT split(vs {h_split.hand}) {pitcher_vs_batter_hand:.3f}, "
                f"pred {score:.3f} [{band}]"
            )
            if score >= 0.300:
                high_risk += 1

        lines.append(
            f"=> summary: high-risk hitters (pred >= .300) = {high_risk}"
        )
        return lines


def _score_band(score: float) -> str:
    if score >= 0.300:
        return "HIGH"
    if score >= 0.260:
        return "MID"
    return "LOW"
