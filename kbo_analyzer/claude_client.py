from __future__ import annotations

import anthropic


class ClaudeAnalyst:
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def analyze_predictions(
        self,
        preds: list[dict],
        matchup_data: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> str:
        prompt = _build_analysis_prompt(preds, matchup_data or [])
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


def _build_analysis_prompt(preds: list[dict], matchup_data: list[dict]) -> str:
    matchup_lookup: dict[tuple[str, str], dict] = {
        (gm["away_team"], gm["home_team"]): gm for gm in matchup_data
    }

    lines = [
        "당신은 KBO(한국 프로야구) 전문 분석가입니다.",
        "아래는 오늘 경기에 대한 머신러닝 예측 데이터입니다. 각 경기를 분석하여 한국어로 작성해주세요.",
        "",
        "## 분석 요청사항",
        "각 경기별로 다음 형식으로 작성하세요:",
        "- **핵심 근거**: 가장 결정적인 우세 요인 1~2가지",
        "- **반대 신호**: 불리하거나 불확실한 지표",
        "- **리스크**: 경기 결과를 뒤집을 수 있는 변수",
        "- **결론**: 추천팀과 신뢰도를 한 줄로",
        "",
        "숫자만 나열하지 말고, 야구 맥락에서 왜 이 수치가 의미 있는지 설명해주세요.",
        "ERA가 낮을수록 투수가 좋고, 타선점수는 리그평균이 0.260입니다.",
        "",
        "---",
        "",
        "## 오늘의 경기 데이터",
        "",
    ]

    for pred in preds:
        s = pred.get("signals", {})
        away = pred["away_team"]
        home = pred["home_team"]
        home_prob = float(pred.get("home_win_prob", 0.5))
        away_prob = float(pred.get("away_win_prob", 0.5))
        pick = home if home_prob >= away_prob else away
        conf = max(home_prob, away_prob)
        confirmed = pred.get("lineup_confirmed", False)

        lines.append(f"### {away} @ {home}")
        lines.append(
            f"- 예측: **{pick}** 승리 ({conf*100:.1f}%)"
            f" | 예상 스코어: {away} {pred.get('away_score_pred','?')} - {home} {pred.get('home_score_pred','?')}"
        )
        lines.append(f"- 라인업: {'✅ 확정' if confirmed else '⚠️ 미확정'}")
        lines.append("")

        matchup = matchup_lookup.get((away, home))
        away_p = matchup["away_pitcher"] if matchup else "미상"
        home_p = matchup["home_pitcher"] if matchup else "미상"

        h_era = float(s.get("home_starter_era", 4.5))
        a_era = float(s.get("away_starter_era", 4.5))
        h_rec_era = float(s.get("home_starter_recent_era", 4.5))
        a_rec_era = float(s.get("away_starter_recent_era", 4.5))
        lines.append("**[투수]**")
        lines.append(f"- 홈 선발 {home_p}: 시즌ERA {h_era:.2f} / 최근3선발ERA {h_rec_era:.2f}")
        lines.append(f"- 원정 선발 {away_p}: 시즌ERA {a_era:.2f} / 최근3선발ERA {a_rec_era:.2f}")
        lines.append("")

        h_atk = float(s.get("home_attack", 0.26))
        a_atk = float(s.get("away_attack", 0.26))
        h_wr = float(s.get("home_recent_winrate", 0.5))
        a_wr = float(s.get("away_recent_winrate", 0.5))
        h_home_wr = float(s.get("home_team_home_wr", 0.5))
        a_away_wr = float(s.get("away_team_away_wr", 0.5))
        h2h = float(s.get("h2h_home_winrate", 0.5))
        h_runs = float(s.get("home_recent_runs", 4.5))
        a_runs = float(s.get("away_recent_runs", 4.5))
        h_ra = float(s.get("home_recent_ra", 4.5))
        a_ra = float(s.get("away_recent_ra", 4.5))
        h_ra2 = float(s.get("home_bullpen_ra2", 4.5))
        a_ra2 = float(s.get("away_bullpen_ra2", 4.5))

        lines.append("**[타선 / 팀 지표]**")
        lines.append(f"- 타선점수(홈/원정): {h_atk:.3f} / {a_atk:.3f}  (리그평균 0.260)")
        lines.append(f"- 최근5경기 승률(홈/원정): {h_wr*100:.0f}% / {a_wr*100:.0f}%")
        lines.append(f"- 홈팀 홈전용 승률: {h_home_wr*100:.0f}% / 원정팀 원정전용 승률: {a_away_wr*100:.0f}%")
        lines.append(f"- 올시즌 상대전적(홈팀 기준): {h2h*100:.0f}%")
        lines.append(f"- 최근5경기 평균득점(홈/원정): {h_runs:.1f} / {a_runs:.1f}")
        lines.append(f"- 최근5경기 평균실점(홈투수진/원정투수진): {h_ra:.1f} / {a_ra:.1f}")
        lines.append(f"- 최근2경기 실점(홈/원정): {h_ra2:.1f} / {a_ra2:.1f}  (불펜 피로 지표)")
        lines.append("")

        if matchup:
            lines.append("**[주요 타자 매치업 — 위협 타자 TOP3]**")
            for pitcher_label, pitcher_name, batter_rows in [
                (f"원정선발 {away_p} vs {home} 타선", away_p, matchup.get("home_batters_vs_away_pitcher", [])),
                (f"홈선발 {home_p} vs {away} 타선", home_p, matchup.get("away_batters_vs_home_pitcher", [])),
            ]:
                top = sorted(batter_rows, key=lambda r: r.get("score", 0), reverse=True)[:3]
                if top:
                    lines.append(f"▶ {pitcher_label}")
                    for r in top:
                        if r.get("h2h_ab", 0) > 0:
                            h2h_str = f"H2H {r['h2h_ab']}타수 {r['h2h_hits']}안타({r.get('h2h_avg', 0):.3f})"
                        else:
                            h2h_str = "H2H 기록없음"
                        lines.append(
                            f"  {r['order']}번 {r['name']}({r['batter_hand']}) "
                            f"점수{r['score']:.3f}[{r['band']}] {h2h_str}"
                        )
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
