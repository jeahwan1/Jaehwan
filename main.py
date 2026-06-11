from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from kbo_analyzer.analyzer import LineupAnalyzer
from kbo_analyzer.claude_client import ClaudeAnalyst
from kbo_analyzer.kbo_client import KBORecordClient
from kbo_analyzer.naver_client import NaverSportsClient
from kbo_analyzer.ollama_client import OllamaClient
from kbo_analyzer.predictor import GamePredictor


def run_once() -> None:
    output_dir = Path(os.getenv("KBO_ANALYZER_OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    naver = NaverSportsClient()
    kbo = KBORecordClient()
    analyzer = LineupAnalyzer(kbo)

    date = dt.datetime.now().date()
    fixed_urls = [u.strip() for u in os.getenv("NAVER_GAME_URLS", "").split(",") if u.strip()]
    game_urls = fixed_urls or naver.fetch_today_game_urls(date)

    if not game_urls:
        print("[INFO] No KBO game URLs found for today.")
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"kbo_analysis_{timestamp}.md"

    lines = [f"# KBO Daily Analysis Report ({date.isoformat()})", ""]
    for game_url in game_urls:
        try:
            lineup = naver.fetch_game_lineup(game_url)
            result = analyzer.analyze_game(lineup)
            lines.append(f"## {result.title}")
            lines.extend(result.notes)
            lines.append("")
        except Exception as e:
            lines.append(f"## Failed: {game_url}")
            lines.append(f"- Error: {e}")
            lines.append("")

    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Report saved: {report_file}")


def build_ollama_prompt(preds: list[dict]) -> str:
    slim = [
        {
            "match": f"{r.get('away_team')} @ {r.get('home_team')}",
            "pick": r.get("home_team") if float(r.get("home_win_prob", 0)) >= 0.5 else r.get("away_team"),
            "home_win_prob": r.get("home_win_prob"),
            "away_win_prob": r.get("away_win_prob"),
            "home_score_pred": r.get("home_score_pred"),
            "away_score_pred": r.get("away_score_pred"),
            "reasons": r.get("reasons", []),
        }
        for r in preds
    ]
    data = json.dumps(slim, ensure_ascii=False, indent=2)
    return (
        "KBO 경기 예측 데이터를 보고 한국어로 간결하게 분석해라.\n"
        "각 경기마다: 추천팀·승률·예상스코어·핵심근거 2개·리스크 1개.\n"
        "불확실성을 명시하고 200자 이내로 작성.\n\n"
        f"데이터:\n{data}\n"
    )


def _confidence_tier(conf: float) -> str:
    if conf >= 0.65:
        return "강함"
    if conf >= 0.58:
        return "보통"
    return "박빙"


def _factor_comment(atk_gap: float, run_gap: float) -> str:
    atk_txt = "홈 타선 우세" if atk_gap > 0.01 else ("원정 타선 우세" if atk_gap < -0.01 else "타선 비슷")
    run_txt = (
        "홈 최근 득점 흐름 우세"
        if run_gap > 0.3
        else ("원정 최근 득점 흐름 우세" if run_gap < -0.3 else "최근 득점 흐름 비슷")
    )
    return f"{atk_txt}, {run_txt}"


def _risk_comment(temp: float, wind: float, conf: float) -> str:
    weather_risk = []
    if temp >= 27:
        weather_risk.append("고온 변수")
    if wind >= 11:
        weather_risk.append("강풍 변수")
    if not weather_risk:
        weather_risk.append("날씨 변수 보통")
    conf_risk = "확률 박빙 구간" if conf < 0.58 else "확률 우세 구간"
    return f"{', '.join(weather_risk)}, {conf_risk}"


def _render_matchup_section(matchup_data: list[dict]) -> list[str]:
    if not matchup_data:
        return []
    lines: list[str] = ["## 선발투수 vs 타자 매치업", ""]
    for gm in matchup_data:
        away, home = gm["away_team"], gm["home_team"]
        away_p, home_p = gm["away_pitcher"], gm["home_pitcher"]
        lines.append(f"### {away} @ {home}")
        lines.append("")
        for pitcher_label, pitcher_name, batter_team, rows in [
            (f"{away} 선발 **{away_p}** vs {home} 타선", away_p, home, gm["home_batters_vs_away_pitcher"]),
            (f"{home} 선발 **{home_p}** vs {away} 타선", home_p, away, gm["away_batters_vs_home_pitcher"]),
        ]:
            lines.append(f"#### {pitcher_label}")
            lines.append("")
            lines.append("| # | 타자 | 손 | vs투수타입 | 투수허용 | 종합점수 | 상대전적 | H2H타율 |")
            lines.append("|--:|:--|:--:|--:|--:|--:|:--|--:|")
            for r in rows:
                band_icon = "🔴" if r["band"] == "HIGH" else ("🟡" if r["band"] == "MID" else "⚪")
                bvp = f".{int(r['batter_vs_pitcher_hand'] * 1000):03d}"
                pvb = f".{int(r['pitcher_vs_batter_hand'] * 1000):03d}"
                score_str = f"{band_icon} {r['score']:.3f}"
                if r.get("h2h_ab", 0) > 0:
                    h2h_str = f"{r['h2h_ab']}-{r['h2h_hits']}-{r['h2h_hr']}"
                    avg_str = f".{int(r['h2h_avg'] * 1000):03d}" if r["h2h_avg"] is not None else "-"
                else:
                    h2h_str, avg_str = "-", "-"
                lines.append(
                    f"| {r['order']} | {r['name']} | {r['batter_hand']} | "
                    f"{bvp} | {pvb} | {score_str} | {h2h_str} | {avg_str} |"
                )
            lines.append("")
    return lines


def build_korean_prediction_report(
    preds: list[dict], source_json_name: str, matchup_data: list[dict] | None = None
) -> str:
    lines: list[str] = []
    lines.append("# KBO 승패 예측 요약(한글)")
    lines.append("")
    lines.append(f"- 원본 파일: {source_json_name}")
    lines.append(f"- 경기 수: {len(preds)}")
    unconfirmed = [r for r in preds if not r.get("lineup_confirmed")]
    if unconfirmed:
        teams = ", ".join(
            f"{r.get('away_team')} @ {r.get('home_team')}" for r in unconfirmed
        )
        lines.append(f"- ⚠️ 라인업 미확정 경기 ({len(unconfirmed)}건): {teams}")
        lines.append("  → 실제 출전 선수가 다를 수 있어 타선 점수가 보정됐습니다. 신뢰도를 낮게 보세요.")
    lines.append("")
    lines.append("## 오늘의 추천 순위")
    lines.append("")
    ranked = sorted(
        preds,
        key=lambda r: max(float(r.get("home_win_prob", 0.0)), float(r.get("away_win_prob", 0.0))),
        reverse=True,
    )
    for i, r in enumerate(ranked, start=1):
        home_prob = float(r.get("home_win_prob", 0.0))
        away_prob = float(r.get("away_win_prob", 0.0))
        pick = r.get("home_team") if home_prob >= away_prob else r.get("away_team")
        conf = max(home_prob, away_prob)
        tier = _confidence_tier(conf)
        away_score = float(r.get("away_score_pred", 0.0))
        home_score = float(r.get("home_score_pred", 0.0))
        reasons = r.get("reasons", [])
        confirmed = r.get("lineup_confirmed", False)
        unconfirmed_tag = " ⚠️미확정" if not confirmed else ""
        lines.append(
            f"{i}. **{r.get('away_team')} @ {r.get('home_team')} → {pick}**"
            f"  ({conf*100:.1f}% · {tier} 우세{unconfirmed_tag})"
        )
        for reason in reasons[:3]:
            lines.append(f"   - {reason}")
        lines.append(
            f"   - 예상 스코어: {r.get('away_team')} {away_score:.1f} : {r.get('home_team')} {home_score:.1f}"
        )
        lines.append("")
    lines.append("## 경기별 예측")
    lines.append("")
    lines.append("| 경기 | 추천 | 홈 승률 | 원정 승률 | 신뢰도 | 예상 스코어(원정-홈) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in preds:
        home = float(r.get("home_win_prob", 0.0))
        away = float(r.get("away_win_prob", 0.0))
        pick = r.get("home_team") if home >= away else r.get("away_team")
        conf = max(home, away)
        pred_score = f"{float(r.get('away_score_pred', 0.0)):.1f}-{float(r.get('home_score_pred', 0.0)):.1f}"
        lines.append(
            f"| {r.get('away_team')} @ {r.get('home_team')} | {pick} | "
            f"{home*100:.1f}% | {away*100:.1f}% | {conf*100:.1f}% | {pred_score} |"
        )
    lines.append("")
    lines.append("## 해석 기준")
    lines.append("")
    lines.append("- 신뢰도 60% 이상: 우세")
    lines.append("- 신뢰도 55~59.9%: 근소 우세")
    lines.append("- 신뢰도 54.9% 이하: 박빙")
    lines.append("")
    lines.append("## 경기별 상세 코멘트")
    lines.append("")
    for r in preds:
        s = r.get("signals", {})
        home = float(r.get("home_win_prob", 0.0))
        away = float(r.get("away_win_prob", 0.0))
        pick = r.get("home_team") if home >= away else r.get("away_team")
        conf = max(home, away)
        tier = _confidence_tier(conf)
        ha = float(s.get("home_attack", 0))
        aa = float(s.get("away_attack", 0))
        hr = float(s.get("home_recent_runs", 0))
        ar = float(s.get("away_recent_runs", 0))
        temp = float(s.get("temp_c", 0))
        wind = float(s.get("wind_kph", 0))
        atk_gap = ha - aa
        run_gap = hr - ar
        lines.append(f"### {r.get('away_team')} @ {r.get('home_team')}")
        lines.append(f"- 추천: **{pick}**")
        lines.append(f"- 승률: 홈 {home*100:.1f}% / 원정 {away*100:.1f}%")
        lines.append(f"- 신뢰도: {conf*100:.1f}% ({tier})")
        lines.append(
            f"- 라인업 상태: {'확정' if bool(r.get('lineup_confirmed', False)) else '미확정(최근 라인업)'}"
        )
        lines.append(f"- 라인업 수집 시각: {r.get('lineup_collected_at', '-')}")
        lines.append(f"- 라인업 소스 메모: {r.get('lineup_source_note', '-')}")
        lines.append(
            f"- 예상 스코어: 원정 {float(r.get('away_score_pred', 0.0)):.1f} : 홈 {float(r.get('home_score_pred', 0.0)):.1f}"
        )
        lines.append(f"- 예상 총 득점: {float(r.get('total_score_pred', 0.0)):.1f}")
        h_era = float(s.get("home_starter_era", 4.5))
        a_era = float(s.get("away_starter_era", 4.5))
        h_recent_era = float(s.get("home_starter_recent_era", 4.5))
        a_recent_era = float(s.get("away_starter_recent_era", 4.5))
        h_home_wr = float(s.get("home_team_home_wr", 0.5))
        a_away_wr = float(s.get("away_team_away_wr", 0.5))
        h2h = float(s.get("h2h_home_winrate", 0.5))
        h_ra2 = float(s.get("home_bullpen_ra2", 4.5))
        a_ra2 = float(s.get("away_bullpen_ra2", 4.5))
        lines.append(f"- 타선 점수(홈/원정): {ha:.3f} / {aa:.3f}")
        lines.append(f"- 선발 ERA(홈/원정): {h_era:.2f} / {a_era:.2f}")
        lines.append(f"- 선발 최근3선발 ERA(홈/원정): {h_recent_era:.2f} / {a_recent_era:.2f}")
        lines.append(f"- 홈팀 홈승률 / 원정팀 원정승률: {h_home_wr*100:.1f}% / {a_away_wr*100:.1f}%")
        lines.append(f"- 올시즌 상대전적(홈팀 기준): {h2h*100:.1f}%")
        lines.append(f"- 최근2경기 실점(홈/원정): {h_ra2:.1f} / {a_ra2:.1f}")
        lines.append(f"- 최근 5경기 득점(홈/원정): {hr:.2f} / {ar:.2f}")
        lines.append(f"- 날씨(기온/풍속): {temp:.1f}C / {wind:.1f}kph")
        lines.append(f"- 우세 근거: {_factor_comment(atk_gap, run_gap)}")
        lines.append(f"- 주의 포인트: {_risk_comment(temp, wind, conf)}")
        reasons = r.get("reasons", [])
        if reasons:
            lines.append("- 핵심 근거:")
            for reason in reasons:
                lines.append(f"  - {reason}")
        lines.append("")
    lines.extend(_render_matchup_section(matchup_data or []))
    return "\n".join(lines)


def run_predict_mode(output_dir: Path) -> None:
    p = GamePredictor(db_path=os.getenv("KBO_DB_PATH", "kbo_predictor.db"))
    days = int(os.getenv("KBO_BACKFILL_DAYS", "30"))
    p.backfill(days=days)
    p.ingest_today_for_predict()
    preds = p.train_and_predict_today()

    if not preds:
        print("[PREDICT] 확정 라인업이 없습니다. KBO 라인업 공개 후(보통 오후 2~3시) 재실행하세요.")
        return

    print("[MATCHUP] 타자 상대전적 조회 중... (확정 경기당 약 1~2분 소요)")
    matchup_data = p.build_today_matchup_report()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_json_file = output_dir / f"kbo_predictions_{ts}.json"
    pred_json_file.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(),
                "modeling_mode": "logistic_mvp",
                "predictions": preds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PREDICT] JSON saved: {pred_json_file}")

    pred_report_file = output_dir / f"kbo_predictions_{ts}.md"
    pred_report_file.write_text(
        build_korean_prediction_report(preds, pred_json_file.name, matchup_data=matchup_data), encoding="utf-8"
    )
    print(f"[PREDICT] 리포트 저장: {pred_report_file}")

    use_claude = os.getenv("KBO_USE_CLAUDE", "0").strip().lower() in ("1", "true", "yes", "y")
    if use_claude:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            print("[CLAUDE] ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        else:
            try:
                model = os.getenv("CLAUDE_MODEL", "").strip() or None
                analyst = ClaudeAnalyst(api_key=api_key, model=model)
                print(f"[CLAUDE] {analyst.model} 분석 중...")
                claude_text = analyst.analyze_predictions(preds, matchup_data)
                claude_file = output_dir / f"kbo_claude_{ts}.md"
                claude_file.write_text(claude_text, encoding="utf-8")
                print(f"[CLAUDE] 분석 저장: {claude_file}")
            except Exception as e:
                print(f"[CLAUDE] 분석 실패: {e}")

    db_path = os.getenv("KBO_DB_PATH", "kbo_predictor.db")
    print_daily_brief(preds, output_dir, db_path)

    use_ollama = os.getenv("KBO_USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes", "y")
    if not use_ollama:
        return

    prompt = build_ollama_prompt(preds)
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama = OllamaClient(model=ollama_model, host=ollama_host)
    try:
        health = ollama.health()
        if not health["ok"]:
            raise RuntimeError(f"Ollama health check failed ({ollama_host}): {health['error']}")
        models = health.get("models", [])
        auto_model = os.getenv("KBO_OLLAMA_AUTO_MODEL", "1").strip().lower() in ("1", "true", "yes", "y")
        if models and ollama_model not in models and auto_model:
            print(f"[AI] Requested model '{ollama_model}' not found. Using '{models[0]}' instead.")
            ollama.model = models[0]

        ai_text = ollama.generate(prompt, timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")))
        ai_file = output_dir / f"kbo_ai_report_{ts}.md"
        ai_file.write_text(ai_text, encoding="utf-8")
        print(f"[AI] Report saved: {ai_file}")
    except Exception as e:
        print(f"[AI] Ollama generation failed: {e}")
        debug_prompt_file = output_dir / f"kbo_ai_prompt_{ts}.txt"
        debug_prompt_file.write_text(prompt, encoding="utf-8")
        print(f"[AI] Prompt saved for retry: {debug_prompt_file}")
        debug_info_file = output_dir / f"kbo_ai_debug_{ts}.txt"
        debug_info_file.write_text(
            "\n".join(
                [
                    f"OLLAMA_HOST={ollama_host}",
                    f"OLLAMA_MODEL={ollama_model}",
                    "Troubleshooting:",
                    "1) run: ollama serve",
                    "2) run: ollama pull <model>",
                    "3) check: curl http://localhost:11434/api/tags",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[AI] Debug info saved: {debug_info_file}")


def print_daily_brief(preds: list[dict], output_dir: Path, db_path: str) -> None:
    """오늘 예측 요약 + 어제 예측 vs 실제 결과를 콘솔에 출력."""
    import sqlite3

    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    SEP = "─" * 54

    # ── 오늘 예측 ───────────────────────────────────────────
    print(f"\n{'═'*54}")
    print(f"  KBO 예측 브리핑  {today.strftime('%Y-%m-%d')}")
    print(f"{'═'*54}")
    print(f"\n  [ 오늘 경기 예측 ]")
    print(f"  {SEP}")

    sorted_preds = sorted(preds, key=lambda r: max(float(r.get("home_win_prob", 0.5)), 1 - float(r.get("home_win_prob", 0.5))), reverse=True)
    for r in sorted_preds:
        home = r["home_team"]
        away = r["away_team"]
        hp = float(r["home_win_prob"])
        pick = home if hp >= 0.5 else away
        conf = max(hp, 1 - hp)
        phs = float(r["home_score_pred"])
        pas = float(r["away_score_pred"])
        total = phs + pas
        tags = []
        if conf >= 0.65:
            tags.append("강추")
        if total >= 14:
            tags.append(f"고득점({total:.1f}점)")
        elif total <= 8:
            tags.append(f"저득점({total:.1f}점)")
        confirmed = "✓" if r.get("lineup_confirmed") else "⚠미확정"
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {confirmed}  {away} @ {home}")
        print(f"        → {pick} 승  ({conf*100:.0f}%)  예상: {away} {pas:.1f} : {phs:.1f} {home}{tag_str}")

    print(f"  {SEP}\n")

    # ── 어제 결과 비교 ──────────────────────────────────────
    print(f"  [ 어제 경기 결과  {yesterday.strftime('%Y-%m-%d')} ]")
    print(f"  {SEP}")

    with sqlite3.connect(db_path) as con:
        actual_rows = con.execute(
            """
            SELECT game_id, home_team, away_team, y_home_win, home_score, away_score
            FROM features
            WHERE game_date = ? AND y_home_win IS NOT NULL
            ORDER BY game_id
            """,
            (yesterday.isoformat(),),
        ).fetchall()

    if not actual_rows:
        print("  어제 경기 결과가 아직 DB에 없습니다.")
        print(f"  {SEP}\n")
        return

    # 어제 날짜의 예측 JSON 파일 로드
    yesterday_str = yesterday.strftime("%Y%m%d")
    pred_by_game: dict[str, dict] = {}
    for json_file in sorted(output_dir.glob(f"kbo_predictions_{yesterday_str}_*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for pred in data.get("predictions", []):
                gid = pred.get("game_id")
                if gid:
                    pred_by_game[gid] = pred
        except Exception:
            pass

    rows_out: list[tuple[str, str, str, str, str]] = []
    total_y = correct_y = 0

    for game_id, home, away, y_home_win, home_score, away_score in actual_rows:
        winner = home if y_home_win == 1 else away
        actual_score = f"{away_score}:{home_score}" if home_score is not None else "?"
        actual_str = f"{winner} ({actual_score})"

        pred = pred_by_game.get(game_id)
        if pred:
            hp = float(pred.get("home_win_prob", 0.5))
            pick = home if hp >= 0.5 else away
            conf = max(hp, 1 - hp)
            phs = float(pred.get("home_score_pred", 0))
            pas = float(pred.get("away_score_pred", 0))
            hit = pick == winner
            total_y += 1
            correct_y += int(hit)
            flag = "✅" if hit else "❌"
            pred_str = f"{pick} ({conf*100:.0f}%)"
            pred_score = f"{pas:.1f}:{phs:.1f}"
        else:
            pred_str = pred_score = "-"
            flag = "-"

        rows_out.append((f"{away} @ {home}", pred_str, pred_score, actual_str, flag))

    # 컬럼 너비 맞추기
    headers = ("경기", "예측", "예상", "실제결과", "")
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows_out))
        for i in range(len(headers))
    ]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows_out:
        print(fmt.format(*row))

    print(f"  {SEP}")
    if total_y > 0:
        print(f"  어제 적중: {correct_y}/{total_y}  ({correct_y/total_y*100:.0f}%)")
    print()


def run_summary_mode(output_dir: Path, days: int = 30) -> None:
    """최근 N일 예측 결과 vs 실제 결과 요약 테이블 출력 및 저장."""
    import sqlite3

    db_path = os.getenv("KBO_DB_PATH", "kbo_predictor.db")

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT game_id, game_date, home_team, away_team,
                   y_home_win, home_score, away_score
            FROM features
            WHERE game_date >= date('now', ?)
              AND y_home_win IS NOT NULL
            ORDER BY game_date DESC, game_id
            """,
            (f"-{days} days",),
        ).fetchall()

    # prediction JSON 파일에서 game_id별 예측 로드
    pred_by_game: dict[str, dict] = {}
    for json_file in sorted(output_dir.glob("kbo_predictions_*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for pred in data.get("predictions", []):
                gid = pred.get("game_id")
                if gid and gid not in pred_by_game:
                    pred_by_game[gid] = pred
        except Exception:
            pass

    if not rows:
        print("[SUMMARY] DB에 완료된 경기가 없습니다. backfill을 먼저 실행하세요.")
        return

    by_date: dict[str, list] = {}
    for row in rows:
        by_date.setdefault(row[1], []).append(row)

    lines: list[str] = []
    lines.append("# KBO 예측 결과 요약")
    lines.append("")

    total = correct = 0

    for date in sorted(by_date.keys(), reverse=True):
        lines.append(f"## {date}")
        lines.append("")
        lines.append("| 경기 | 예측 | 예상스코어 | 실제결과 | 정오답 |")
        lines.append("|---|:---:|:---:|:---:|:---:|")

        for game_id, _, home, away, y_home_win, home_score, away_score in by_date[date]:
            winner = home if y_home_win == 1 else away
            actual = (
                f"{away} {away_score}:{home_score} {home}"
                if home_score is not None
                else f"{winner} 승"
            )

            pred = pred_by_game.get(game_id)
            if pred:
                hp = float(pred.get("home_win_prob", 0.5))
                pick = home if hp >= 0.5 else away
                conf = max(hp, 1 - hp)
                phs = float(pred.get("home_score_pred", 0))
                pas = float(pred.get("away_score_pred", 0))
                pred_score = f"{away} {pas:.1f}:{phs:.1f} {home}"
                hit = pick == winner
                flag = "✅" if hit else "❌"
                total += 1
                correct += int(hit)
                pred_str = f"{pick} ({conf*100:.0f}%)"
            else:
                pred_str = pred_score = "-"
                flag = "-"

            lines.append(f"| {away} @ {home} | {pred_str} | {pred_score} | {actual} | {flag} |")

        lines.append("")

    lines.append("---")
    lines.append("")
    if total > 0:
        lines.append(f"**총 {total}경기 예측 → {correct}경기 적중 ({correct/total*100:.1f}%)**")
    else:
        lines.append("예측 파일과 매칭된 경기가 없습니다. (output/ 폴더의 JSON 파일을 확인하세요)")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    report_path = output_dir / "kbo_summary.md"
    report_path.write_text(text, encoding="utf-8")
    print(f"[SUMMARY] 저장: {report_path}")


if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("KBO_RUN_MODE", "all").strip().lower()
    output_dir = Path(os.getenv("KBO_ANALYZER_OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "analyze":
        run_once()
    elif mode == "predict":
        run_predict_mode(output_dir)
    elif mode == "summary":
        days = int(os.getenv("KBO_SUMMARY_DAYS", "30"))
        run_summary_mode(output_dir, days=days)
    else:  # "all" or unrecognized: run both
        run_once()
        run_predict_mode(output_dir)
