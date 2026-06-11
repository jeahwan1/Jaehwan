# KBO Analyzer + Predictor + Ollama

KBO 공식 엔드포인트 기반으로 라인업/상대전적/스플릿 데이터를 수집하고,
분석 리포트 및 승패 예측을 생성합니다. 예측 결과를 Ollama 에이전트에 전달해
AI 해설 리포트까지 생성할 수 있습니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 환경변수

`.env.example`을 `.env`로 복사 후 필요 시 수정:

- `KBO_RUN_MODE=analyze|predict`
- `KBO_BACKFILL_DAYS=30`
- `KBO_ANALYZER_OUTPUT_DIR=./output`
- `KBO_USE_OLLAMA=1`
- `OLLAMA_HOST=http://localhost:11434`
- `OLLAMA_MODEL=llama3.1:8b`
- `KBO_OLLAMA_AUTO_MODEL=1` (요청 모델 없을 때 설치된 첫 모델 자동 사용)

## 실행

### 1) 분석 리포트

```bash
python main.py
```

출력: `output/kbo_analysis_*.md`

### 2) 승패 예측 + Ollama 리포트

```bash
set KBO_RUN_MODE=predict
python main.py
```

출력:
- `output/kbo_predictions_*.json`
- `output/kbo_ai_report_*.md` (Ollama 성공 시)
- `output/kbo_ai_prompt_*.txt` (Ollama 실패 시 디버그용)
- `output/kbo_ai_debug_*.txt` (Ollama 실패 원인 점검 정보)

## Ollama 준비

```bash
ollama serve
ollama pull llama3.1:8b
```

## 현재 MVP 예측 피처

- 홈 어드밴티지
- 홈/원정 타선 공격 점수(선발투수 split 기반)
- 홈/원정 최근 5경기 승률
- 홈/원정 최근 5경기 득점/실점
- 구장 팩터
- 경기 시점 날씨(기온/풍속/강수량)
