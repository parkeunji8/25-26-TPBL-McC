## Play-By-Play 데이터 수집

https://tpbl.basketball/schedule/{GameID}/detail

1. 맥컬러 경기 아이디 가져오기 (`find_ids.py` -> `stat.json`, `leopards_25_26_game_ids.json` 생성)
   1. request+bs 구조 안 먹힘 & api 직접 호출 안됨. (초기 HTML에 데이터 없는 듯)
   2. url 상의 GameID를 자동 수집할 방법이 없어서 ID 역산.
   3. ID 규칙: 261019 첫 경기(게임5) = 1375(1370+5), 261026 두 번째 경기(게임10) = 1380(1370+10). 25-26시즌이 총 126경기 이므로 게임1(1371) ~ 게임126(1496) 중 Leopards(McC 팀) 들어가는 ID 필터링.
   5. 260425(game118) 번호가 갑자기 1503이라 수동 추가함.

2. 해당 36경기 게임 아이디에 대해 play-by-play stats 크롤링 (`crawl.py`)
   1. `raw_games` 폴더에 36개 경기 각각 원본 텍스트 저장 (대만어)

3. 맥컬러 출전 경기만 필터링 (`filter_mcc.py` -> `mcc_game_ids.json`, `raw_games_mcc` 폴더 생성)
   1. 외국인 선수 출전 제한 규칙으로 인해 총 24경기만 출전.
   2. `raw_games` 폴더에서 맥컬러 이름 있는 경기로 필터링

4. 한국어/영어 용어집 생성 (클로드가 선수들 이름 대조해서 `name_glossary_eng.json` 생성)
   1. 선수/팀 이름이 똑같은 영어 표기로 나오게 하기 위함 (클로드가 위키백과 대조해서 가져옴)
   2. (참고) `extract_names.py`는 안 씀. 이걸론 이름/그냥 단어 구별 잘 못해서 포기. 단 새로운 선수 들어오면 재활용 가능

5. 크롤링 결과 번역 (`translate_games.py` -> `traslated_games` 폴더 생성)
   1. 선수/팀 이름은 영어로 번역
   2. 그 외는 한국어로 번역.
   3. 경기장이나 심판이름은 일단 번역 안함(시간 나면 할 듯)

6. 맥컬러 온코트 타임스탬프 (`analyze_mcc_stats.py` -> `mcc_oncourt_stats_우리팀`, `mcc_oncourt_stats_상대팀.csv` 생성)
   1. 맥컬러 선수가 경기에 뛰고 있는 시간 추출
   2. 해당 타임라인에서 주요 스탯 기록한 선수들과 이벤트들 추출 (우리팀/상대팀 나눠서 파일 생성)

7. 분석용 csv 생성 (`analyze_mcc_stats.py` -> `mcc_우리팀_fit.csv`, `mcc_lineup_stints.csv` 생성)
* 7-1 `mcc_우리팀_fit.csv`
   1. 라인업 공존시간 추가 (같이 뛴 시간을 절대적으로 분석하면 이상치 못 거름)
   2. CoOccur_Min은 24경기 누적해서 맥컬러랑 같이 뛴 시간
   3. 스탯은 per36 비율로 변환 ((누적 스탯 / CoOccur_Min) * 36)
* 7-2 `mcc_lineup_stints.csv`
   1. mcc_우리팀_fit.csv는 표본이 너무 적어서 (14명) 클러스터링 힘듦
   2. 따라서 stint(on court 구간) 기준으로 데이터 생성해서 안정적인 클러스터링 유도.

## csv_raw 폴더 설명

1. `mcc_우리팀_fit`: 맥컬러와 우리팀 선수가 얼마나 잘 맞았는지. 선수 개개인을 클러스터링에 쓰기 좋게 가공.
2. `mcc_linup_stints`: 5인 조합 구간 (선수 교체되기 전까지) -> 어떤 조합에서 가장 좋았나. 조합별 클러스터 가능.
3. `mcc_oncourt_stats_우리팀`: mcc_우리팀_fit 가공 전 파일. (온코트 구간 비율 변환 전)
4. `mcc_oncourt_stats_상대팀`: 맥컬러 상대로 어떤 선수가 잘하는지 분석 가능. BUT 데이터 너무 적음.
