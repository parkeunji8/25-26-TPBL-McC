# Network용 데이터 설명 
`Network` -> `output_data`에 데이터 있습니다 !!


## 1. 분석 목적

**맥컬러가 어떤 유형의 동료와 함께 뛸 때 성과가 좋았는지**를 찾고,
이를 바탕으로 맥컬러의 활용법에 대한 인사이트를 도출.
(e.g. 맥컬러는 공격적인 가드와 뛸 때 어시스트를 받은 림 근처 득점이 늘었다 ...)

- 대상 선수: 맥컬러 (`player_id` = `10860`, 타오위안 Leopards `team_id` = `5`)
- 시즌: TPBL 2025-26 **정규시즌**
- 선수: 리그 정규시즌 출전 선수 137명
- 경기: 리그 126경기 (공출전 네트워크), 레오파즈 36경기 (나머지 네트워크). 이 중 맥컬러 출전 24경기

---

## 2. 폴더 구성과 용도

| 파일 | 한 행의 단위 | 방향 | 범위 | 내용 |
|---|---|---|---|---|
| `nodes.csv` | 선수 1명 | - | 리그 137명 | 선수 정보, 시즌 스탯, (레오파즈 선수) 온/오프 코트 득실 |
| `edges_coplay.csv` | 같은 팀 선수 쌍 | 무방향 | **리그 전체** | ① 공출전: 두 선수가 함께 뛴 시간 |
| `edges_assist.csv` | 패서 → 득점자 | **방향** | 레오파즈 경기 | ② 어시스트: 누가 누구의 득점을 도왔나 |
| `edges_pair_performance.csv` | 레오파즈 선수 쌍 | 무방향 | 레오파즈 | ③ 2인 조합 성과: 함께 뛸 때의 팀 득실 |
| `edges_mcc_ego.csv` | 맥컬러 ↔ 동료 | 무방향 | 레오파즈 | ④ 맥컬러 중심: 함께 뛸 때 vs 따로 뛸 때 비교 |
| `edges_matchup.csv` | 레오파즈 선수 × 상대 선수 | 무방향 (이분) | 레오파즈 경기 | ⑤ 매치업: 맞대고 뛴 시간과 그동안의 득실 |
| `graphml/*.graphml` | 네트워크 1개 | - | - | 위 ①~⑤의 GraphML 버전 (networkx, igraph, Gephi에서 바로 열림) |
| `report.json` | - | - | - | 데이터 규모 요약, 검증 결과 |

> ④ `edges_mcc_ego.csv`가 분석 목적(맥컬러와 동료의 궁합)에 가장 직접적인 파일임 !!

---

## 3. 공통 규칙

- **ID는 문자열**로 읽어야 함: `pd.read_csv(..., dtype={"player_id": str, "source": str, "target": str, "team_id": str})`
- 모든 간선 파일은 **`source`, `target` = `player_id`**. `nodes.csv`의 `player_id`와 조인
- 모든 간선 파일에 **`weight`** 컬럼이 있음 (파일별 기본 가중치, 아래 표 참고). 다른 컬럼을 가중치로 써도 됨
- 성과가 들어간 간선(③④⑤)은 **모두 Leopards 관점**으로 생성되어 있습니다 !! (`pts_for` = 레오파즈 득점, `pts_against` = 상대 득점)
- **스틴트**: 양 팀 코트 위 10명이 바뀌지 않은 구간. 모든 집계는 5대5 정상 스틴트만 사용
- **포제션(`poss`)**: 추정치 = FGA − OREB + TOV + 0.44 × FTA. 성과 간선의 `poss`는 양 팀 포제션 평균
- **축소된 득실(`net_per100_shrunk`)**: `100 × 득실 / (포제션 + 50)`. 함께 뛴 시간이 짧은 조합이 극단값으로 튀지 않도록 0 쪽으로 당긴 값. 원값은 `net_per100`
- **표본 크기**: 모든 성과 간선에 `minutes`, `poss`, `n_stints`가 있음 → 분석 목적에 맞게 최소 기준을 정해서 걸러낼 것
- 비율 지표에서 **분모가 0이면 빈 값(NaN)**
- **GraphML**: 노드 ID = `player_id`, 노드 속성 = `nodes.csv` 컬럼, 간선 속성 = 각 간선 파일 컬럼. 해당 네트워크에 등장하는 선수만 노드로 포함

| 파일 | `weight` 기본값 |
|---|---|
| `edges_coplay` | `shared_min` (함께 뛴 분) |
| `edges_assist` | `assists` (어시스트 수) |
| `edges_pair_performance` | `net_per100_shrunk` (**음수 가능**) |
| `edges_mcc_ego` | `shared_minutes` (함께 뛴 분) |
| `edges_matchup` | `minutes` (맞대고 뛴 분) |

> `edges_pair_performance`의 `weight`는 음수가 있다 -> 음수 가중치를 지원하지 않는 알고리즘(최단 경로, 일부 중심성 등)에는 그대로 쓰면 안 됨

---

## 4. 파일별 컬럼 설명

### 4-1. `nodes.csv` (선수 137명)

정규시즌 박스스코어 **원값** 기준 (표준화, 축소 없음).

**메타 데이터**

| 컬럼 | 설명 |
|---|---|
| `player_id` | 선수 ID |
| `player_name_original` | 원문 이름 (중국어) |
| `player_name_english` | 영문 이름 |
| `position` | 원본 포지션 표기 (예: `PointGuardAndShootingGuard`) |
| `national_identity` | `Local`(국내) / `Imported`(외국인) |
| `is_imported` | 외국인 선수 여부 |
| `height_cm`, `weight_kg` | 신장, 체중 |
| `age` | 2025-10-01 기준 나이 |
| `team_id`, `team_name_english` | 소속 팀 (정규시즌에 가장 오래 뛴 팀) |
| `is_leopards` | 레오파즈 선수 여부 |
| `is_mcc` | 맥컬러 여부 |

**출전량**

| 컬럼 | 설명 |
|---|---|
| `games` | 정규시즌 출전 경기 수 |
| `minutes` | 정규시즌 총 출전 시간(분), 박스스코어 기준 |
| `min_per_game` | 경기당 출전 시간 |
| `on_court_min_league_stints` | 교체 기록으로 재구성한 5대5 스틴트 기준 출전 시간(분) |

**36분당 스탯**

| 컬럼 | 설명 |
|---|---|
| `PTS_per36` | 득점 |
| `FGA_per36`, `3PA_per36`, `FTA_per36` | 야투, 3점, 자유투 시도 |
| `OREB_per36`, `DREB_per36`, `REB_per36` | 공격, 수비, 전체 리바운드 |
| `AST_per36`, `STL_per36`, `BLK_per36`, `TOV_per36` | 어시스트, 스틸, 블록, 턴오버 |

**슈팅/스타일**

| 컬럼 | 설명 |
|---|---|
| `FG_pct`, `3P_pct`, `FT_pct` | 야투, 3점, 자유투 성공률 |
| `TS_pct` | True Shooting % = 득점 / (2 × (FGA + 0.44 × FTA)) |
| `3PA_rate` | 3점 시도 / 야투 시도 |
| `FT_rate` | 자유투 시도 / 야투 시도 |
| `paint_FGA_share` | 페인트존 시도 / 야투 시도 |
| `AST_TOV` | 어시스트 / 턴오버 |

**온/오프 코트 팀 득실** (레오파즈 선수만 값 있음, 나머지는 빈 값)

| 컬럼 | 설명 |
|---|---|
| `on_minutes`, `on_poss` | 그 선수가 코트에 있을 때 시간(분), 포제션 |
| `on_net_per100` | 코트에 있을 때 팀 100포제션당 득실 |
| `on_net_per100_shrunk` | 위 값의 축소 버전 |
| `off_minutes`, `off_poss` | 그 선수가 벤치에 있을 때 시간(분), 포제션 |
| `off_net_per100` | 벤치에 있을 때 팀 100포제션당 득실 |
| `on_off_net_diff` | `on_net_per100` − `off_net_per100` (그 선수가 있을 때 팀이 얼마나 좋아지나) |

> 온/오프는 레오파즈 36경기 PBP 기준. 박스스코어의 `minutes`와 몇 초 차이 날 수 있음

### 4-2. `edges_coplay.csv` (① 공출전, 리그 전체)

리그 정규시즌 126경기의 교체 기록으로 재구성한 스틴트 기준. 함께 뛴 시간 **60초 이상**인 쌍만 포함.

| 컬럼 | 설명 |
|---|---|
| `source`, `target` | 두 선수의 `player_id` |
| `team_id` | 두 선수의 소속 팀 |
| `n_stints` | 함께 뛴 스틴트 수 |
| `shared_min` | 함께 뛴 시간(분) |
| `overlap` | 함께 뛴 시간 ÷ 두 선수 중 덜 뛴 선수의 출전 시간 (0~1) |
| `jaccard` | 함께 뛴 시간 ÷ 두 선수 출전 시간의 합집합 (0~1) |
| `weight` | = `shared_min` |

> `shared_min`을 그대로 쓰면 출전 시간이 많은 주전끼리만 강하게 연결됨 → 출전량으로 정규화한 `overlap`, `jaccard`를 함께 제공
> 공출전은 **팀 소속**에 강하게 좌우됨 (같은 팀끼리만 같이 뛰니까). 네트워크가 팀 단위로 7개 덩어리로 나뉘는 게 정상

### 4-3. `edges_assist.csv` (② 어시스트, 방향 있음)

레오파즈 36경기 PBP에서 추출. **`source`(패서) → `target`(득점자)** 방향.

| 컬럼 | 설명 |
|---|---|
| `source` | 어시스트한 선수 |
| `target` | 어시스트를 받아 득점한 선수 |
| `team_id` | 두 선수의 소속 팀 |
| `assists` | 어시스트 수 |
| `assists_fg` | 야투 성공으로 이어진 어시스트 |
| `assists_ft` | 슈팅파울 유도 → 자유투로 이어진 어시스트 (이 리그는 이것도 어시스트로 기록) |
| `pts_created` | 어시스트로 연결된 득점 (자유투 어시스트는 연결된 자유투 1개 기준이라 실제보다 작게 잡힘) |
| `shared_min` | 두 선수가 같은 경기들에서 함께 뛴 시간(분) |
| `assists_per36_shared` | 함께 뛴 36분당 어시스트 (출전량 차이를 보정한 값) |
| `coverage` | `leopards_all_games`: 레오파즈 선수 (정규시즌 전 경기 반영) / `vs_leopards_only`: 상대 팀 선수 (레오파즈전만 반영) |
| `weight` | = `assists` |

> 상대 팀 선수 간선(`vs_leopards_only`)은 레오파즈전 몇 경기만의 값이라 표본이 매우 적음. 레오파즈 내부 패스 네트워크를 볼 때는 `coverage == 'leopards_all_games'`로 필터링
> 어시스트 중 약 3%는 연결할 득점을 찾지 못해 빠져 있음

### 4-4. `edges_pair_performance.csv` (③ 레오파즈 2인 조합 성과)

레오파즈 선수 두 명이 **함께 코트에 있을 때** 팀 성과. 최소 시간 제한 없음 (전부 포함).

| 컬럼 | 설명 |
|---|---|
| `source`, `target` | 두 선수의 `player_id` |
| `n_stints` | 함께 뛴 스틴트 수 |
| `minutes` | 함께 뛴 시간(분) |
| `poss` | 함께 뛴 동안의 포제션 (양 팀 평균) |
| `pts_for`, `pts_against` | 그동안 레오파즈 득점, 실점 |
| `margin` | 득실 (`pts_for` − `pts_against`) |
| `ortg` | 100포제션당 득점 |
| `drtg` | 100포제션당 실점 |
| `net_per100` | 100포제션당 득실 |
| `net_per100_shrunk` | 축소된 100포제션당 득실 |
| `weight` | = `net_per100_shrunk` (음수 가능) |

### 4-5. `edges_mcc_ego.csv` (④ 맥컬러 ↔ 동료)

맥컬러와 함께 뛴 적이 있는 레오파즈 동료 1명당 1행. **"그 동료와 함께 뛸 때"와 "따로 뛸 때"를 비교**하는 파일.

컬럼 이름 규칙

| 접두사 | 기준 구간 | 의미 |
|---|---|---|
| `team_with_*` | 맥컬러 ON + 동료 ON | 둘이 함께 뛸 때 팀 성과 |
| `team_without_*` | 맥컬러 ON + 동료 OFF | 맥컬러가 그 동료 **없이** 뛸 때 팀 성과 |
| `mcc_with_*` | 맥컬러 ON + 동료 ON | 둘이 함께 뛸 때 **맥컬러** 스탯 |
| `mcc_without_*` | 맥컬러 ON + 동료 OFF | 그 동료 없을 때 **맥컬러** 스탯 |
| `tm_with_mcc_*` | 동료 ON + 맥컬러 ON | 둘이 함께 뛸 때 **동료** 스탯 |
| `tm_without_mcc_*` | 동료 ON + 맥컬러 OFF | 맥컬러 없을 때 **동료** 스탯 |
| `*_diff_*`, `team_net_diff` | - | 함께 − 따로 (양수면 함께 뛸 때 값이 더 큼) |

**식별**

| 컬럼 | 설명 |
|---|---|
| `source` | 맥컬러 (`10860`) |
| `target` | 동료 `player_id` |
| `target_name`, `target_position`, `target_is_imported` | 동료 영문 이름, 포지션, 외국인 여부 |
| `shared_minutes` | 둘이 함께 뛴 시간(분) (= `team_with_minutes`) |
| `weight` | = `shared_minutes` |

**팀 성과** (`team_with_*`, `team_without_*`)

| 접미사 | 설명 |
|---|---|
| `minutes`, `poss` | 해당 구간 시간(분), 포제션 |
| `pts_for`, `pts_against`, `margin` | 레오파즈 득점, 실점, 득실 |
| `ortg`, `drtg` | 100포제션당 득점, 실점 |
| `net_per100`, `net_per100_shrunk` | 100포제션당 득실, 축소 버전 |

**개인 스탯** (`mcc_with_*`, `mcc_without_*`, `tm_with_mcc_*`, `tm_without_mcc_*`)

| 접미사 | 설명 |
|---|---|
| `minutes` | 해당 구간 시간(분) |
| `FGA` | 야투 시도 수 (비율 지표의 표본 크기 확인용) |
| `PTS_per36`, `FGA_per36`, `REB_per36`, `AST_per36`, `TOV_per36` | 36분당 득점, 야투 시도, 리바운드, 어시스트, 턴오버 |
| `TS_pct` | True Shooting % |
| `3PA_rate` | 3점 시도 / 야투 시도 |
| `FT_rate` | 자유투 시도 / 야투 시도 |
| `rim_share` | 림 존(골대 1.25m 이내) 시도 / 야투 시도 |
| `paint_share` | 페인트존(림 존 제외) 시도 / 야투 시도 |
| `mid_share` | 미드레인지 시도 / 야투 시도 |
| `three_share` | 3점(코너 + 탑) 시도 / 야투 시도 |
| `assisted_share` | 어시스트 받은 야투 성공 / 야투 성공 |
| `usage_share` | (FGA + 0.44 × FTA + TOV) / 팀 포제션 (팀 공격 중 그 선수가 끝낸 비중) |

> 슛 존은 코트 좌표(FIBA 규격 28m × 15m)로 계산

**차이** (함께 − 따로)

| 컬럼 | 설명 |
|---|---|
| `team_net_diff` | `team_with_net_per100` − `team_without_net_per100` |
| `mcc_diff_{지표}` | `mcc_with_{지표}` − `mcc_without_{지표}` |
| `tm_diff_{지표}` | `tm_with_mcc_{지표}` − `tm_without_mcc_{지표}` |

`{지표}` = `PTS_per36`, `TS_pct`, `3PA_rate`, `rim_share`, `paint_share`, `mid_share`, `three_share`, `assisted_share`, `usage_share`, `AST_per36`, `REB_per36`, `TOV_per36`

> 예시 해석: 어떤 가드의 `mcc_diff_assisted_share`가 양수고 `mcc_diff_rim_share`도 양수면 → 그 가드와 뛸 때 맥컬러가 어시스트 받은 림 근처 득점이 늘었다

### 4-6. `edges_matchup.csv` (⑤ 매치업, 이분 그래프)

`source` = 레오파즈 선수, `target` = 상대 선수. 두 선수가 동시에 코트에 있었던 구간 기준.

| 컬럼 | 설명 |
|---|---|
| `source` | 레오파즈 선수 |
| `target` | 상대 선수 |
| `target_team_id` | 상대 선수 소속 팀 |
| `n_stints`, `minutes`, `poss` | 맞대고 뛴 스틴트 수, 시간(분), 포제션 |
| `pts_for`, `pts_against`, `margin` | 그동안 레오파즈 득점, 실점, 득실 |
| `ortg`, `drtg`, `net_per100`, `net_per100_shrunk` | 100포제션당 득점, 실점, 득실, 축소 버전 |
| `weight` | = `minutes` |

> 코트 위 10명 전체의 결과라서 "이 두 선수의 1대1 대결 결과"가 아님. 상대 팀당 3~4경기뿐이라 표본이 적음

### 4-7. `report.json`

| 키 | 설명 |
|---|---|
| `phase` | 사용한 국면 (`regular_season`) |
| `league_stints`, `league_games` | 공출전 네트워크에 쓴 리그 스틴트 수, 경기 수 |
| `leopards_stints`, `leopards_games` | 레오파즈 PBP 스틴트 수, 경기 수 |
| `leopards_margin_sum_stints` | 레오파즈 스틴트 득실 합계 |
| `leopards_margin_sum_official` | 레오파즈 공식 점수차 합계 |
| `nodes`, `leopards_nodes` | 전체 선수 수, 레오파즈 선수 수 |
| `edges` | 네트워크별 간선 수 |
| `mcc_minutes` | 맥컬러 출전 시간(분), 5대5 스틴트 기준 |
| `graphml` | 생성된 GraphML 파일 목록 |

> `leopards_margin_sum_stints`와 `leopards_margin_sum_official`의 차이 = 교체 기록 오류로 제외된 구간(1503 경기 일부)에서 난 득실

---

## 5. 데이터 관련 참고사항

1. **데이터 검증**: PBP 이벤트를 합산한 값이 레오파즈 경기의 최종 점수, 선수별 박스스코어(득점/야투/3점/자유투/리바운드/어시스트/스틸/블록/턴오버/파울)와 모두 일치. 출전 시간은 선수-경기의 약 96%가 공식 기록과 1초 이내
2. **범위 차이**: 리그 전체를 커버하는 건 ① 공출전뿐. ②~⑤는 레오파즈 경기 PBP로만 만들 수 있어서 레오파즈 중심 (리그 전체 PBP는 아직 미수집)
3. **표본 크기**: 스틴트 대부분이 1분 미만이고, 맥컬러와 의미 있는 시간을 함께 뛴 국내 동료는 7명 정도. 짧게 함께 뛴 조합의 득실/비율 지표는 변동이 매우 큼 → `minutes`, `poss`, `FGA` 기준으로 걸러서 볼 것
4. **외국인 선수 규정**: TPBL은 외국인 2명 동시 출전이 가능해서, 맥컬러 출전 시간의 약 99%에 다른 외국인 선수 1명이 함께 있음. 즉 `edges_mcc_ego`에서 외국인 동료(`target_is_imported == True`)의 "따로 뛸 때"는 대부분 **다른 외국인과 뛸 때**를 의미함. 추후 KBL(외국인 동시 출전 1명) 적용 시 국내 선수 조합을 따로 봐야 함
5. **"함께 vs 따로" 비교의 한계**: 동료가 없을 때는 다른 선수가 그 자리를 채우고, 경기 상황(점수 차, 상대 라인업, 시간대)도 달라짐. 차이 값은 그 동료만의 효과가 아니라 **로테이션 전체의 차이**가 섞여 있음