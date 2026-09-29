# GNN용 데이터 설명 
(`GNN` -> `output_data` -> `gnn_dataset`에 데이터 있습니다 !!)


## 1. 분석 목적

**맥컬러가 어떤 유형의 동료와 함께 뛸 때 성과가 좋았는지**를 찾고,
이를 바탕으로 맥컬러의 활용법에 대한 인사이트를 도출.
(e.g. 맥컬러는 공격적인 가드와 뛸 때 어시스트를 받은 림 근처 득점이 늘었다 ...)

- 대상 선수: 크리스 맥컬러 (Chris McCullough, `player_id` = `10860`, 타오위안 레오파즈 `team_id` = `5`)
- 시즌: TPBL 2025-26
- 선수 데이터: 리그 정규시즌 출전 선수 137명
- 경기별 PBP(스틴트) 데이터: 레오파즈 42경기 (정규시즌 36, 프리시즌 2, 플레이오프 4). 이 중 맥컬러 출전 28경기

---

## 2. 폴더 구성과 용도

| 파일 | 한 행의 단위 | 용도 |
|---|---|---|
| `nodes.csv` | 선수 1명 | 선수 피처 (모델 입력 가능) |
| `feature_columns.json` | - | `nodes.csv` 중 모델 입력용 컬럼 목록 |
| `edges_teammate.csv` | 같은 팀 선수 쌍 | 리그 전체 공출전 관계 (모델 입력 가능) |
| `edges_opponent.csv` | 상대 팀 선수 쌍 | 리그 전체 매치업 관계 (모델 입력 가능) |
| `hard_negatives.csv` | 같은 팀 선수 쌍 | 같은 팀인데 거의 같이 안 뛴 쌍 |
| `edges_assist.csv` | 패서 → 득점자 | 어시스트 관계 (레오파즈 경기만 커버) |
| `stint_graphs.csv` | 스틴트 1개 | 스틴트별 라인업, 맥락, 결과 (전체 스틴트, 검토용) |
| `stint_graphs.npz` | 스틴트 1개 | `stint_graphs.csv` 중 `use == True`만 배열로 저장 |
| `stint_nodes.csv` | 스틴트 × 코트 위 선수 | 스틴트 안에서 각 선수가 기록한 스탯 (**모델 입력 금지**, 해석 전용) |
| `summary.json` | - | 데이터 규모 요약 |

---

## 3. 공통 규칙

- **ID는 문자열**로 읽어야 함: `pd.read_csv(..., dtype={"player_id": str, "game_id": str, "team_id": str, "stint_id": str})`
- **`node_idx`**: 선수 137명에게 부여한 0~136 정수 인덱스. 모든 파일의 `*_idx` 컬럼은 `nodes.csv`의 `node_idx`를 가리킴. `-1`은 `nodes.csv`에 없는 선수(프리시즌에만 뛴 선수 등)
- **스틴트**: 양 팀 코트 위 10명이 바뀌지 않은 구간. 선수 교체가 일어나면 새 스틴트가 시작됨
- **모든 스틴트 데이터는 Leopards (team_id `5`) 관점**으로 생성되어 있습니다 !! (`leo_*` = 레오파즈, `opp_*` = 상대 팀)
- **시간**: `*_sec`은 경기 경과 초 (0 = 경기 시작, 2880 = 정규시간 종료, 연장은 5분씩 추가)
- **경기 국면**: `regular_season`(정규시즌), `playoffs`(플레이오프), `preseason`(프리시즌)
- **리그 평균 축소(shrinkage)**: 출전이 적은 선수의 비율 스탯이 극단값으로 튀지 않도록 `(개인 합계 + k × 리그 평균) / (개인 분모 + k)`로 계산. 표본이 적을수록 리그 평균에 가까워짐
- **무방향 관계 파일**(`edges_teammate`, `edges_opponent`, `hard_negatives`)은 한 쌍을 **한 번만** 저장 (`src` < `dst`, 문자열 비교 기준)

---

## 4. 파일별 컬럼 설명

### 4-1. `nodes.csv` (선수 137명)

2025-26 **정규시즌** 박스스코어 합계로 계산. 모델 입력에 쓸 컬럼은 `feature_columns.json` 참고 (4-2).

**메타 데이터** (모델 입력 X)

| 컬럼 | 설명 |
|---|---|
| `player_id` | 선수 ID |
| `node_idx` | 노드 인덱스 (0~136) |
| `player_name_original` | 원문 이름 (중국어) |
| `player_name_english` | 영문 이름 |
| `position` | 원본 포지션 표기 (e.g. `PointGuardAndShootingGuard`) |
| `national_identity` | `Local`(국내) / `Imported`(외국인) |
| `team_id`, `team_name_english` | 소속 팀 (정규시즌 스틴트에서 가장 오래 뛴 팀) |
| `stint_on_court_min` | 재구성한 스틴트 기준 정규시즌 출전 시간(분) |

**신체 및 포지션**

| 컬럼 | 설명 |
|---|---|
| `height_cm`, `weight_kg`, `bmi` | 신장, 체중, BMI |
| `age` | 2025-10-01 기준 나이 |
| `is_imported` | 외국인 선수 여부 (1/0) |
| `pos_PG`, `pos_SG`, `pos_SF`, `pos_PF`, `pos_C` | 포지션 멀티핫 (겸업 포지션이면 둘 다 1) |
| `pos_num` | 포지션 번호 평균 (PG=1 ~ C=5). e.g. PG/SG 겸업이면 1.5 |

**출전량**

| 컬럼 | 설명 |
|---|---|
| `minutes` | 정규시즌 총 출전 시간(분) |
| `log_minutes` | log(1 + minutes) |
| `min_per_game` | 경기당 출전 시간 |
| `low_minutes` | 정규시즌 200분 미만이면 1. 이 선수들의 스탯은 리그 평균 쪽으로 강하게 축소되어 있음 |

**36분당 스탯** (출전 시간 기준 축소, k = 200분)

| 컬럼 | 설명 |
|---|---|
| `points_p36` | 득점 |
| `FGA_p36`, `three_PA_p36`, `FTA_p36` | 야투 시도, 3점 시도, 자유투 시도 |
| `OREB_p36`, `DREB_p36` | 공격 리바운드, 수비 리바운드 |
| `AST_p36`, `STL_p36`, `BLK_p36`, `TOV_p36`, `PF_p36` | 어시스트, 스틸, 블록, 턴오버, 파울 |
| `points_in_paint_p36` | 페인트존 득점 |
| `fast_break_points_p36` | 속공 득점 |
| `second_chance_points_p36` | 세컨드 찬스 득점 |

**슈팅 효율** (시도 수 기준 축소)

| 컬럼 | 설명 |
|---|---|
| `FG_pct` | 야투 성공률 (k = 100 시도) |
| `three_P_pct` | 3점 성공률 (k = 50 시도) |
| `FT_pct` | 자유투 성공률 (k = 30 시도) |
| `TS_pct` | True Shooting % = 득점 / (2 × (FGA + 0.44 × FTA)) |

**슛 스타일** (야투 시도 중 비중, k = 50 시도로 축소)

| 컬럼 | 설명 |
|---|---|
| `three_PA_rate` | 3점 시도 / 야투 시도 |
| `FT_rate` | 자유투 시도 / 야투 시도 |
| `paint_FGA_share` | 페인트존 시도 비중 |
| `fastbreak_FGA_share` | 속공 시도 비중 |
| `shot_rim_share` | 림 어택: 레이업, 드라이빙 레이업, 덩크, 풋백, 앨리웁 |
| `shot_mid_catch_share` | 2점 점프슛 (캐치 앤 슛형) |
| `shot_mid_create_share` | 2점 풀업, 스텝백, 페이드어웨이, 턴어라운드 (자체 창출형) |
| `shot_float_hook_share` | 플로터, 훅슛 |
| `shot_three_catch_share` | 3점 점프슛 (캐치 앤 슛형) |
| `shot_three_create_share` | 3점 풀업, 스텝백 등 (자체 창출형) |

> 슛 스타일은 리그 공식 슛 유형 분류 기준 (좌표 기반 존 아님). 6개 `shot_*_share`의 합은 약 1

**역할 지표**

| 컬럼 | 설명 |
|---|---|
| `AST_TOV` | 어시스트 / 턴오버 (36분당 축소값끼리 나눔) |
| `OREB_share` | 공격 리바운드 / 전체 리바운드 |

**표준화 컬럼 `z_*`**
위 연속형 컬럼들을 137명 기준으로 z-score 변환한 값 (평균 0, 표준편차 1). e.g. `z_points_p36`

### 4-2. `feature_columns.json`

| 키 | 설명 |
|---|---|
| `model_feature_columns` | 모델 입력용 45개 컬럼: `z_*` 38개 + 이진 7개 (`is_imported`, `pos_PG`~`pos_C`, `low_minutes`) |
| `continuous` | 표준화 전 연속형 원본 컬럼 목록 |
| `binary` | 이진 컬럼 목록 |
| `edge_attr_columns` | 관계 파일의 수치 속성 컬럼 (`overlap`, `jaccard`, `log_shared_min`) |
| `config` | 생성 설정값 (국면 필터, 축소 강도, 최소 공출전 시간 등) |

### 4-3. `edges_teammate.csv` / `edges_opponent.csv`

리그 **정규시즌 전체** 5대5 정상 스틴트 기준.
- `edges_teammate`: 같은 팀으로 함께 뛴 쌍. 함께 뛴 시간 60초 이상만 포함
- `edges_opponent`: 서로 상대 팀으로 코트에 같이 있었던 쌍. 최소 시간 제한 없음

| 컬럼 | 설명 |
|---|---|
| `src`, `dst` | 두 선수의 `player_id` (`src` < `dst`) |
| `src_idx`, `dst_idx` | 두 선수의 `node_idx` |
| `shared_sec` | 함께(teammate) / 맞대고(opponent) 뛴 시간(초) |
| `n_stints` | 함께 / 맞대고 뛴 스틴트 수 |
| `overlap` | 공유 시간 ÷ 두 선수 중 덜 뛴 선수의 출전 시간 (0~1) |
| `jaccard` | 공유 시간 ÷ 두 선수 출전 시간의 합집합 (0~1) |
| `log_shared_min` | log(1 + 공유 분) |
| `edge_type` | `teammate` / `opponent` |
| `same_team` | teammate면 True, opponent면 False |

> 공유 시간을 그대로 쓰면 출전 시간이 많은 주전끼리만 강하게 연결되어서, 출전량으로 정규화한 `overlap`, `jaccard`를 함께 제공
> 공출전 관계는 **팀 소속**에 강하게 좌우되는 특성이 있음 (같은 팀끼리만 같이 뛰니까)

### 4-4. `hard_negatives.csv`

같은 팀 소속인데 정규시즌 동안 함께 뛴 시간이 **60초 이하**인 쌍. (e.g. 같은 포지션이라 교대로 기용되는 선수들)

| 컬럼 | 설명 |
|---|---|
| `src`, `dst` | 두 선수의 `player_id` |
| `src_idx`, `dst_idx` | 두 선수의 `node_idx` |
| `team_id` | 소속 팀 |
| `shared_sec` | 함께 뛴 시간(초), 0~60 |

### 4-5. `edges_assist.csv`

레오파즈 42경기 PBP에서 추출한 어시스트 관계. **방향 있음** (패서 → 득점자). 국면 구분 없이 42경기 전체 합산.

| 컬럼 | 설명 |
|---|---|
| `src` | 어시스트한 선수 `player_id` |
| `dst` | 어시스트를 받아 득점한 선수 `player_id` |
| `src_idx`, `dst_idx` | 각 `node_idx` |
| `team_id` | 두 선수의 소속 팀 |
| `assists` | 어시스트 수 |
| `assists_fg` | 야투 성공으로 이어진 어시스트 |
| `assists_ft` | 슈팅파울 유도 → 자유투로 이어진 어시스트 (이 리그는 이것도 어시스트로 기록) |
| `pts_created` | 어시스트로 연결된 득점 (자유투 어시스트는 연결된 자유투 1개 기준이라 실제보다 작게 잡힘) |
| `shared_min` | 두 선수가 42경기 동안 함께 뛴 시간(분) |
| `assists_per36_shared` | 함께 뛴 36분당 어시스트 수 |
| `coverage` | `leopards_all_games`: 레오파즈 선수 (전 경기 반영) / `vs_leopards_only`: 상대 선수 (레오파즈전만 반영) |

> 어시스트 중 약 3%는 연결할 슛을 찾지 못해 이 파일에서 빠져 있음

### 4-6. `stint_graphs.csv` (스틴트 1개 = 1행)

레오파즈 42경기의 **모든** 스틴트. 학습에 쓸 스틴트는 `use == True`로 필터링.

**식별/시간**

| 컬럼 | 설명 |
|---|---|
| `stint_id` | `{game_id}_{순번}` (e.g. `1375_001`) |
| `game_id` | 경기 ID |
| `competition_phase` | 경기 국면 |
| `quarter` | 쿼터 (5 이상은 연장) |
| `start_sec`, `end_sec` | 시작 및 종료 시각 (경기 경과 초) |
| `duration_sec` | 스틴트 길이(초) |

**라인업**

| 컬럼 | 설명 |
|---|---|
| `leo_ids`, `opp_ids` | 레오파즈 / 상대 코트 위 선수 `player_id`, `\|`로 구분 |
| `leo_node_idx`, `opp_node_idx` | 위 선수들의 `node_idx`, `\|`로 구분 (`-1` = 노드 없음) |
| `opp_team_id` | 상대 팀 ID |
| `mcc_on` | 맥컬러가 코트에 있었는지 |

**맥락** (스틴트 시작 시점에 알 수 있는 정보)

| 컬럼 | 설명 |
|---|---|
| `leo_is_home` | 레오파즈 홈 경기 여부 |
| `margin_at_start` | 스틴트 시작 시점 점수 차 (레오파즈 − 상대) |
| `sec_remaining_regulation` | 정규시간 기준 남은 시간(초). 연장에서는 0 |
| `is_ot` | 연장전 여부 |
| `garbage_time` | 4쿼터 이후 + 시작 시점 점수 차 20점 이상 |
| `opp_prior_ORtg`, `opp_prior_DRtg`, `opp_prior_Net_Rating` | 상대 팀의 **해당 경기 이전** 경기들 평균 공격/수비/순 레이팅 (해당 경기 결과 미포함) |
| `opp_prior_games` | 위 평균에 쓰인 이전 경기 수. 0이면 리그 평균값으로 대체됨 |

**팀 결과**

| 컬럼 | 설명 |
|---|---|
| `leo_pts`, `opp_pts` | 스틴트 동안 양 팀 득점 |
| `margin` | 득실 (`leo_pts` − `opp_pts`) |
| `leo_poss`, `opp_poss` | 추정 포제션 = FGA − OREB + TOV + 0.44 × FTA |
| `poss_avg` | 양 팀 포제션 평균 |
| `net_per100` | 100포제션당 득실 = 100 × `margin` / `poss_avg` (포제션 0이면 빈 값) |
| `leo_ortg`, `leo_drtg` | 레오파즈 100포제션당 득점 / 실점 |

> 스틴트 대부분이 1분 미만이라 `net_per100`은 ±150까지 나올 정도로 변동이 큼

**맥컬러 개인 결과** (`mcc_on == True`일 때만 값, 아니면 빈 값)

| 컬럼 | 설명 |
|---|---|
| `mcc_PTS` | 득점 |
| `mcc_FGA`, `mcc_FGM` | 야투 시도 / 성공 |
| `mcc_2PA`, `mcc_2PM`, `mcc_3PA`, `mcc_3PM` | 2점 / 3점 시도 및 성공 |
| `mcc_FTA`, `mcc_FTM` | 자유투 시도 / 성공 |
| `mcc_OREB`, `mcc_DREB` | 공격 / 수비 리바운드 |
| `mcc_AST`, `mcc_STL`, `mcc_BLK`, `mcc_TOV`, `mcc_PF` | 어시스트, 스틸, 블록, 턴오버, 파울 |
| `mcc_FGM_assisted` | 어시스트 받은 야투 성공 |
| `mcc_FGM_unassisted` | 혼자 만든 야투 성공 |
| `mcc_rim_FGA`, `mcc_rim_FGM` | 림 존 시도 / 성공 (골대 1.25m 이내) |
| `mcc_paint_FGA`, `mcc_paint_FGM` | 페인트존 시도 / 성공 (림 존 제외) |
| `mcc_mid_FGA`, `mcc_mid_FGM` | 미드레인지 시도 / 성공 (그 외 2점) |
| `mcc_c3_FGA`, `mcc_c3_FGM` | 코너 3점 시도 / 성공 (베이스라인에서 2.99m 이내) |
| `mcc_ab3_FGA`, `mcc_ab3_FGM` | 탑(윙/정면) 3점 시도 / 성공 |
| `mcc_TSA` | 슈팅 시도 = FGA + 0.44 × FTA |
| `mcc_usage_poss` | 맥컬러가 끝낸 포제션 = FGA + 0.44 × FTA + TOV |
| `mcc_usage_share` | `mcc_usage_poss` / `leo_poss` (팀 포제션 중 맥컬러 비중) |

> 슛 존은 코트 좌표(FIBA 규격 28m × 15m)로 계산. 존 컬럼 합 = FGA (좌표가 없는 슛이 있으면 그만큼 존 집계에서 빠짐)

**필터/분할**

| 컬럼 | 설명 |
|---|---|
| `valid` | 양 팀 모두 5명인 정상 라인업 여부 |
| `all_nodes_known` | 10명 모두 `nodes.csv`에 있는지 |
| `use` | **학습 사용 권장 여부** = `valid` & `duration_sec` > 0 & `all_nodes_known` |
| `fold` | 경기 단위 그룹 번호 (0~4). 경기를 날짜순으로 돌아가며 배정, 같은 경기의 스틴트는 모두 같은 fold |

> 1502(프리시즌), 1503(정규시즌) 경기는 원본 교체 기록 오류로 일부 구간이 `valid == False`

### 4-7. `stint_graphs.npz`

`stint_graphs.csv`에서 `use == True`인 스틴트만 배열로 저장. 행 순서는 모든 키에서 동일.
문자열 배열이 있어서 `np.load(..., allow_pickle=True)`로 불러와야 함.

| 키 | 형태 | 설명 |
|---|---|---|
| `stint_id`, `game_id` | (S,) | 식별자 |
| `phase` | (S,) | 경기 국면 (= `competition_phase`) |
| `fold` | (S,) | 경기 단위 그룹 번호 |
| `leo_idx` | (S, 5) | 레오파즈 5명의 `node_idx` |
| `opp_idx` | (S, 5) | 상대 5명의 `node_idx` |
| `mcc_pos` | (S,) | 맥컬러가 `leo_idx`의 몇 번째 열인지 (0~4). 코트에 없으면 -1 |
| `duration_sec` | (S,) | 스틴트 길이(초) |
| `ctx` | (S, 8) | 맥락 피처. 컬럼 순서는 `ctx_cols` |
| `ctx_cols` | (8,) | `leo_is_home`, `margin_at_start`, `sec_remaining_regulation`, `is_ot`, `garbage_time`, `opp_prior_ORtg`, `opp_prior_DRtg`, `opp_prior_Net_Rating` |
| `y_team` | (S, 7) | 팀 결과. 컬럼 순서는 `y_team_cols` |
| `y_team_cols` | (7,) | `margin`, `net_per100`, `leo_pts`, `opp_pts`, `leo_poss`, `opp_poss`, `poss_avg` |
| `y_mcc` | (S, 31) | 맥컬러 개인 결과. `mcc_pos == -1`인 행은 NaN |
| `y_mcc_cols` | (31,) | 4-6의 `mcc_*` 컬럼 31개 (`mcc_on` 제외) |

### 4-8. `stint_nodes.csv` (스틴트 × 코트 위 선수)

스틴트 * 코트 위 선수 10명, 각 선수가 **그 구간에서 기록한 스탯** 데이터. (기록이 없으면 0)
레오파즈 42경기의 모든 스틴트 포함 → 학습 스틴트만 보려면 `stint_graphs.csv`의 `use`와 `stint_id`로 조인.

**이 파일은 모델 입력으로 쓰면 안 됩니다 !!** (결과값이라 타깃이 새어 들어감)

| 컬럼 | 설명 |
|---|---|
| `stint_id`, `game_id` | 스틴트, 경기 ID |
| `player_id`, `node_idx` | 선수 ID, 노드 인덱스 (`-1` = 노드 없음) |
| `team_id` | 소속 팀 |
| `side` | `leo`(레오파즈) / `opp`(상대) |
| `is_mcc` | 맥컬러 여부 |
| `duration_sec` | 스틴트 길이(초) |
| `PTS`, `FGA`, `FGM`, `2PA`, `2PM`, `3PA`, `3PM`, `FTA`, `FTM` | 득점, 슈팅 |
| `OREB`, `DREB`, `AST`, `STL`, `BLK`, `TOV`, `PF` | 리바운드, 어시스트, 스틸, 블록, 턴오버, 파울 |
| `FGM_assisted`, `FGM_unassisted` | 어시스트 받은 / 혼자 만든 야투 성공 |
| `rim_*`, `paint_*`, `mid_*`, `c3_*`, `ab3_*` (`_FGA`, `_FGM`) | 슛 존별 시도 / 성공 (정의는 4-6과 동일) |

이후 완성된 모델의 결과 해석에 활용 가능한 데이터임.
- 특정 역할 유형 동료와 뛸 때 맥컬러의 슛 존 구성이 어떻게 바뀌었는가
- 맥컬러와 함께 뛸 때 가드들의 어시스트가 늘었는가
- 모델이 "좋다"고 판단한 조합에서 실제로 무엇이 달라졌는가

(가능하면) `edges_assist.csv`도 해석에 활용 가능.

### 4-9. `summary.json`

| 키 | 설명 |
|---|---|
| `stints_total` | 전체 스틴트 수 |
| `stints_use` | `use == True` 스틴트 수 |
| `stints_use_by_phase` | 국면별 `use` 스틴트 수 |
| `mcc_on_stints_use` | 맥컬러가 코트에 있던 `use` 스틴트 수 |
| `mcc_on_minutes_use`, `mcc_on_minutes_regular_use` | 위 스틴트의 총 시간(분), 그중 정규시즌 |
| `stints_unknown_nodes`, `unknown_player_ids` | `nodes.csv`에 없는 선수가 포함된 스틴트 수, 해당 선수 목록 |
| `assist_edges` | `edges_assist.csv` 행 수 |
| `games` | 포함된 경기 수 |

---

## 5. 데이터 관련 참고사항

1. **데이터 검증**: PBP 이벤트를 합산한 값이 42경기 최종 점수, 1,118개 선수-경기의 박스스코어(득점/야투/3점/자유투/리바운드/어시스트/스틸/블록/턴오버/파울)와 모두 일치. 출전 시간은 선수-경기의 약 96%가 공식 기록과 1초 이내
2. **노드 피처 기간**: `nodes.csv`는 정규시즌 전체 합계라서 스틴트가 속한 경기의 개인 기록도 일부 포함되어 있음 (득실은 미포함)
3. **외국인 선수 규정**: TPBL은 외국인 2명 동시 출전이 가능해서, 맥컬러 출전 시간의 약 99%에 다른 외국인 선수 1명이 함께 있음. 추후 KBL 썬더스 매핑 시 `is_imported`로 국내 선수 조합을 따로 봐야 할 듯..
4. **표본 크기**: 맥컬러와 의미 있는 공출전 시간이 있는 같은 팀 선수는 7명 정도 .. 나머지는 함께 뛴 시간이 너무 적음 .. 득실보다는 맥컬러 개인의 스탯을 중심으로 분석하는 게 더 안정적일 것 같습니다 ..!!
5. **시즌 분류**: 주 분석은 정규시즌, 플레이오프는 보조, 프리시즌은 제외 권장.