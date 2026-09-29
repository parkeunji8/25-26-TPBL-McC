# TPBL 2025-26 맥컬러 라인업 GNN 데이터셋

모든 스틴트 테이블은 **Leopards(team_id 5) 관점**. 스틴트 = 양 팀 10명이 바뀌지 않은 구간.

## 파일
| 파일 | 단위 | 설명 |
|---|---|---|
| nodes.csv | 선수 | 리그 정규시즌 출전 선수. 모델 입력 피처는 feature_columns.json의 model_feature_columns (z_* 표준화 + 이진) |
| edges_teammate.csv / edges_opponent.csv | 선수 쌍(무방향) | 리그 전체 공출전/매치업. overlap, jaccard, log_shared_min |
| hard_negatives.csv | 선수 쌍 | 같은 팀인데 공출전 ≤ 60초 (링크 예측 네거티브) |
| edges_assist.csv | 패서→득점자(방향) | 42경기 PBP. coverage=leopards_all_games(Leopards 선수, 전 경기) / vs_leopards_only(상대 선수, Leopards전만) |
| stint_graphs.csv | 스틴트 | 맥락 피처 + 타깃 |
| stint_nodes.csv | 스틴트×선수(10) | node_idx, side(leo/opp), 스틴트 내 개인 스탯 |
| stint_graphs.npz | 배열 | stint_graphs에서 use==True 행만. leo_idx/opp_idx (S×5), mcc_pos (S, 없으면 -1), 타깃·맥락 배열 |

## stint_graphs.csv 주요 컬럼
- 식별: stint_id, game_id, competition_phase, quarter, start_sec, end_sec, duration_sec
- 라인업: leo_ids / opp_ids (player_id '|' 구분), leo_node_idx / opp_node_idx
- 맥락(입력 가능): leo_is_home, margin_at_start, sec_remaining_regulation, is_ot, garbage_time,
  opp_prior_ORtg / opp_prior_DRtg / opp_prior_Net_Rating (해당 경기 이전 경기들 평균 → 누수 없음), opp_prior_games
- 팀 타깃: leo_pts, opp_pts, margin, leo_poss, opp_poss(추정: FGA−OREB+TOV+0.44FTA), poss_avg,
  net_per100, leo_ortg, leo_drtg (포제션 0이면 NaN)
- 맥컬러 타깃(mcc_on==True일 때만): mcc_PTS, mcc_FGA, … 존별 mcc_rim/paint/mid/c3/ab3_FGA·FGM,
  mcc_FGM_assisted / mcc_FGM_unassisted, mcc_TSA(=FGA+0.44FTA), mcc_usage_poss, mcc_usage_share
- 필터/분할: valid(5v5), all_nodes_known, **use**(학습 사용 권장 = valid & duration>0 & 노드 존재),
  fold(경기 단위 0~4, 같은 경기는 같은 fold)

## 사용 시 주의
1. 스틴트 대부분이 1분 미만이라 타깃 노이즈가 매우 큼 → 손실에 poss_avg(또는 duration_sec) 가중치 권장.
2. 분할은 반드시 fold(경기 단위)로. 스틴트 무작위 분할은 누수.
3. 국면: 주 분석은 competition_phase=='regular_season'. preseason은 제외 권장, playoffs는 보조.
4. 노드 피처는 정규시즌 시즌 합계 기반 → 같은 경기의 개인 생산량이 일부 포함됨(득실은 미포함).
   엄격한 예측 평가가 필요하면 경기 이전 누적으로 재계산 필요.
5. TPBL은 외국인 2명 동시 출전 가능 → 맥컬러 스틴트 대부분에 다른 외국인 1명이 함께 있음.
   KBL 매핑 시 국내 선수 조합 해석을 분리할 것 (nodes.is_imported 활용).
6. garbage_time==True 구간은 성과 해석에서 제외하거나 따로 볼 것.
