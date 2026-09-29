from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── 설정 ───────────────────────────
GNN_DATA_DIR = Path("output_data/gnn_data")
PBP_DIR = Path("pbp_tables")
GAMES_CSV = Path("input_data/games.csv")
TEAM_LOGS_CSV = Path("input_data/team_game_logs.csv")
OUT_DIR = Path("output_data/gnn_dataset")

LEO_TEAM_ID = "5"
MCC_ID = "10860"
N_FOLDS = 5                   # 경기 단위 교차검증 fold (같은 경기 스틴트는 같은 fold)
GARBAGE_MARGIN = 20           # 4쿼터 이후 점수 차가 이 이상이면 가비지 타임 플래그
STINT_STAT_COLS = ["PTS", "FGA", "FGM", "2PA", "2PM", "3PA", "3PM", "FTA", "FTM",
                   "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
                   "FGM_assisted", "FGM_unassisted",
                   "rim_FGA", "rim_FGM", "paint_FGA", "paint_FGM", "mid_FGA", "mid_FGM",
                   "c3_FGA", "c3_FGM", "ab3_FGA", "ab3_FGM"]
ID = {"game_id": str, "player_id": str, "team_id": str, "home_team_id": str,
      "away_team_id": str, "opponent_id": str, "stint_id": str}


# ─────────────────────────── 1. 리그 그래프 (prep_gnn 결과 재사용) ───────────────────────────
def copy_league_graph() -> pd.DataFrame:
    for f in ["nodes.csv", "edges_teammate.csv", "edges_opponent.csv",
              "hard_negatives.csv", "feature_columns.json"]:
        shutil.copy2(GNN_DATA_DIR / f, OUT_DIR / f)
    return pd.read_csv(GNN_DATA_DIR / "nodes.csv", dtype={"player_id": str})


# ─────────────────────────── 2. 상대 전력 (누수 없는 사전값) ───────────────────────────
def opponent_strength(games: pd.DataFrame) -> pd.DataFrame:
    """각 경기 시점에서 팀의 '이전 경기들' 평균 ORtg/DRtg/Net (해당 경기 제외, 전 국면)."""
    t = pd.read_csv(TEAM_LOGS_CSV, dtype=ID)
    t = t.merge(games[["game_id", "game_datetime_local"]], on="game_id")
    t = t.sort_values(["team_id", "game_datetime_local", "game_id"])
    g = t.groupby("team_id")
    for c in ["ORtg", "DRtg", "Net_Rating"]:
        t[f"prior_{c}"] = g[c].transform(lambda s: s.shift().expanding().mean())
        t[f"prior_{c}"] = t[f"prior_{c}"].fillna(t[c].mean() if c != "Net_Rating" else 0.0)
    t["prior_games"] = g.cumcount()
    return t[["game_id", "team_id", "prior_ORtg", "prior_DRtg", "prior_Net_Rating", "prior_games"]]


# ─────────────────────────── 3. 스틴트 그래프 ───────────────────────────
def build_stint_graphs(nodes: pd.DataFrame, games: pd.DataFrame):
    st = pd.read_csv(PBP_DIR / "stints.csv", dtype=ID | {"home_ids": str, "away_ids": str})
    # 한 팀 선수가 0명인 무효 스틴트는 라인업 칸이 비어 NaN으로 읽힘 → 빈 문자열로
    st["home_ids"] = st.home_ids.fillna("")
    st["away_ids"] = st.away_ids.fillna("")
    sp = pd.read_csv(PBP_DIR / "stint_players.csv", dtype=ID)
    idx = dict(zip(nodes.player_id, nodes.node_idx))

    st = st.sort_values(["game_id", "start_sec"]).reset_index(drop=True)
    leo_home = st.home_team_id == LEO_TEAM_ID
    assert (leo_home | (st.away_team_id == LEO_TEAM_ID)).all(), "Leopards 경기가 아닌 스틴트 포함"

    # Leopards 관점으로 정렬
    g = pd.DataFrame({
        "stint_id": st.stint_id, "game_id": st.game_id,
        "competition_phase": st.competition_phase, "quarter": st.quarter.astype(int),
        "start_sec": st.start_sec, "end_sec": st.end_sec, "duration_sec": st.duration_sec,
        "valid": st.valid.astype(str).str.lower() == "true",
        "leo_is_home": leo_home,
        "opp_team_id": np.where(leo_home, st.away_team_id, st.home_team_id),
        "leo_ids": np.where(leo_home, st.home_ids, st.away_ids),
        "opp_ids": np.where(leo_home, st.away_ids, st.home_ids),
        "leo_pts": np.where(leo_home, st.home_pts, st.away_pts).astype(float),
        "opp_pts": np.where(leo_home, st.away_pts, st.home_pts).astype(float),
        "leo_poss": np.where(leo_home, st.home_poss_est, st.away_poss_est).astype(float),
        "opp_poss": np.where(leo_home, st.away_poss_est, st.home_poss_est).astype(float),
    })

    # 맥락 피처: 스틴트 시작 시점 점수 차, 남은 시간, 가비지 타임
    grp = g.groupby("game_id")
    g["margin_at_start"] = (grp.leo_pts.cumsum() - g.leo_pts) - (grp.opp_pts.cumsum() - g.opp_pts)
    reg_end = 2880.0
    g["sec_remaining_regulation"] = (reg_end - g.start_sec).clip(lower=0)
    g["is_ot"] = g.quarter >= 5
    g["garbage_time"] = (g.quarter >= 4) & (g.margin_at_start.abs() >= GARBAGE_MARGIN)

    # 상대 전력 (이전 경기 기준)
    osn = opponent_strength(games).rename(columns={"team_id": "opp_team_id"})
    osn.columns = ["game_id", "opp_team_id"] + [f"opp_{c}" for c in osn.columns[2:]]
    g = g.merge(osn, on=["game_id", "opp_team_id"], how="left")

    # 타깃: 팀 득실
    g["margin"] = g.leo_pts - g.opp_pts
    g["poss_avg"] = (g.leo_poss + g.opp_poss) / 2
    g["net_per100"] = np.where(g.poss_avg > 0, 100 * g.margin / g.poss_avg.where(g.poss_avg > 0), np.nan)
    g["leo_ortg"] = np.where(g.leo_poss > 0, 100 * g.leo_pts / g.leo_poss.where(g.leo_poss > 0), np.nan)
    g["leo_drtg"] = np.where(g.opp_poss > 0, 100 * g.opp_pts / g.opp_poss.where(g.opp_poss > 0), np.nan)

    # 타깃: 맥컬러 개인 (코트에 있을 때만 값, 아니면 NaN)
    g["mcc_on"] = g.leo_ids.map(lambda s: MCC_ID in s.split("|"))
    m = sp[sp.player_id == MCC_ID].set_index("stint_id")[STINT_STAT_COLS].add_prefix("mcc_")
    g = g.join(m, on="stint_id")
    g["mcc_TSA"] = g.mcc_FGA + 0.44 * g.mcc_FTA                    # TS 분모/2
    g["mcc_usage_poss"] = g.mcc_FGA + 0.44 * g.mcc_FTA + g.mcc_TOV    # 맥컬러가 끝낸 포제션
    g["mcc_usage_share"] = np.where(g.leo_poss > 0, g.mcc_usage_poss / g.leo_poss.where(g.leo_poss > 0), np.nan)

    # 노드 인덱스 매핑 확인
    def to_idx(s):
        return [idx.get(p, -1) for p in s.split("|")] if s else [-1]
    g["leo_node_idx"] = g.leo_ids.map(to_idx)
    g["opp_node_idx"] = g.opp_ids.map(to_idx)
    unknown = g.leo_node_idx.map(lambda l: -1 in l) | g.opp_node_idx.map(lambda l: -1 in l)
    g["all_nodes_known"] = ~unknown

    # 학습 사용 여부: 5v5, 길이 > 0, 노드 모두 존재
    g["use"] = g.valid & (g.duration_sec > 0) & g.all_nodes_known

    # 경기 단위 fold (날짜순으로 경기를 돌려가며 배정 → 시즌 흐름이 fold마다 고르게)
    order = games.set_index("game_id").loc[g.game_id.unique()].sort_values("game_datetime_local").index
    fold = {gid: i % N_FOLDS for i, gid in enumerate(order)}
    g["fold"] = g.game_id.map(fold)

    # 스틴트 × 선수 (10명)
    sn = sp.merge(g[["stint_id", "leo_is_home"]], on="stint_id")
    sn["side"] = np.where(sn.team_id == LEO_TEAM_ID, "leo", "opp")
    sn["node_idx"] = sn.player_id.map(idx).fillna(-1).astype(int)
    sn["is_mcc"] = sn.player_id == MCC_ID
    sn = sn[["stint_id", "game_id", "player_id", "node_idx", "team_id", "side", "is_mcc",
             "duration_sec"] + STINT_STAT_COLS]
    return g, sn


# ─────────────────────────── 4. 어시스트(패스) 간선 ───────────────────────────
def build_assist_edges(nodes: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_csv(PBP_DIR / "events.csv", dtype=ID | {"assist_to": str})
    shots = ev.set_index(["game_id", "seq"])
    a = ev[(ev.event_type == "Assist") & ev.assist_to.notna()].copy()
    a["seq_target"] = a.assisted_shot_seq.astype(float).astype(int)
    a["pts_created"] = [shots.at[(gid, s), "points"] for gid, s in zip(a.game_id, a.seq_target)]
    e = (a.groupby(["player_id", "assist_to", "team_id"])
          .agg(assists=("seq", "size"),
               assists_fg=("assist_kind", lambda k: (k == "FG").sum()),
               assists_ft=("assist_kind", lambda k: (k == "FT").sum()),
               pts_created=("pts_created", "sum"))
          .reset_index().rename(columns={"player_id": "src", "assist_to": "dst"}))

    # 정규화 분모: 두 선수가 같이 뛴 시간 (같은 42경기 스틴트 기준)
    shared = {}
    for ids, d, use in zip(g.leo_ids.tolist() + g.opp_ids.tolist(),
                           g.duration_sec.tolist() * 2, g.use.tolist() * 2):
        if not use:
            continue
        ps = [p for p in ids.split("|") if p]
        for i in ps:
            for j in ps:
                if i != j:
                    shared[(i, j)] = shared.get((i, j), 0.0) + d
    e["shared_min"] = [shared.get((s, d), 0.0) / 60 for s, d in zip(e.src, e.dst)]
    e["assists_per36_shared"] = np.where(e.shared_min > 0, 36 * e.assists / e.shared_min.where(e.shared_min > 0), np.nan)
    idx = dict(zip(nodes.player_id, nodes.node_idx))
    e["src_idx"] = e.src.map(idx).fillna(-1).astype(int)
    e["dst_idx"] = e.dst.map(idx).fillna(-1).astype(int)
    e["coverage"] = np.where(e.team_id == LEO_TEAM_ID, "leopards_all_games", "vs_leopards_only")
    return e


# ─────────────────────────── 5. README ───────────────────────────
README = """# TPBL 2025-26 맥컬러 라인업 GNN 데이터셋

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
  fold(경기 단위 0~{n_folds_m1}, 같은 경기는 같은 fold)

## 사용 시 주의
1. 스틴트 대부분이 1분 미만이라 타깃 노이즈가 매우 큼 → 손실에 poss_avg(또는 duration_sec) 가중치 권장.
2. 분할은 반드시 fold(경기 단위)로. 스틴트 무작위 분할은 누수.
3. 국면: 주 분석은 competition_phase=='regular_season'. preseason은 제외 권장, playoffs는 보조.
4. 노드 피처는 정규시즌 시즌 합계 기반 → 같은 경기의 개인 생산량이 일부 포함됨(득실은 미포함).
   엄격한 예측 평가가 필요하면 경기 이전 누적으로 재계산 필요.
5. TPBL은 외국인 2명 동시 출전 가능 → 맥컬러 스틴트 대부분에 다른 외국인 1명이 함께 있음.
   KBL 매핑 시 국내 선수 조합 해석을 분리할 것 (nodes.is_imported 활용).
6. garbage_time==True 구간은 성과 해석에서 제외하거나 따로 볼 것.
"""


# ─────────────────────────── 메인 ───────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(GAMES_CSV, dtype=ID)
    nodes = copy_league_graph()

    g, sn = build_stint_graphs(nodes, games)
    ea = build_assist_edges(nodes, g)

    g_out = g.copy()
    g_out["leo_node_idx"] = g_out.leo_node_idx.map(lambda l: "|".join(map(str, l)))
    g_out["opp_node_idx"] = g_out.opp_node_idx.map(lambda l: "|".join(map(str, l)))
    g_out.to_csv(OUT_DIR / "stint_graphs.csv", index=False)
    sn.to_csv(OUT_DIR / "stint_nodes.csv", index=False)
    ea.to_csv(OUT_DIR / "edges_assist.csv", index=False)

    u = g[g.use].reset_index(drop=True)
    leo = np.array(u.leo_node_idx.tolist(), dtype=np.int64)
    opp = np.array(u.opp_node_idx.tolist(), dtype=np.int64)
    mcc_node = int(nodes.loc[nodes.player_id == MCC_ID, "node_idx"].iloc[0])
    mcc_pos = np.array([row.index(mcc_node) if mcc_node in row else -1 for row in u.leo_node_idx])
    ctx_cols = ["leo_is_home", "margin_at_start", "sec_remaining_regulation", "is_ot", "garbage_time",
                "opp_prior_ORtg", "opp_prior_DRtg", "opp_prior_Net_Rating"]
    y_team = ["margin", "net_per100", "leo_pts", "opp_pts", "leo_poss", "opp_poss", "poss_avg"]
    y_mcc = [c for c in u.columns if c.startswith("mcc_") and c != "mcc_on"]
    np.savez(OUT_DIR / "stint_graphs.npz",
             stint_id=u.stint_id.to_numpy(), game_id=u.game_id.to_numpy(),
             phase=u.competition_phase.to_numpy(), fold=u.fold.to_numpy(),
             leo_idx=leo, opp_idx=opp, mcc_pos=mcc_pos,
             duration_sec=u.duration_sec.to_numpy(np.float32),
             ctx=u[ctx_cols].astype(float).to_numpy(np.float32), ctx_cols=np.array(ctx_cols),
             y_team=u[y_team].to_numpy(np.float32), y_team_cols=np.array(y_team),
             y_mcc=u[y_mcc].to_numpy(np.float32), y_mcc_cols=np.array(y_mcc))
    (OUT_DIR / "README.md").write_text(README.replace("{n_folds_m1}", str(N_FOLDS - 1)),
                                       encoding="utf-8")

    summ = {
        "stints_total": len(g), "stints_use": int(g.use.sum()),
        "stints_use_by_phase": g[g.use].competition_phase.value_counts().to_dict(),
        "mcc_on_stints_use": int((g.use & g.mcc_on).sum()),
        "mcc_on_minutes_use": round(g.loc[g.use & g.mcc_on, "duration_sec"].sum() / 60, 1),
        "mcc_on_minutes_regular_use": round(g.loc[g.use & g.mcc_on & (g.competition_phase == "regular_season"),
                                                  "duration_sec"].sum() / 60, 1),
        "stints_unknown_nodes": int((~g.all_nodes_known).sum()),
        "unknown_player_ids": sorted(
            {p for col in ("leo_ids", "opp_ids") for ids in g.loc[~g.all_nodes_known, col]
             for p in ids.split("|") if p} - set(nodes.player_id)),
        "assist_edges": len(ea), "games": int(g.game_id.nunique()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()