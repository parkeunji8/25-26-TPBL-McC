from __future__ import annotations
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── 설정 ───────────────────────────
DATA_DIR = Path("input_data")
OUT_DIR = Path("output_data/gnn_data")
PHASES = ["regular_season"]          # 국면 필터. 전 국면이면 None
SEASON_START = pd.Timestamp("2025-10-01")  # 나이 계산 기준일
K_MIN = 200.0          # 분당 스탯 축소 강도(분). 출전 200분이면 개인:리그 = 1:1
K_ATT = {"FG": 100, "3P": 50, "FT": 30, "TS": 100}  # 성공률 축소 강도(시도 수)
K_SHARE = 50.0         # 슛 유형 비중 축소 강도(FGA)
MIN_EDGE_SEC = 60      # 공출전 60초 미만 동료 간선은 버림
HARD_NEG_MAX_SEC = 60  # 같은 팀인데 공출전 60초 이하면 hard negative
MIN_MINUTES_FLAG = 200 # 이보다 적게 뛴 선수는 low_minutes 플래그

# 프로필 수집 실패 3명 수동 보강 (결측 필드만 채움)
MANUAL_PROFILES = {
    "99":    dict(player_name_english="Lu Kuan-Liang",      nationality="Local",    national_identity="Local"),
    "10885": dict(player_name_english="Liu Cheng-Hsun",     nationality="Local",    national_identity="Local"),
    "14437": dict(player_name_english="Michael Frazier II", nationality="Imported", national_identity="Imported"),
}

ID_DTYPES = {"player_id": str, "game_id": str, "team_id": str,
             "home_team_id": str, "away_team_id": str, "opponent_id": str}


# ─────────────────────────── 1. 선수 메타 ───────────────────────────
def load_players() -> pd.DataFrame:
    p = pd.read_csv(DATA_DIR / "players.csv", dtype=ID_DTYPES)
    for pid, fields in MANUAL_PROFILES.items():
        mask = p.player_id == pid
        assert mask.sum() == 1, f"player {pid} not found"
        for col, val in fields.items():
            cur = p.loc[mask, col].iloc[0]
            if pd.isna(cur):
                p.loc[mask, col] = val
            elif cur != val:
                print(f"[warn] {pid}.{col}: 기존값 {cur!r} 유지 (수동값 {val!r})")
    missing = p[["nationality", "height_cm", "weight_kg", "birthday", "position"]].isna().sum()
    assert missing.sum() == 0, f"메타 결측 남음: {missing[missing > 0].to_dict()}"

    # 포지션 multi-hot: 'PointGuardAndShootingGuard' → PG, SG
    pos_map = {"PointGuard": "PG", "ShootingGuard": "SG", "SmallForward": "SF",
               "PowerForward": "PF", "Center": "C"}
    parts = p.position.str.split("And")
    for full, short in pos_map.items():
        p[f"pos_{short}"] = parts.apply(lambda xs: float(full in xs))
    assert (p[[f"pos_{s}" for s in pos_map.values()]].sum(axis=1) > 0).all(), "알 수 없는 포지션 표기"
    # 포지션 번호(1~5)의 평균: 가드-빅 스펙트럼을 연속값으로
    order = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    p["pos_num"] = parts.apply(lambda xs: np.mean([order[pos_map[x]] for x in xs]))

    p["is_imported"] = (p.national_identity == "Imported").astype(float)
    p["age"] = (SEASON_START - pd.to_datetime(p.birthday)).dt.days / 365.25
    p["bmi"] = p.weight_kg / (p.height_cm / 100) ** 2
    return p


# ─────────────────────────── 2. 스틴트 재구성 ───────────────────────────
def elapsed_sec(q: int, ms: int) -> float:
    """경기 경과 초. 정규 쿼터 12분, 연장(quarter>=5) 5분."""
    plen = 720 if q <= 4 else 300
    base = (q - 1) * 720 if q <= 4 else 2880 + (q - 5) * 300
    return base + plen - ms / 1000


def build_stints(games: pd.DataFrame) -> pd.DataFrame:
    se = pd.read_csv(DATA_DIR / "substitution_events.csv", dtype=ID_DTYPES)
    se["t"] = [elapsed_sec(q, ms) for q, ms in zip(se.quarter, se.clock_remaining_ms)]
    se["ord"] = se.action.map({"Leaving": 0, "Entering": 1})  # 같은 시각: 나감 → 들어옴
    se = se.sort_values(["game_id", "quarter", "t", "ord", "event_order"])
    gi = games.set_index("game_id")

    rows = []
    for gid, ge in se.groupby("game_id", sort=False):
        H, A = gi.at[gid, "home_team_id"], gi.at[gid, "away_team_id"]
        for q, qe in ge.groupby("quarter"):
            on = defaultdict(set)
            t_prev = None
            for t, batch in qe.groupby("t", sort=True):  # 같은 시각 교체는 한 묶음
                if t_prev is not None and t > t_prev:
                    rows.append((gid, q, t_prev, t, H, A,
                                 tuple(sorted(on[H])), tuple(sorted(on[A]))))
                for tid, pid, act in zip(batch.team_id, batch.player_id, batch.action):
                    (on[tid].discard if act == "Leaving" else on[tid].add)(pid)
                t_prev = t
    st = pd.DataFrame(rows, columns=["game_id", "quarter", "start", "end",
                                     "home_team_id", "away_team_id", "home", "away"])
    st["duration_sec"] = st.end - st.start
    st["valid"] = (st.home.str.len() == 5) & (st.away.str.len() == 5)
    st["competition_phase"] = st.game_id.map(gi.competition_phase)
    st.insert(0, "stint_id", st.game_id + "_" +
              st.groupby("game_id").cumcount().add(1).astype(str).str.zfill(3))
    return st


def validate_minutes(st: pd.DataFrame, pg: pd.DataFrame) -> dict:
    """재구성 출전시간 vs 박스스코어 time_on_court_seconds."""
    rec = defaultdict(float)
    for gid, h, a, d in zip(st.game_id, st.home, st.away, st.duration_sec):
        for pid in h + a:
            rec[(gid, pid)] += d
    rec = pd.Series(rec, name="rebuilt")
    rec.index.names = ["game_id", "player_id"]
    cmp = (pg.set_index(["game_id", "player_id"]).time_on_court_seconds
             .to_frame().join(rec, how="outer").fillna(0))
    diff = (cmp.rebuilt - cmp.time_on_court_seconds).abs()
    return {"player_games": int(len(cmp)), "within_1s": int((diff <= 1).sum()),
            "within_10s": int((diff <= 10).sum()), "max_abs_diff_sec": float(diff.max())}


# ─────────────────────────── 3. 간선 ───────────────────────────
def build_edges(st: pd.DataFrame, team_of: dict):
    tm_sec, tm_n, op_sec, op_n = (defaultdict(float) for _ in range(4))
    for h, a, d in zip(st.home, st.away, st.duration_sec):
        for side in (h, a):
            for i, j in combinations(side, 2):  # 이미 정렬됨 → i<j
                tm_sec[(i, j)] += d; tm_n[(i, j)] += 1
        for i, j in product(h, a):
            key = (i, j) if i < j else (j, i)
            op_sec[key] += d; op_n[key] += 1

    # 선수별 총 출전(초): 정규화 분모
    on_sec = defaultdict(float)
    for h, a, d in zip(st.home, st.away, st.duration_sec):
        for pid in h + a:
            on_sec[pid] += d

    def to_df(sec, n, kind):
        df = pd.DataFrame([(i, j, s, n[(i, j)]) for (i, j), s in sec.items()],
                          columns=["src", "dst", "shared_sec", "n_stints"])
        mi = df[["src", "dst"]].apply(lambda r: min(on_sec[r.src], on_sec[r.dst]), axis=1)
        ma = df[["src", "dst"]].apply(lambda r: max(on_sec[r.src], on_sec[r.dst]), axis=1)
        # overlap: 덜 뛴 선수 기준 몇 %를 함께/맞대고 뛰었나 (팀 크기·출전량 영향 완화)
        df["overlap"] = df.shared_sec / mi
        # jaccard: 두 선수 출전시간 합집합 대비 공유 비율
        union = df.src.map(on_sec) + df.dst.map(on_sec) - df.shared_sec
        df["jaccard"] = df.shared_sec / union
        df["log_shared_min"] = np.log1p(df.shared_sec / 60)
        df["edge_type"] = kind
        return df

    tm = to_df(tm_sec, tm_n, "teammate")
    tm = tm[tm.shared_sec >= MIN_EDGE_SEC].reset_index(drop=True)
    op = to_df(op_sec, op_n, "opponent")

    # hard negatives: 같은 팀, 둘 다 출전했는데 공출전이 거의 없는 쌍
    players = sorted(on_sec)
    by_team = defaultdict(list)
    for pid in players:
        by_team[team_of[pid]].append(pid)
    neg = []
    for t, ps in by_team.items():
        for i, j in combinations(sorted(ps), 2):
            if tm_sec.get((i, j), 0.0) <= HARD_NEG_MAX_SEC:
                neg.append((i, j, t, tm_sec.get((i, j), 0.0)))
    neg = pd.DataFrame(neg, columns=["src", "dst", "team_id", "shared_sec"])
    return tm, op, neg, on_sec


# ─────────────────────────── 4. 노드 피처 (축소 포함) ───────────────────────────
SHOT_GROUPS = {  # source_ 슛 유형 → 그룹 (시도 기준, FGA 합과 100% 일치 확인됨)
    "rim":  ["two_pointers_layup", "two_pointers_driving_layup", "two_pointers_dunk",
             "two_pointers_putback_dunk", "two_pointers_putback_tip_in", "two_pointers_alley_oop"],
    "mid_catch": ["two_pointers_jump_shot"],
    "mid_create": ["two_pointers_pull_up_jump_shot", "two_pointers_step_back_jump_shot",
                   "two_pointers_fadeaway_jump_shot", "two_pointers_turnaround_jump_shot"],
    "float_hook": ["two_pointers_floating_jump_shot", "two_pointers_hook_shot"],
    "three_catch": ["three_pointers_jump_shot"],
    "three_create": ["three_pointers_pull_up_jump_shot", "three_pointers_step_back_jump_shot",
                     "three_pointers_fadeaway_jump_shot", "three_pointers_turnaround_jump_shot",
                     "three_pointers_floating_jump_shot", "three_pointers_hook_shot"],
}
COUNT_STATS = ["points", "FGA", "three_PA", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
               "source_points_in_paint", "source_fast_break_points", "source_second_chance_points"]


def shrink(num, den, prior, k):
    """경험적 베이즈식 축소: (num + k*prior) / (den + k). den=0이면 prior."""
    return (num + k * prior) / (den + k)


def build_node_features(pg: pd.DataFrame, node_ids: list[str]) -> pd.DataFrame:
    g = pg[pg.appearance == True].copy()
    if PHASES:
        g = g[g.competition_phase.isin(PHASES)].copy()
    for grp, cols in SHOT_GROUPS.items():
        g = g.assign(**{f"att_{grp}": g[[f"source_{c}_attempted" for c in cols]].sum(axis=1)})
    agg_cols = COUNT_STATS + ["minutes", "FGM", "three_PM", "FTM",
                              "source_field_goals_attempted_in_the_paint",
                              "source_field_goals_attempted_on_fast_break"] + \
               [f"att_{k}" for k in SHOT_GROUPS]
    tot = g.groupby("player_id")[agg_cols].sum()
    tot["games"] = g.groupby("player_id").game_id.nunique()
    tot = tot.reindex(node_ids).fillna(0.0)  # 해당 국면 출전 0이면 전부 리그 prior로

    L = tot.sum()  # 리그 합계 (노드 전체 = 출전 선수 전체)
    f = pd.DataFrame(index=tot.index)
    f["minutes"] = tot.minutes
    f["log_minutes"] = np.log1p(tot.minutes)
    f["min_per_game"] = np.where(tot.games > 0, tot.minutes / tot.games.clip(lower=1), 0.0)

    # per-36 (분 기준 축소)
    for c in COUNT_STATS:
        f[f"{c.replace('source_', '')}_p36"] = 36 * shrink(tot[c], tot.minutes, L[c] / L.minutes, K_MIN)

    # 성공률 (시도 기준 축소)
    f["FG_pct"] = shrink(tot.FGM, tot.FGA, L.FGM / L.FGA, K_ATT["FG"])
    f["three_P_pct"] = shrink(tot.three_PM, tot.three_PA, L.three_PM / L.three_PA, K_ATT["3P"])
    f["FT_pct"] = shrink(tot.FTM, tot.FTA, L.FTM / L.FTA, K_ATT["FT"])
    tsa = 2 * (tot.FGA + 0.44 * tot.FTA)
    f["TS_pct"] = shrink(tot.points, tsa, L.points / (2 * (L.FGA + 0.44 * L.FTA)), K_ATT["TS"] * 2)

    # 스타일 비중 (FGA 기준 축소)
    shares = {"three_PA_rate": "three_PA", "FT_rate": "FTA",
              "paint_FGA_share": "source_field_goals_attempted_in_the_paint",
              "fastbreak_FGA_share": "source_field_goals_attempted_on_fast_break",
              **{f"shot_{k}_share": f"att_{k}" for k in SHOT_GROUPS}}
    for name, c in shares.items():
        f[name] = shrink(tot[c], tot.FGA, L[c] / L.FGA, K_SHARE)

    # 역할 비율 (분모가 작을 때 폭주 방지: 분당 축소값끼리 나눔)
    f["AST_TOV"] = f.AST_p36 / f.TOV_p36
    f["OREB_share"] = f.OREB_p36 / (f.OREB_p36 + f.DREB_p36)
    f["low_minutes"] = (tot.minutes < MIN_MINUTES_FLAG).astype(float)
    return f


# ─────────────────────────── 5. 메인 ───────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    players = load_players()
    games = pd.read_csv(DATA_DIR / "games.csv", dtype=ID_DTYPES)
    pg = pd.read_csv(DATA_DIR / "player_game_logs.csv", dtype=ID_DTYPES)

    st_all = build_stints(games)
    report = {"minutes_check_all_stints": validate_minutes(st_all, pg)}
    st = st_all[st_all.valid]
    if PHASES:
        st = st[st.competition_phase.isin(PHASES)]
    report["stints"] = {"rebuilt": int(len(st_all)), "valid": int(st_all.valid.sum()),
                        "used": int(len(st)), "used_min": round(st.duration_sec.sum() / 60, 1),
                        "games_used": int(st.game_id.nunique())}

    # 선수 → 팀 (스틴트에서 가장 많이 뛴 팀; 이적 대비)
    tsec = defaultdict(float)
    for H, A, h, a, d in zip(st.home_team_id, st.away_team_id, st.home, st.away, st.duration_sec):
        for pid in h: tsec[(pid, H)] += d
        for pid in a: tsec[(pid, A)] += d
    tdf = pd.Series(tsec).rename_axis(["player_id", "team_id"]).reset_index(name="sec")
    team_of = tdf.sort_values("sec").groupby("player_id").team_id.last().to_dict()
    report["multi_team_players"] = int((tdf.groupby("player_id").size() > 1).sum())

    tm, op, neg, on_sec = build_edges(st, team_of)

    # 노드 인덱스: 스틴트에 등장한 선수 전체
    node_ids = sorted(on_sec)
    idx = {pid: i for i, pid in enumerate(node_ids)}
    feats = build_node_features(pg, node_ids)

    meta = players.set_index("player_id").loc[node_ids]
    teams = pd.read_csv(DATA_DIR / "teams.csv", dtype=ID_DTYPES).set_index("team_id")
    bio_cols = ["height_cm", "weight_kg", "bmi", "age", "is_imported", "pos_num",
                "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"]
    nodes = pd.concat([meta[["player_name_original", "player_name_english", "position",
                             "national_identity"]],
                       meta[bio_cols].astype(float), feats], axis=1)
    nodes.insert(0, "node_idx", range(len(nodes)))
    nodes["team_id"] = [team_of[p] for p in node_ids]
    nodes["team_name_english"] = nodes.team_id.map(teams.team_name_english)
    nodes["stint_on_court_min"] = [on_sec[p] / 60 for p in node_ids]
    nodes.index.name = "player_id"

    # 모델 입력 피처: 연속형은 z-score, 이진/원핫은 그대로
    binary = ["is_imported", "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C", "low_minutes"]
    continuous = [c for c in bio_cols + list(feats.columns) if c not in binary]
    for c in continuous:
        sd = nodes[c].std(ddof=0)
        nodes[f"z_{c}"] = (nodes[c] - nodes[c].mean()) / (sd if sd > 0 else 1.0)
    model_cols = [f"z_{c}" for c in continuous] + binary
    assert not nodes[model_cols].isna().any().any(), "피처에 NaN"

    # 간선에 노드 인덱스 부여
    for df in (tm, op, neg):
        df["src_idx"] = df.src.map(idx); df["dst_idx"] = df.dst.map(idx)
    tm["same_team"] = True
    op["same_team"] = False

    # 저장
    nodes.reset_index().to_csv(OUT_DIR / "nodes.csv", index=False)
    tm.to_csv(OUT_DIR / "edges_teammate.csv", index=False)
    op.to_csv(OUT_DIR / "edges_opponent.csv", index=False)
    neg.to_csv(OUT_DIR / "hard_negatives.csv", index=False)
    st.assign(home=st.home.str.join("|"), away=st.away.str.join("|")) \
      .to_csv(OUT_DIR / "stints_used.csv", index=False)
    np.savez(OUT_DIR / "graph.npz",
             x=nodes[model_cols].to_numpy(np.float32),
             player_id=np.array(node_ids),
             team_idx=pd.factorize(nodes.team_id)[0],  # 팀 누수 진단용 (모델 입력 아님)
             tm_edge_index=tm[["src_idx", "dst_idx"]].to_numpy().T,
             tm_edge_attr=tm[["overlap", "jaccard", "log_shared_min"]].to_numpy(np.float32),
             op_edge_index=op[["src_idx", "dst_idx"]].to_numpy().T,
             op_edge_attr=op[["overlap", "jaccard", "log_shared_min"]].to_numpy(np.float32),
             hard_neg_index=neg[["src_idx", "dst_idx"]].to_numpy().T)
    json.dump({"model_feature_columns": model_cols, "continuous": continuous, "binary": binary,
               "edge_attr_columns": ["overlap", "jaccard", "log_shared_min"],
               "config": {"PHASES": PHASES, "K_MIN": K_MIN, "K_ATT": K_ATT, "K_SHARE": K_SHARE,
                          "MIN_EDGE_SEC": MIN_EDGE_SEC, "HARD_NEG_MAX_SEC": HARD_NEG_MAX_SEC}},
              open(OUT_DIR / "feature_columns.json", "w"), indent=2, ensure_ascii=False)

    report.update({"nodes": len(nodes), "features": len(model_cols),
                   "teammate_edges": int(len(tm)), "opponent_edges": int(len(op)),
                   "hard_negatives": int(len(neg)),
                   "low_minutes_nodes": int(nodes.low_minutes.sum())})
    json.dump(report, open(OUT_DIR / "report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
