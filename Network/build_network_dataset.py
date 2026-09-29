from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── 설정 ───────────────────────────
INPUT_DIR = Path("input_data")
OUT_DIR = Path("output_data")
PHASE = "regular_season"
LEO_TEAM_ID = "5"
MCC_ID = "10860"
SEASON_START = pd.Timestamp("2025-10-01")

MIN_COPLAY_SEC = 60        # ① 공출전 간선 최소 시간
K_POSS = 50                # ③④⑤ 득실 축소: 100 × 득실 / (포제션 + K_POSS) → 표본 적으면 0 쪽으로
MANUAL_PROFILES = {        # 프로필 수집 실패 3명 수동 보강
    "99":    dict(player_name_english="Lu Kuan-Liang",      nationality="Local",    national_identity="Local"),
    "10885": dict(player_name_english="Liu Cheng-Hsun",     nationality="Local",    national_identity="Local"),
    "14437": dict(player_name_english="Michael Frazier II", nationality="Imported", national_identity="Imported"),
}
ID = {"game_id": str, "player_id": str, "team_id": str, "home_team_id": str, "away_team_id": str,
      "opponent_id": str, "stint_id": str, "assist_to": str, "assist_by": str}

# 개인 스탯 집계용 컬럼 (stint_players.csv)
SP_STATS = ["PTS", "FGA", "FGM", "3PA", "3PM", "FTA", "FTM", "OREB", "DREB", "AST", "STL",
            "BLK", "TOV", "PF", "FGM_assisted", "FGM_unassisted",
            "rim_FGA", "paint_FGA", "mid_FGA", "c3_FGA", "ab3_FGA"]


def safe_div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


# ─────────────────────────── 공통: 데이터 로드 ───────────────────────────
def load_inputs():
    d = {}
    d["games"] = pd.read_csv(INPUT_DIR / "games.csv", dtype=ID)
    d["players"] = pd.read_csv(INPUT_DIR / "players.csv", dtype=ID)
    d["teams"] = pd.read_csv(INPUT_DIR / "teams.csv", dtype=ID)
    d["pg"] = pd.read_csv(INPUT_DIR / "player_game_logs.csv", dtype=ID)
    d["subs"] = pd.read_csv(INPUT_DIR / "substitution_events.csv", dtype=ID)
    d["events"] = pd.read_csv(INPUT_DIR / "events.csv", dtype=ID, low_memory=False)
    st = pd.read_csv(INPUT_DIR / "stints.csv", dtype=ID | {"home_ids": str, "away_ids": str})
    st["home_ids"] = st.home_ids.fillna("")
    st["away_ids"] = st.away_ids.fillna("")
    d["stints"] = st
    d["sp"] = pd.read_csv(INPUT_DIR / "stint_players.csv", dtype=ID)
    return d


# ─────────────────────────── 리그 스틴트 (교체 기록 기반, 공출전용) ───────────────────────────
def elapsed_sec(q: int, ms: float) -> float:
    plen = 720 if q <= 4 else 300
    base = (q - 1) * 720 if q <= 4 else 2880 + (q - 5) * 300
    return base + plen - ms / 1000


def league_stints(subs: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    gi = games.set_index("game_id")
    reg = set(games.game_id[games.competition_phase == PHASE])
    se = subs[subs.game_id.isin(reg)].copy()
    se["t"] = [elapsed_sec(q, ms) for q, ms in zip(se.quarter, se.clock_remaining_ms)]
    se["ord"] = se.action.map({"Leaving": 0, "Entering": 1})
    se = se.sort_values(["game_id", "quarter", "t", "ord", "event_order"])
    rows = []
    for gid, ge in se.groupby("game_id", sort=False):
        H, A = gi.at[gid, "home_team_id"], gi.at[gid, "away_team_id"]
        for _, qe in ge.groupby("quarter"):
            on = defaultdict(set)
            t_prev = None
            for t, b in qe.groupby("t", sort=True):
                if t_prev is not None and t > t_prev:
                    rows.append((gid, H, A, tuple(sorted(on[H])), tuple(sorted(on[A])), t - t_prev))
                for tid, pid, act in zip(b.team_id, b.player_id, b.action):
                    (on[tid].discard if act == "Leaving" else on[tid].add)(pid)
                t_prev = t
    st = pd.DataFrame(rows, columns=["game_id", "home_team_id", "away_team_id", "home", "away", "sec"])
    return st[(st.home.str.len() == 5) & (st.away.str.len() == 5)].reset_index(drop=True)


# ─────────────────────────── 레오파즈 스틴트 (PBP 기반, 득실 포함) ───────────────────────────
def leo_stints(stints: pd.DataFrame) -> pd.DataFrame:
    s = stints[(stints.competition_phase == PHASE)
               & (stints.valid.astype(str).str.lower() == "true")
               & (stints.duration_sec > 0)].copy()
    home = s.home_team_id == LEO_TEAM_ID
    out = pd.DataFrame({
        "stint_id": s.stint_id, "game_id": s.game_id, "sec": s.duration_sec,
        "opp_team_id": np.where(home, s.away_team_id, s.home_team_id),
        "leo": np.where(home, s.home_ids, s.away_ids),
        "opp": np.where(home, s.away_ids, s.home_ids),
        "leo_pts": np.where(home, s.home_pts, s.away_pts).astype(float),
        "opp_pts": np.where(home, s.away_pts, s.home_pts).astype(float),
        "leo_poss": np.where(home, s.home_poss_est, s.away_poss_est).astype(float),
        "opp_poss": np.where(home, s.away_poss_est, s.home_poss_est).astype(float),
    })
    out["leo"] = out.leo.str.split("|").map(tuple)
    out["opp"] = out.opp.str.split("|").map(tuple)
    out["poss"] = (out.leo_poss + out.opp_poss) / 2
    return out.reset_index(drop=True)


def perf_cols(sec, leo_pts, opp_pts, leo_poss, opp_poss, prefix=""):
    """시간·득실·포제션 합계 → 성과 지표 dict."""
    poss = (leo_poss + opp_poss) / 2
    margin = leo_pts - opp_pts
    return {
        f"{prefix}minutes": sec / 60, f"{prefix}poss": poss,
        f"{prefix}pts_for": leo_pts, f"{prefix}pts_against": opp_pts, f"{prefix}margin": margin,
        f"{prefix}ortg": float(safe_div(100 * leo_pts, leo_poss)),
        f"{prefix}drtg": float(safe_div(100 * opp_pts, opp_poss)),
        f"{prefix}net_per100": float(safe_div(100 * margin, poss)),
        f"{prefix}net_per100_shrunk": 100 * margin / (poss + K_POSS),
    }


# ─────────────────────────── 노드 ───────────────────────────
def build_nodes(d, lst: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    p = d["players"].copy()
    for pid, fields in MANUAL_PROFILES.items():
        m = p.player_id == pid
        for c, v in fields.items():
            p.loc[m & p[c].isna(), c] = v

    # 정규시즌 출전 선수 = 리그 스틴트에 등장한 선수
    sec = defaultdict(float); team_sec = defaultdict(float)
    for H, A, h, a, s in zip(lst.home_team_id, lst.away_team_id, lst.home, lst.away, lst.sec):
        for pid in h: sec[pid] += s; team_sec[(pid, H)] += s
        for pid in a: sec[pid] += s; team_sec[(pid, A)] += s
    ids = sorted(sec)
    ts = pd.Series(team_sec).rename_axis(["player_id", "team_id"]).reset_index(name="s")
    team_of = ts.sort_values("s").groupby("player_id").team_id.last()

    n = p.set_index("player_id").loc[ids, ["player_name_original", "player_name_english", "position",
                                            "national_identity", "height_cm", "weight_kg", "birthday"]].copy()
    n["team_id"] = team_of.reindex(ids).values
    n["team_name_english"] = n.team_id.map(d["teams"].set_index("team_id").team_name_english)
    n["is_leopards"] = n.team_id == LEO_TEAM_ID
    n["is_mcc"] = n.index == MCC_ID
    n["is_imported"] = n.national_identity == "Imported"
    n["age"] = ((SEASON_START - pd.to_datetime(n.birthday)).dt.days / 365.25).round(1)
    n = n.drop(columns="birthday")

    # 정규시즌 박스스코어 (원값, 해석용)
    g = d["pg"][(d["pg"].competition_phase == PHASE) & (d["pg"].appearance == True)]
    tot = g.groupby("player_id")[["minutes", "points", "FGA", "FGM", "three_PA", "three_PM", "FTA", "FTM",
                                   "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
                                   "source_field_goals_attempted_in_the_paint"]].sum()
    tot["games"] = g.groupby("player_id").game_id.nunique()
    tot = tot.reindex(ids).fillna(0)
    n["games"] = tot.games.astype(int)
    n["minutes"] = tot.minutes.round(1)
    n["min_per_game"] = safe_div(tot.minutes, tot.games).round(1)
    for c, name in [("points", "PTS"), ("OREB", "OREB"), ("DREB", "DREB"), ("AST", "AST"), ("STL", "STL"),
                    ("BLK", "BLK"), ("TOV", "TOV"), ("FGA", "FGA"), ("three_PA", "3PA"), ("FTA", "FTA")]:
        n[f"{name}_per36"] = (36 * safe_div(tot[c], tot.minutes)).round(2)
    n["REB_per36"] = n.OREB_per36 + n.DREB_per36
    n["FG_pct"] = safe_div(tot.FGM, tot.FGA).round(3)
    n["3P_pct"] = safe_div(tot.three_PM, tot.three_PA).round(3)
    n["FT_pct"] = safe_div(tot.FTM, tot.FTA).round(3)
    n["TS_pct"] = safe_div(tot.points, 2 * (tot.FGA + 0.44 * tot.FTA)).round(3)
    n["3PA_rate"] = safe_div(tot.three_PA, tot.FGA).round(3)
    n["FT_rate"] = safe_div(tot.FTA, tot.FGA).round(3)
    n["paint_FGA_share"] = safe_div(tot.source_field_goals_attempted_in_the_paint, tot.FGA).round(3)
    n["AST_TOV"] = safe_div(tot.AST, tot.TOV).round(2)

    # 레오파즈 선수: 온/오프 코트 팀 득실 (PBP 스틴트)
    tot_l = ls[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].sum()
    rows = {}
    for pid in n.index[n.is_leopards]:
        on = ls[ls.leo.map(lambda t: pid in t)]
        s_on = on[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].sum()
        s_off = tot_l - s_on
        r = perf_cols(*s_on.values, prefix="on_")
        r.update(perf_cols(*s_off.values, prefix="off_"))
        r["on_off_net_diff"] = r["on_net_per100"] - r["off_net_per100"]
        rows[pid] = r
    oo = pd.DataFrame(rows).T
    keep = ["on_minutes", "on_poss", "on_net_per100", "on_net_per100_shrunk",
            "off_minutes", "off_poss", "off_net_per100", "on_off_net_diff"]
    n = n.join(oo[keep].astype(float).round(2))
    n["on_court_min_league_stints"] = [round(sec[i] / 60, 1) for i in ids]
    n.index.name = "player_id"
    return n.reset_index()


# ─────────────────────────── ① 공출전 ───────────────────────────
def build_coplay(lst: pd.DataFrame) -> pd.DataFrame:
    shared, nst, on = defaultdict(float), defaultdict(int), defaultdict(float)
    team = {}
    for H, A, h, a, s in zip(lst.home_team_id, lst.away_team_id, lst.home, lst.away, lst.sec):
        for side, tid in ((h, H), (a, A)):
            for pid in side:
                on[pid] += s
            for i, j in combinations(side, 2):
                shared[(i, j)] += s; nst[(i, j)] += 1; team[(i, j)] = tid
    e = pd.DataFrame([(i, j, team[(i, j)], s, nst[(i, j)]) for (i, j), s in shared.items()],
                     columns=["source", "target", "team_id", "shared_sec", "n_stints"])
    e = e[e.shared_sec >= MIN_COPLAY_SEC].copy()
    a, b = e.source.map(on), e.target.map(on)
    e["shared_min"] = (e.shared_sec / 60).round(2)
    e["overlap"] = (e.shared_sec / np.minimum(a, b)).round(4)
    e["jaccard"] = (e.shared_sec / (a + b - e.shared_sec)).round(4)
    e["weight"] = e.shared_min
    return e.drop(columns="shared_sec").sort_values("shared_min", ascending=False).reset_index(drop=True)


# ─────────────────────────── ② 어시스트 ───────────────────────────
def build_assist(ev: pd.DataFrame, games: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    reg = set(games.game_id[games.competition_phase == PHASE])
    ev = ev[ev.game_id.isin(reg)]
    shot = ev.set_index(["game_id", "seq"]).points
    a = ev[(ev.event_type == "Assist") & ev.assist_to.notna()].copy()
    a["pts"] = [shot.get((g, int(float(s))), 0) for g, s in zip(a.game_id, a.assisted_shot_seq)]
    e = (a.groupby(["player_id", "assist_to", "team_id"])
          .agg(assists=("seq", "size"),
               assists_fg=("assist_kind", lambda k: int((k == "FG").sum())),
               assists_ft=("assist_kind", lambda k: int((k == "FT").sum())),
               pts_created=("pts", "sum"))
          .reset_index().rename(columns={"player_id": "source", "assist_to": "target"}))
    # 정규화: 두 선수가 같이 뛴 시간 (같은 경기들의 PBP 스틴트, 양 팀 모두)
    shared = defaultdict(float)
    for side_col in ("leo", "opp"):
        for side, s in zip(ls[side_col], ls.sec):
            for i, j in product(side, side):
                if i != j:
                    shared[(i, j)] += s
    e["shared_min"] = [round(shared.get((s, t), 0) / 60, 2) for s, t in zip(e.source, e.target)]
    e["assists_per36_shared"] = (36 * safe_div(e.assists, e.shared_min)).round(2)
    e["coverage"] = np.where(e.team_id == LEO_TEAM_ID, "leopards_all_games", "vs_leopards_only")
    e["weight"] = e.assists
    return e.sort_values("assists", ascending=False).reset_index(drop=True)


# ─────────────────────────── ③ 레오파즈 2인 조합 성과 ───────────────────────────
def build_pair_performance(ls: pd.DataFrame) -> pd.DataFrame:
    acc = defaultdict(lambda: np.zeros(6))
    for leo, vals in zip(ls.leo, ls[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].to_numpy()):
        for i, j in combinations(sorted(leo), 2):
            acc[(i, j)] += np.r_[vals, 1]
    rows = []
    for (i, j), v in acc.items():
        r = {"source": i, "target": j, "n_stints": int(v[5])}
        r.update(perf_cols(*v[:5]))
        rows.append(r)
    e = pd.DataFrame(rows)
    e["weight"] = e.net_per100_shrunk
    return e.round(3).sort_values("minutes", ascending=False).reset_index(drop=True)


# ─────────────────────────── ④ 맥컬러 ego ───────────────────────────
def indiv_metrics(t: pd.Series, team_poss: float, sec: float, prefix: str) -> dict:
    """개인 스탯 합계 → 해석용 지표."""
    m = sec / 60
    fga = t.FGA
    return {
        f"{prefix}minutes": m,
        f"{prefix}PTS_per36": float(safe_div(36 * t.PTS, m)),
        f"{prefix}FGA_per36": float(safe_div(36 * fga, m)),
        f"{prefix}REB_per36": float(safe_div(36 * (t.OREB + t.DREB), m)),
        f"{prefix}AST_per36": float(safe_div(36 * t.AST, m)),
        f"{prefix}TOV_per36": float(safe_div(36 * t.TOV, m)),
        f"{prefix}TS_pct": float(safe_div(t.PTS, 2 * (fga + 0.44 * t.FTA))),
        f"{prefix}3PA_rate": float(safe_div(t["3PA"], fga)),
        f"{prefix}FT_rate": float(safe_div(t.FTA, fga)),
        f"{prefix}rim_share": float(safe_div(t.rim_FGA, fga)),
        f"{prefix}paint_share": float(safe_div(t.paint_FGA, fga)),
        f"{prefix}mid_share": float(safe_div(t.mid_FGA, fga)),
        f"{prefix}three_share": float(safe_div(t.c3_FGA + t.ab3_FGA, fga)),
        f"{prefix}assisted_share": float(safe_div(t.FGM_assisted, t.FGM)),
        f"{prefix}usage_share": float(safe_div(fga + 0.44 * t.FTA + t.TOV, team_poss)),
        f"{prefix}FGA": fga,
    }


def build_mcc_ego(ls: pd.DataFrame, sp: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    sp = sp[sp.stint_id.isin(ls.stint_id)]
    st = ls.set_index("stint_id")
    mcc_stints = st[st.leo.map(lambda t: MCC_ID in t)]
    mcc_rows = sp[sp.player_id == MCC_ID].set_index("stint_id")[SP_STATS]

    teammates = sorted({p for t in mcc_stints.leo for p in t} - {MCC_ID})
    rows = []
    for tm in teammates:
        with_ = mcc_stints.leo.map(lambda t: tm in t)
        r = {"source": MCC_ID, "target": tm}
        for label, sel in (("with", with_), ("without", ~with_)):
            ss = mcc_stints[sel]
            team = ss[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].sum()
            r.update(perf_cols(*team.values, prefix=f"team_{label}_"))
            r.update(indiv_metrics(mcc_rows.reindex(ss.index).fillna(0).sum(),
                                   team.leo_poss, team.sec, prefix=f"mcc_{label}_"))
        # 동료 입장: 맥컬러와 함께 / 맥컬러 없이
        tm_st = st[st.leo.map(lambda t: tm in t)]
        tm_rows = sp[sp.player_id == tm].set_index("stint_id")[SP_STATS]
        for label, sel in (("with_mcc", tm_st.leo.map(lambda t: MCC_ID in t)),
                           ("without_mcc", ~tm_st.leo.map(lambda t: MCC_ID in t))):
            ss = tm_st[sel]
            team = ss[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].sum()
            r.update(indiv_metrics(tm_rows.reindex(ss.index).fillna(0).sum(),
                                   team.leo_poss, team.sec, prefix=f"tm_{label}_"))
        rows.append(r)
    e = pd.DataFrame(rows)

    # 차이 (함께 − 따로)
    e["team_net_diff"] = e.team_with_net_per100 - e.team_without_net_per100
    for m in ["PTS_per36", "TS_pct", "3PA_rate", "rim_share", "paint_share", "mid_share",
              "three_share", "assisted_share", "usage_share", "AST_per36", "REB_per36", "TOV_per36"]:
        e[f"mcc_diff_{m}"] = e[f"mcc_with_{m}"] - e[f"mcc_without_{m}"]
        e[f"tm_diff_{m}"] = e[f"tm_with_mcc_{m}"] - e[f"tm_without_mcc_{m}"]
    nm = nodes.set_index("player_id")
    e.insert(2, "target_name", e.target.map(nm.player_name_english))
    e.insert(3, "target_position", e.target.map(nm.position))
    e.insert(4, "target_is_imported", e.target.map(nm.is_imported))
    e["shared_minutes"] = e.team_with_minutes
    e["weight"] = e.team_with_minutes
    return e.round(4).sort_values("shared_minutes", ascending=False).reset_index(drop=True)


# ─────────────────────────── ⑤ 매치업 ───────────────────────────
def build_matchup(ls: pd.DataFrame) -> pd.DataFrame:
    acc = defaultdict(lambda: np.zeros(6))
    opp_team = {}
    for leo, opp, ot, vals in zip(ls.leo, ls.opp, ls.opp_team_id,
                                  ls[["sec", "leo_pts", "opp_pts", "leo_poss", "opp_poss"]].to_numpy()):
        for i, j in product(leo, opp):
            acc[(i, j)] += np.r_[vals, 1]; opp_team[j] = ot
    rows = []
    for (i, j), v in acc.items():
        r = {"source": i, "target": j, "target_team_id": opp_team[j], "n_stints": int(v[5])}
        r.update(perf_cols(*v[:5]))
        rows.append(r)
    e = pd.DataFrame(rows)
    e["weight"] = e.minutes
    return e.round(3).sort_values("minutes", ascending=False).reset_index(drop=True)


# ─────────────────────────── GraphML ───────────────────────────
def write_graphml(nodes: pd.DataFrame, edges: dict[str, tuple[pd.DataFrame, bool]]):
    try:
        import networkx as nx
    except ImportError:
        print("[info] networkx 미설치 → GraphML 생략 (pip install networkx)")
        return []
    out = OUT_DIR / "graphml"; out.mkdir(exist_ok=True)
    attrs = nodes.set_index("player_id")
    attrs = attrs.astype(object).where(attrs.notna(), None)
    written = []
    for name, (e, directed) in edges.items():
        G = nx.DiGraph() if directed else nx.Graph()
        used = set(e.source) | set(e.target)
        for pid in used:
            a = attrs.loc[pid].to_dict() if pid in attrs.index else {}
            G.add_node(pid, **{k: v for k, v in a.items() if v is not None})
        cols = [c for c in e.columns if c not in ("source", "target")]
        for rec in e[["source", "target"] + cols].itertuples(index=False):
            d = {c: v for c, v in zip(cols, rec[2:]) if pd.notna(v)}
            G.add_edge(rec[0], rec[1], **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                                             float(v) if isinstance(v, (int, float, np.number)) else str(v))
                                          for k, v in d.items()})
        for _, a in G.nodes(data=True):
            for k, v in list(a.items()):
                a[k] = bool(v) if isinstance(v, (bool, np.bool_)) else \
                       float(v) if isinstance(v, (int, float, np.number)) else str(v)
        nx.write_graphml(G, out / f"{name}.graphml")
        written.append(f"graphml/{name}.graphml")
    return written


# ─────────────────────────── 메인 ───────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_inputs()

    lst = league_stints(d["subs"], d["games"])
    ls = leo_stints(d["stints"])
    nodes = build_nodes(d, lst, ls)

    coplay = build_coplay(lst)
    assist = build_assist(d["events"], d["games"], ls)
    pair = build_pair_performance(ls)
    ego = build_mcc_ego(ls, d["sp"], nodes)
    matchup = build_matchup(ls)

    nodes.to_csv(OUT_DIR / "nodes.csv", index=False)
    coplay.to_csv(OUT_DIR / "edges_coplay.csv", index=False)
    assist.to_csv(OUT_DIR / "edges_assist.csv", index=False)
    pair.to_csv(OUT_DIR / "edges_pair_performance.csv", index=False)
    ego.to_csv(OUT_DIR / "edges_mcc_ego.csv", index=False)
    matchup.to_csv(OUT_DIR / "edges_matchup.csv", index=False)

    gml = write_graphml(nodes, {
        "coplay": (coplay, False), "assist": (assist, True), "pair_performance": (pair, False),
        "mcc_ego": (ego, False), "matchup": (matchup, False)})

    # 검증: 레오파즈 PBP 스틴트 득실 합 = 해당 경기 최종 점수차 합
    g = d["games"].set_index("game_id")
    leo_games = ls.game_id.unique()
    final = sum((g.at[x, "home_points"] - g.at[x, "away_points"]) * (1 if g.at[x, "home_team_id"] == LEO_TEAM_ID else -1)
                for x in leo_games)
    report = {
        "phase": PHASE,
        "league_stints": len(lst), "league_games": int(lst.game_id.nunique()),
        "leopards_stints": len(ls), "leopards_games": int(len(leo_games)),
        "leopards_margin_sum_stints": float(ls.leo_pts.sum() - ls.opp_pts.sum()),
        "leopards_margin_sum_official": float(final),
        "note_margin": "두 값 차이 = 무효(5대5 아님) 스틴트에서 난 득실",
        "nodes": len(nodes), "leopards_nodes": int(nodes.is_leopards.sum()),
        "edges": {"coplay": len(coplay), "assist": len(assist), "pair_performance": len(pair),
                  "mcc_ego": len(ego), "matchup": len(matchup)},
        "mcc_minutes": round(float(ls[ls.leo.map(lambda t: MCC_ID in t)].sec.sum() / 60), 1),
        "graphml": gml,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(INPUT_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    INPUT_DIR, OUT_DIR = Path(args.input_dir), Path(args.out_dir)
    main()