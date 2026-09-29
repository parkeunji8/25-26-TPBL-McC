from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── 설정 ───────────────────────────
RAW_API_DIR = Path("raw_api")
GAMES_CSV = Path("input_data/games.csv")
PLAYER_LOGS_CSV = Path("input_data/player_game_logs.csv")
OUT_DIR = Path("pbp_tables")

ASSIST_LOOKBACK_EVENTS = 6    # 어시스트와 슛 사이 최대 이벤트 간격 (앞/뒤 양방향)
ASSIST_LOOKBACK_SEC = 25      # 그리고 경기시계 차이 N초 이내

# FIBA 코트 (좌표 단위 mm, 28m × 15m). offense_basket 0 → 왼쪽 골대, 1 → 오른쪽 골대
COURT_X, COURT_Y = 28000, 15000
BASKET_X = {0: 1575, 1: COURT_X - 1575}
BASKET_Y = COURT_Y / 2
RIM_RADIUS_MM = 1250          # 제한구역 반원
KEY_HALF_WIDTH_MM = 2450      # 페인트존 폭 4.9m
KEY_LENGTH_MM = 5800          # 페인트존 길이 5.8m
CORNER3_DEPTH_MM = 2990       # 코너 3점(직선 구간) 베이스라인으로부터 깊이

SHOT_ACTION_GROUP = {
    "Layup": "layup", "DrivingLayup": "drive", "Dunk": "dunk", "AlleyOop": "dunk",
    "PutbackDunk": "putback", "PutbackTipIn": "putback",
    "JumpShot": "jumper", "PullUpJumpShot": "pullup", "StepBackJumpShot": "pullup",
    "FadeawayJumpShot": "post_fade", "TurnaroundJumpShot": "post_fade", "HookShot": "post_fade",
    "FloatingJumpShot": "floater",
}
MADE = {"Made", "AndOne"}
STAT_COLS = ["PTS", "FGA", "FGM", "2PA", "2PM", "3PA", "3PM", "FTA", "FTM",
             "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
             "FGM_assisted", "FGM_unassisted",
             "rim_FGA", "rim_FGM", "paint_FGA", "paint_FGM", "mid_FGA", "mid_FGM",
             "c3_FGA", "c3_FGM", "ab3_FGA", "ab3_FGM"] + \
            [f"act_{g}_FGA" for g in sorted(set(SHOT_ACTION_GROUP.values()))]


# ─────────────────────────── 유틸 ───────────────────────────
def period_len_ms(q: int) -> int:
    return 720_000 if q <= 4 else 300_000


def elapsed_sec(q: int, clock_ms: int) -> float:
    base = (q - 1) * 720 if q <= 4 else 2880 + (q - 5) * 300
    return base + (period_len_ms(q) - clock_ms) / 1000


def period_end_sec(q: int) -> float:
    return elapsed_sec(q, 0)


def shot_zone(event_type, x, y, basket):
    """반환: (거리 m, 존). 좌표 없으면 (nan, None)."""
    if x is None or y is None or basket not in BASKET_X:
        return np.nan, None
    bx = BASKET_X[basket]
    dist = math.hypot(x - bx, y - BASKET_Y)
    depth = x if basket == 0 else COURT_X - x          # 베이스라인으로부터 거리
    if event_type == "ThreePointer":
        zone = "c3" if depth <= CORNER3_DEPTH_MM else "ab3"
    elif dist <= RIM_RADIUS_MM:
        zone = "rim"
    elif abs(y - BASKET_Y) <= KEY_HALF_WIDTH_MM and depth <= KEY_LENGTH_MM:
        zone = "paint"
    else:
        zone = "mid"
    return dist / 1000, zone


# ─────────────────────────── 1. 이벤트 파싱 ───────────────────────────
def parse_events(gid: str, raw: dict) -> pd.DataFrame:
    xy = {s["order"]: s for s in raw.get("score_events", [])}
    rows = []
    for rnd in raw["rounds"]:
        for e in rnd["events"]:
            s = xy.get(e["order"], {})
            q = e.get("quarter") if e.get("quarter") is not None else rnd.get("round")
            rows.append({
                "game_id": gid, "order": e["order"], "quarter": q,
                "clock_ms": e.get("event_quarter_time"),
                "event_type": e["event_type"], "event_outcome": e["event_outcome"],
                "shoot_action": e.get("shoot_action"),
                "team_id": str(e["team"]["id"]) if e.get("team") else None,
                "player_id": str(e["player"]["id"]) if e.get("player") else None,
                "x_mm": s.get("event_point_X"), "y_mm": s.get("event_point_Y"),
                "offense_basket": s.get("offense_basket", e.get("offense_basket")),
            })
    ev = pd.DataFrame(rows)
    ev["player_id"] = ev.player_id.astype(object).where(ev.player_id.notna(), None)

    # 쿼터 값이 없는 이벤트는 쓸 수 없음 → 제외하고 경고
    no_q = ev.quarter.isna()
    if no_q.any():
        print(f"   [warn] {gid}: 쿼터 정보 없는 이벤트 {int(no_q.sum())}개 제외 "
              f"({ev.loc[no_q, 'event_type'].value_counts().to_dict()})")
        ev = ev[~no_q].copy()
    ev["quarter"] = ev.quarter.astype(int)

    # 시계 값이 없는 이벤트: 같은 쿼터에서 입력순서(order)상 직전 이벤트의 시계로 채움
    # (직전이 없으면 직후, 둘 다 없으면 쿼터 시작 시각)
    ev["clock_imputed"] = ev.clock_ms.isna()
    if ev.clock_imputed.any():
        ev = ev.sort_values(["quarter", "order"])
        ev["clock_ms"] = ev.groupby("quarter").clock_ms.transform(lambda c: c.ffill().bfill())
        start = ev.quarter.map(period_len_ms)
        ev["clock_ms"] = ev.clock_ms.fillna(start)
        print(f"   [warn] {gid}: 시계 값 없는 이벤트 {int(ev.clock_imputed.sum())}개 보정 "
              f"({ev.loc[ev.clock_imputed, 'event_type'].value_counts().to_dict()})")
    ev["clock_ms"] = ev.clock_ms.astype(int)
    ev["is_ot"] = ev.quarter >= 5
    ev["elapsed_sec"] = [elapsed_sec(q, c) for q, c in zip(ev.quarter, ev.clock_ms)]

    # 시간순: 쿼터 → 남은시간 내림차순 → 입력순서(order).
    # 단, 쿼터 종료(0:00) 시점의 교체(전원 Leaving)는 같은 시각의 다른 이벤트보다 뒤로
    ev["_end_rot"] = (ev.clock_ms == 0) & (ev.event_type == "Rotation")
    ev = ev.sort_values(["quarter", "clock_ms", "_end_rot", "order"],
                        ascending=[True, False, True, True]).drop(columns="_end_rot") \
           .reset_index(drop=True)
    ev.insert(1, "seq", range(len(ev)))

    fg = ev.event_type.isin(["TwoPointer", "ThreePointer"])
    ev["is_fga"] = fg
    ev["is_made"] = ev.event_outcome.isin(MADE) & (fg | (ev.event_type == "FreeThrow"))
    ev.loc[ev.event_type == "FreeThrow", "is_made"] = ev.event_outcome == "Made"
    ev["points"] = np.select(
        [fg & ev.is_made & (ev.event_type == "TwoPointer"),
         fg & ev.is_made & (ev.event_type == "ThreePointer"),
         (ev.event_type == "FreeThrow") & ev.is_made], [2, 3, 1], 0)

    zones = [shot_zone(t, x, y, b) if f else (np.nan, None)
             for t, x, y, b, f in zip(ev.event_type, ev.x_mm, ev.y_mm, ev.offense_basket, fg)]
    ev["shot_dist_m"] = [z[0] for z in zones]
    ev["shot_zone"] = [z[1] for z in zones]
    ev["shot_group"] = ev.shoot_action.map(SHOT_ACTION_GROUP)
    return ev


def link_assists(ev: pd.DataFrame) -> pd.DataFrame:
    """어시스트 → 가장 가까운 같은 팀 필드골 성공과 연결.
    기록원이 슛과 어시스트를 입력하는 순서가 경기마다 달라서(앞/뒤 모두 관찰됨) 양방향 탐색.
    후보 중 입력 시각(order) 차이가 가장 작은 슛을 선택하고, 슛 하나에는 어시스트 하나만."""
    ev["assist_by"] = None           # 슛 행: 어시스트한 선수
    ev["assisted_shot_seq"] = np.nan  # 어시스트 행: 연결된 슛/자유투의 seq
    ev["assist_to"] = None           # 어시스트 행: 받은 선수 (패스 네트워크용)
    ev["assist_kind"] = None         # 어시스트 행: 'FG' | 'FT'(슈팅파울 유도 후 자유투) | None
    made_fg = ev.index[ev.is_fga & ev.is_made]
    cands = []
    for i in ev.index[ev.event_type == "Assist"]:
        a = ev.loc[i]
        for j in made_fg:
            if abs(j - i) > ASSIST_LOOKBACK_EVENTS:
                continue
            c = ev.loc[j]
            if (c.team_id == a.team_id and c.player_id != a.player_id and c.quarter == a.quarter
                    and abs(a.elapsed_sec - c.elapsed_sec) <= ASSIST_LOOKBACK_SEC):
                cands.append((abs(int(a.order) - int(c.order)), i, j))
    used_a, used_s = set(), set()
    for _, i, j in sorted(cands):          # 가까운 쌍부터 확정 (1:1 매칭)
        if i in used_a or j in used_s:
            continue
        ev.at[j, "assist_by"] = ev.at[i, "player_id"]
        ev.at[i, "assisted_shot_seq"] = ev.at[j, "seq"]
        ev.at[i, "assist_to"] = ev.at[j, "player_id"]
        ev.at[i, "assist_kind"] = "FG"
        used_a.add(i); used_s.add(j)
    # 남은 어시스트: 이 리그는 슈팅파울을 유도한 패스에도 어시스트를 기록함 → 같은 시각 자유투와 연결
    fts = ev.index[ev.event_type == "FreeThrow"]
    for i in ev.index[(ev.event_type == "Assist")]:
        if i in used_a:
            continue
        a = ev.loc[i]
        near = [j for j in fts if abs(j - i) <= ASSIST_LOOKBACK_EVENTS
                and ev.at[j, "team_id"] == a.team_id and ev.at[j, "quarter"] == a.quarter
                and ev.at[j, "clock_ms"] == a.clock_ms and ev.at[j, "player_id"] != a.player_id]
        if near:
            j = min(near, key=lambda j: abs(j - i))
            ev.at[i, "assisted_shot_seq"] = ev.at[j, "seq"]
            ev.at[i, "assist_to"] = ev.at[j, "player_id"]
            ev.at[i, "assist_kind"] = "FT"
    return ev


# ─────────────────────────── 2. 라인업 구간 ───────────────────────────
def build_stints(ev: pd.DataFrame, home: str, away: str):
    """교체 이벤트로 코트 위 선수를 추적.
    - 구간의 '시간 경계'는 교체가 기록된 시각(데드볼 시계)
    - 이벤트의 '구간 귀속'은 기록 순서: 교체 묶음 이후 첫 비교체 이벤트부터 새 라인업
    반환: stints DataFrame, 이벤트별 stint_id Series."""
    stint_of = pd.Series(pd.NA, index=ev.index, dtype="object")
    stints = []
    gid = ev.game_id.iat[0]

    for q, qe in ev.groupby("quarter", sort=True):
        on = {home: set(), away: set()}
        cur = None
        dirty, sub_t = True, elapsed_sec(q, period_len_ms(q))

        for i, e in qe.iterrows():
            if e.event_type == "Rotation":
                if e.team_id in on:
                    (on[e.team_id].add if e.event_outcome == "Entering"
                     else on[e.team_id].discard)(e.player_id)
                dirty, sub_t = True, e.elapsed_sec
                continue
            if dirty:
                h, a = tuple(sorted(on[home])), tuple(sorted(on[away]))
                if cur is None or (h, a) != (cur["home_ids"], cur["away_ids"]):
                    start = elapsed_sec(q, period_len_ms(q)) if cur is None else sub_t
                    if cur is not None:
                        cur["end_sec"] = start
                        stints.append(cur)
                    cur = {"game_id": gid, "quarter": q, "start_sec": start,
                           "home_ids": h, "away_ids": a, "stint_no": len(stints) + 1}
                dirty = False
            stint_of.at[i] = f"{gid}_{cur['stint_no']:03d}"
        if cur is not None:
            cur["end_sec"] = period_end_sec(q)
            stints.append(cur)

    st = pd.DataFrame(stints)
    st["stint_id"] = st.game_id + "_" + st.stint_no.astype(int).map("{:03d}".format)
    st["duration_sec"] = st.end_sec - st.start_sec
    st["home_team_id"], st["away_team_id"] = home, away
    st["valid"] = (st.home_ids.str.len() == 5) & (st.away_ids.str.len() == 5)
    return st, stint_of


# ─────────────────────────── 3. 구간 집계 ───────────────────────────
def add_stat(c: dict, e) -> None:
    t, o = e.event_type, e.event_outcome
    if e.is_fga:
        two = t == "TwoPointer"
        c["FGA"] += 1; c["2PA" if two else "3PA"] += 1
        if e.is_made:
            c["FGM"] += 1; c["2PM" if two else "3PM"] += 1
            c["FGM_assisted" if e.assist_by else "FGM_unassisted"] += 1
        if e.shot_zone:
            c[f"{e.shot_zone}_FGA"] += 1; c[f"{e.shot_zone}_FGM"] += int(e.is_made)
        if isinstance(e.shot_group, str):
            c[f"act_{e.shot_group}_FGA"] += 1
    elif t == "FreeThrow":
        c["FTA"] += 1; c["FTM"] += int(e.is_made)
    elif t == "Rebound":
        c["OREB" if o == "Offensive" else "DREB"] += 1
    elif t == "Assist": c["AST"] += 1
    elif t == "Steal": c["STL"] += 1
    elif t == "Block": c["BLK"] += 1
    elif t == "Turnover": c["TOV"] += 1
    elif t == "RosterFoul": c["PF"] += 1
    c["PTS"] += int(e.points)


def aggregate(ev: pd.DataFrame, st: pd.DataFrame):
    # 팀 단위 (선수 없는 팀 리바운드·팀 턴오버 포함)
    team = defaultdict(lambda: defaultdict(float))
    player = defaultdict(lambda: defaultdict(float))
    off_lineup = 0
    lineup = {r.stint_id: (set(r.home_ids), set(r.away_ids)) for r in st.itertuples()}
    for e in ev[ev.stint_id.notna() & (ev.event_type != "Rotation")].itertuples():
        add_stat(team[(e.stint_id, e.team_id)], e)
        if isinstance(e.player_id, str):
            add_stat(player[(e.stint_id, e.player_id)], e)
            h, a = lineup[e.stint_id]
            off_lineup += int(e.player_id not in (h | a))

    def poss(c):  # 박스스코어 추정식과 동일
        return c["FGA"] - c["OREB"] + c["TOV"] + 0.44 * c["FTA"]

    for side in ("home", "away"):
        tid = st[f"{side}_team_id"]
        st[f"{side}_pts"] = [team[(s, t)]["PTS"] for s, t in zip(st.stint_id, tid)]
        st[f"{side}_poss_est"] = [poss(team[(s, t)]) for s, t in zip(st.stint_id, tid)]

    rows = []
    for r in st.itertuples():
        for side, ids, tid, opp in (("home", r.home_ids, r.home_team_id, r.away_team_id),
                                    ("away", r.away_ids, r.away_team_id, r.home_team_id)):
            for pid in ids:
                c = player[(r.stint_id, pid)]
                rows.append({"stint_id": r.stint_id, "game_id": r.game_id, "player_id": pid,
                             "team_id": tid, "opponent_id": opp, "side": side,
                             "duration_sec": r.duration_sec, "valid_stint": r.valid,
                             **{k: c.get(k, 0) for k in STAT_COLS}})
    sp = pd.DataFrame(rows)
    return st, sp, off_lineup


# ─────────────────────────── 4. 검증 ───────────────────────────
BOX_MAP = {"PTS": "points", "FGA": "FGA", "FGM": "FGM", "3PA": "three_PA", "3PM": "three_PM",
           "FTA": "FTA", "FTM": "FTM", "OREB": "OREB", "DREB": "DREB", "AST": "AST",
           "STL": "STL", "BLK": "BLK", "TOV": "TOV", "PF": "PF"}


def validate(gid, ev, st, games, pg):
    g = games.loc[gid]
    pts = ev.groupby("team_id").points.sum()
    # 이벤트 전체(구간 무관) 기준 선수 합계
    tot = defaultdict(lambda: defaultdict(float))
    for e in ev[ev.player_id.notna() & (ev.event_type != "Rotation")].itertuples():
        add_stat(tot[e.player_id], e)
    mine = pd.DataFrame(tot).T.reindex(columns=list(BOX_MAP)).fillna(0).add_prefix("ev_")
    box = pg[pg.game_id == gid].set_index("player_id")
    box = box[list(BOX_MAP.values()) + ["time_on_court_seconds"]].add_prefix("box_")
    j = mine.join(box, how="outer").fillna(0)
    # 구간 기준 출전시간
    sec = defaultdict(float)
    for r in st.itertuples():
        for pid in r.home_ids + r.away_ids:
            sec[pid] += r.duration_sec
    j["rebuilt_sec"] = pd.Series(sec)
    j["rebuilt_sec"] = j.rebuilt_sec.fillna(0)
    prow = []
    for pid, r in j.iterrows():
        d = {k: r[f"ev_{k}"] - r[f"box_{v}"] for k, v in BOX_MAP.items()}
        prow.append({"game_id": gid, "player_id": pid,
                     "stat_mismatches": ",".join(k for k, x in d.items() if x != 0),
                     "minutes_diff_sec": r.rebuilt_sec - r.box_time_on_court_seconds})
    n_ast = int((ev.event_type == "Assist").sum())
    grow = {
        "game_id": gid, "competition_phase": g.competition_phase,
        "home_pts_events": int(pts.get(str(g.home_team_id), 0)), "home_pts_official": int(g.home_points),
        "away_pts_events": int(pts.get(str(g.away_team_id), 0)), "away_pts_official": int(g.away_points),
        "n_events": len(ev), "n_stints": len(st), "valid_time_share":
            round(st.duration_sec[st.valid].sum() / max(st.duration_sec.sum(), 1), 4),
        "assists": n_ast, "assists_linked": int(ev.assisted_shot_seq.notna().sum()),
        "assists_to_ft": int((ev.assist_kind == "FT").sum()),
        "fga_missing_xy": int((ev.is_fga & ev.x_mm.isna()).sum()),
        "clock_imputed": int(ev.clock_imputed.sum()),
        "rotation_clock_imputed": int((ev.clock_imputed & (ev.event_type == "Rotation")).sum()),
    }
    grow["score_ok"] = (grow["home_pts_events"] == grow["home_pts_official"]
                        and grow["away_pts_events"] == grow["away_pts_official"])
    return grow, prow


# ─────────────────────────── 메인 ───────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(GAMES_CSV, dtype={"game_id": str, "home_team_id": str,
                                          "away_team_id": str}).set_index("game_id")
    pg = pd.read_csv(PLAYER_LOGS_CSV, dtype={"game_id": str, "player_id": str})

    files = sorted(RAW_API_DIR.glob("*/games_broadcasts.json"), key=lambda p: int(p.parent.name))
    all_ev, all_st, all_sp, vg, vp = [], [], [], [], []
    for f in files:
        gid = f.parent.name
        if gid not in games.index:
            print(f"[skip] {gid}: games.csv에 없음"); continue
        raw = json.loads(f.read_text(encoding="utf-8"))
        g = games.loc[gid]
        ev = link_assists(parse_events(gid, raw))
        st, stint_of = build_stints(ev, g.home_team_id, g.away_team_id)
        ev["stint_id"] = stint_of
        # 누적 점수 (이벤트 직후 기준). 원본 current_points는 '이벤트 직전' 값이라 쓰지 않음
        ev["home_score"] = ev.points.where(ev.team_id == g.home_team_id, 0).cumsum()
        ev["away_score"] = ev.points.where(ev.team_id == g.away_team_id, 0).cumsum()
        st, sp, off = aggregate(ev, st)
        st["competition_phase"] = g.competition_phase
        grow, prow = validate(gid, ev, st, games, pg)
        grow["events_off_lineup"] = off
        all_ev.append(ev); all_st.append(st); all_sp.append(sp); vg.append(grow); vp += prow
        flag = "OK" if grow["score_ok"] else "SCORE MISMATCH"
        print(f"{gid}: events={len(ev)} stints={len(st)} valid_time={grow['valid_time_share']} "
              f"assist_link={grow['assists_linked']}/{grow['assists']} [{flag}]")

    ev = pd.concat(all_ev, ignore_index=True)
    st = pd.concat(all_st, ignore_index=True)
    sp = pd.concat(all_sp, ignore_index=True)
    st["home_ids"] = st.home_ids.map("|".join)
    st["away_ids"] = st.away_ids.map("|".join)
    cols = ["stint_id", "game_id", "competition_phase", "quarter", "start_sec", "end_sec",
            "duration_sec", "home_team_id", "away_team_id", "home_ids", "away_ids", "valid",
            "home_pts", "away_pts", "home_poss_est", "away_poss_est"]
    ev.to_csv(OUT_DIR / "events.csv", index=False)
    st[cols].to_csv(OUT_DIR / "stints.csv", index=False)
    sp.to_csv(OUT_DIR / "stint_players.csv", index=False)
    vg = pd.DataFrame(vg); vp = pd.DataFrame(vp)
    vg.to_csv(OUT_DIR / "validation_games.csv", index=False)
    vp.to_csv(OUT_DIR / "validation_players.csv", index=False)

    report = {
        "games": len(vg), "score_ok_games": int(vg.score_ok.sum()),
        "player_games": len(vp),
        "player_games_all_stats_match": int((vp.stat_mismatches == "").sum()),
        "player_games_minutes_within_1s": int((vp.minutes_diff_sec.abs() <= 1).sum()),
        "assist_link_rate": round(vg.assists_linked.sum() / max(vg.assists.sum(), 1), 4),
        "valid_time_share": round(st.duration_sec[st.valid].sum() / st.duration_sec.sum(), 4),
        "events_off_lineup": int(vg.events_off_lineup.sum()),
        "events_clock_imputed": int(vg.clock_imputed.sum()),
        "games_with_rotation_clock_imputed": vg.loc[vg.rotation_clock_imputed > 0, "game_id"].tolist(),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()