"""
출력  : 2_build_feature_output/
  - player_features_all.csv      : 전 선수(국내 포함) 시즌 피처 (원값 + 보정값 + 해석용 컬럼)
  - cluster_input_imports.csv    : 외국선수 & 최소 출전시간 통과자, 군집 입력 피처만
  - feature_dictionary.csv       : 피처 정의 / 그룹 / KBL 이식 가능 여부
  - feature_correlation.csv      : 군집 입력 피처 간 상관행렬 (중복 피처 점검용)

"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "2_build_feature_input"
OUT_DIR = BASE_DIR / "2_build_feature_output"

EXCLUDE_PHASES = ["preseason"]   # 프리시즌 제외
MIN_MINUTES = 200                # 군집 대상 최소 출전시간(분)

NATIONALITY_OVERRIDE: dict[int, str] = {}

# 관심 선수 (같이 뛴 외국인 3인)
FOCUS = {10860: "Chris McCullough", 16: "Malcolm Miller",
         68: "Lasannah Kromah", 10861: "Cheick Diallo"}

# 축소 보정 강도: 리그 평균을 "가상의 분모 k만큼" 더해 준다.
# shrunk = (num + k * league_rate) / (den + k)
# 분모 단위가 피처마다 다르므로 k도 그 단위 기준
SHRINK_K = {
    "usg_pct":         150,  # 분모: 코트 위 팀 플레이 수
    "ast_pct":         100,  # 분모: 코트 위 동료 FGM
    "tov_pct":          60,  # 분모: 본인 플레이 수
    "orb_pct":          60,  # 분모: 코트 위 공격 리바운드 기회
    "drb_pct":          60,  # 분모: 코트 위 수비 리바운드 기회
    "stl_pct":         150,  # 분모: 코트 위 상대 포제션
    "blk_pct":         100,  # 분모: 코트 위 상대 2점 시도
    "three_par":        60,  # 분모: FGA
    "ft_rate":          60,  # 분모: FGA
    "paint_pts_share":  80,  # 분모: 득점
    "fb_pts_share":     80,
    "sc_pts_share":     80,
    "pf_per36":         80,  # 분모: 출전 분 (per-minute rate → 36분 환산)
}
SHOT_SHARE_K = 60            # 슛 타입 비중(분모 FGA) 공통 k

def load_data():
    pg = pd.read_csv(DATA_DIR / "player_game_logs.csv")
    players = pd.read_csv(DATA_DIR / "players.csv")
    tg = pd.read_csv(DATA_DIR / "team_game_logs.csv")

    pg = pg[(pg["appearance"] == True) & (~pg["competition_phase"].isin(EXCLUDE_PHASES))].copy()
    pg = pg[pg["minutes"].fillna(0) > 0]
    tg = tg[~tg["competition_phase"].isin(EXCLUDE_PHASES)].copy()

    num_cols = pg.select_dtypes("number").columns
    pg[num_cols] = pg[num_cols].fillna(0)

    for pid, nat in NATIONALITY_OVERRIDE.items():
        players.loc[players["player_id"] == pid, "nationality"] = nat
    return pg, players, tg


def attach_team_context(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    team_cols = ["FGM", "FGA", "three_PA", "FTA", "TOV", "OREB", "DREB", "possessions_est"]
    team = tg[["game_id", "team_id"] + team_cols].rename(columns={c: f"tm_{c}" for c in team_cols})
    opp = tg[["game_id", "team_id"] + team_cols].rename(
        columns={"team_id": "opponent_id", **{c: f"opp_{c}" for c in team_cols}})

    tm_min = (pg.groupby(["game_id", "team_id"])["minutes"].sum()
                .rename("tm_minutes").reset_index())

    df = (pg.merge(team, on=["game_id", "team_id"], how="left")
            .merge(opp, on=["game_id", "opponent_id"], how="left")
            .merge(tm_min, on=["game_id", "team_id"], how="left")).copy()

    missing = df["tm_FGA"].isna().sum() + df["opp_FGA"].isna().sum()
    if missing:
        print(f"[경고] 팀/상대 합계가 조인되지 않은 행: {missing}")


    df["court_share"] = df["minutes"] / (df["tm_minutes"] / 5)
    return df

SHOT_TYPES = {
    "self_jumper": ["two_pointers_pull_up_jump_shot", "three_pointers_pull_up_jump_shot",
                    "two_pointers_step_back_jump_shot", "three_pointers_step_back_jump_shot",
                    "two_pointers_fadeaway_jump_shot", "three_pointers_fadeaway_jump_shot",
                    "two_pointers_turnaround_jump_shot", "three_pointers_turnaround_jump_shot"],
    "catch_jumper": ["two_pointers_jump_shot", "three_pointers_jump_shot"],
    "drive":        ["two_pointers_driving_layup"],
    "layup":        ["two_pointers_layup"],
    "rim_finish":   ["two_pointers_dunk", "two_pointers_alley_oop"],
    "putback":      ["two_pointers_putback_dunk", "two_pointers_putback_tip_in"],
    "floater":      ["two_pointers_floating_jump_shot", "three_pointers_floating_jump_shot"],
    "post_hook":    ["two_pointers_hook_shot", "three_pointers_hook_shot"],
}


def aggregate_season(df: pd.DataFrame) -> pd.DataFrame:
    s = df["court_share"]
    d = pd.DataFrame({
        "player_id": df["player_id"],
        "games": 1,
        "minutes": df["minutes"],
        "pts": df["points"], "fgm": df["FGM"], "fga": df["FGA"],
        "tpa": df["three_PA"], "fta": df["FTA"], "ftm": df["FTM"],
        "oreb": df["OREB"], "dreb": df["DREB"], "ast": df["AST"],
        "stl": df["STL"], "blk": df["BLK"], "tov": df["TOV"], "pf": df["PF"],
        "paint_pts": df["source_points_in_paint"],
        "fb_pts": df["source_fast_break_points"],
        "sc_pts": df["source_second_chance_points"],
        # 분모 (코트 위에 있던 동안의 팀/상대 기회)
        "den_usg": s * (df["tm_FGA"] + 0.44 * df["tm_FTA"] + df["tm_TOV"]),
        "den_ast": s * df["tm_FGM"] - df["FGM"],
        "den_orb": s * (df["tm_OREB"] + df["opp_DREB"]),
        "den_drb": s * (df["tm_DREB"] + df["opp_OREB"]),
        "den_stl": s * df["opp_possessions_est"],
        "den_blk": s * (df["opp_FGA"] - df["opp_three_PA"]),
    })
    for grp, cols in SHOT_TYPES.items():
        d[f"shot_{grp}"] = df[[f"source_{c}_attempted" for c in cols]].sum(axis=1)

    agg = d.groupby("player_id").sum()
    agg["team_ids"] = df.groupby("player_id")["team_id"].agg(lambda x: "|".join(map(str, sorted(set(x)))))
    return agg

def feature_specs(agg: pd.DataFrame):
    plays = agg["fga"] + 0.44 * agg["fta"] + agg["tov"]
    specs = {
        "usg_pct":         (plays,            agg["den_usg"], 1),
        "ast_pct":         (agg["ast"],       agg["den_ast"], 1),
        "tov_pct":         (agg["tov"],       plays,          1),
        "orb_pct":         (agg["oreb"],      agg["den_orb"], 1),
        "drb_pct":         (agg["dreb"],      agg["den_drb"], 1),
        "stl_pct":         (agg["stl"],       agg["den_stl"], 1),
        "blk_pct":         (agg["blk"],       agg["den_blk"], 1),
        "three_par":       (agg["tpa"],       agg["fga"],     1),
        "ft_rate":         (agg["fta"],       agg["fga"],     1),
        "paint_pts_share": (agg["paint_pts"], agg["pts"],     1),
        "fb_pts_share":    (agg["fb_pts"],    agg["pts"],     1),
        "sc_pts_share":    (agg["sc_pts"],    agg["pts"],     1),
        "pf_per36":        (agg["pf"],        agg["minutes"], 36),
    }
    for grp in SHOT_TYPES:
        specs[f"shot_{grp}_share"] = (agg[f"shot_{grp}"], agg["fga"], 1)
    return specs


def compute_features(agg: pd.DataFrame, pool_mask: pd.Series) -> pd.DataFrame:
    """pool_mask: 리그 평균(사전값)을 계산할 선수 집합 (외국선수 전체)"""
    out = pd.DataFrame(index=agg.index)
    for name, (num, den, mult) in feature_specs(agg).items():
        den = den.clip(lower=0)
        raw = np.where(den > 0, num / den, np.nan)
        prior = num[pool_mask].sum() / den[pool_mask].sum()
        k = SHRINK_K.get(name, SHOT_SHARE_K)
        out[f"{name}_raw"] = raw * mult
        out[name] = (num + k * prior) / (den + k) * mult
    return out


def interpretation_cols(agg: pd.DataFrame) -> pd.DataFrame:
    """군집 입력에는 넣지 않는 해석용 지표"""
    out = pd.DataFrame(index=agg.index)
    out["games"] = agg["games"]
    out["minutes"] = agg["minutes"]
    out["mpg"] = agg["minutes"] / agg["games"]
    out["pts_per36"] = agg["pts"] / agg["minutes"] * 36
    out["reb_per36"] = (agg["oreb"] + agg["dreb"]) / agg["minutes"] * 36
    out["ast_per36"] = agg["ast"] / agg["minutes"] * 36
    out["ts_pct"] = agg["pts"] / (2 * (agg["fga"] + 0.44 * agg["fta"]))
    out["fga"] = agg["fga"]
    out["team_ids"] = agg["team_ids"]
    return out


CLUSTER_FEATURES = {
    # 피처: (그룹, 설명, KBL 이식)
    "usg_pct":         ("ball",     "코트 위 팀 플레이 중 본인이 마무리한 비율", "가능"),
    "ast_pct":         ("ball",     "코트 위 동료 야투 성공 중 본인 어시스트 비율", "가능"),
    "tov_pct":         ("ball",     "본인 플레이 중 턴오버 비율", "가능"),
    "three_par":       ("shooting", "FGA 중 3점 시도 비율", "가능"),
    "ft_rate":         ("shooting", "FGA 대비 자유투 시도", "가능"),
    "paint_pts_share": ("shooting", "득점 중 페인트 득점 비중", "확인 필요"),
    "orb_pct":         ("rebound",  "공격 리바운드 점유율", "가능"),
    "drb_pct":         ("rebound",  "수비 리바운드 점유율", "가능"),
    "stl_pct":         ("defense",  "상대 포제션 대비 스틸", "가능"),
    "blk_pct":         ("defense",  "상대 2점 시도 대비 블록", "가능"),
    "pf_per36":        ("defense",  "36분당 파울", "가능"),
}
AUX_FEATURES = {
    "fb_pts_share":           ("context",  "득점 중 속공 득점 비중", "확인 필요"),
    "sc_pts_share":           ("context",  "득점 중 세컨드찬스 득점 비중", "확인 필요"),
    **{f"shot_{g}_share": ("shot_type", f"FGA 중 {g} 비중", "확인 필요") for g in SHOT_TYPES},
}


def main():
    OUT_DIR.mkdir(exist_ok=True)

    pg, players, tg = load_data()
    print(f"[1] 출전 기록 {len(pg):,}행 / 경기 {pg.game_id.nunique()} / 선수 {pg.player_id.nunique()}")

    df = attach_team_context(pg, tg)
    agg = aggregate_season(df)

    meta = players.set_index("player_id")[
        ["player_name_original", "player_name_english", "nationality",
         "position", "height_cm", "weight_kg"]]
    is_import = meta["nationality"].reindex(agg.index).eq("Imported")

    feats = compute_features(agg, pool_mask=is_import)
    table = meta.reindex(agg.index).join(interpretation_cols(agg)).join(feats)
    table["is_import"] = is_import
    table["in_cluster_sample"] = is_import & (table["minutes"] >= MIN_MINUTES)
    table = table.reset_index().sort_values(["is_import", "minutes"], ascending=False)
    table.to_csv(OUT_DIR / "player_features_all.csv", index=False, encoding="utf-8-sig")

    model_cols = list(CLUSTER_FEATURES) + list(AUX_FEATURES)
    cluster_in = table.loc[table["in_cluster_sample"],
                           ["player_id", "player_name_english", "position", "height_cm",
                            "minutes"] + model_cols]
    cluster_in.to_csv(OUT_DIR / "cluster_input_imports.csv", index=False, encoding="utf-8-sig")

    dict_rows = [{"feature": f, "role": "core", "group": g, "desc": d, "kbl_portable": p}
                 for f, (g, d, p) in CLUSTER_FEATURES.items()]
    dict_rows += [{"feature": f, "role": "aux", "group": g, "desc": d, "kbl_portable": p}
                  for f, (g, d, p) in AUX_FEATURES.items()]
    pd.DataFrame(dict_rows).to_csv(OUT_DIR / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

    corr = cluster_in[list(CLUSTER_FEATURES)].corr()
    corr.to_csv(OUT_DIR / "feature_correlation.csv", encoding="utf-8-sig")

    print(f"[2] 외국선수 {int(is_import.sum())}명 중 {MIN_MINUTES}분 이상 → 군집 대상 {len(cluster_in)}명")
    high = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]
            if abs(corr.loc[a, b]) >= 0.7]
    print("[3] |상관| ≥ 0.7 피처 쌍:", "없음" if not high else "")
    for a, b, r in high:
        print(f"     {a} ↔ {b}: {r:+.2f}")

    focus = cluster_in[cluster_in["player_id"].isin(FOCUS)].set_index("player_name_english")
    print("[4] 관심 선수 핵심 피처 (보정값)")
    with pd.option_context("display.width", 200, "display.max_columns", 10,
                           "display.float_format", "{:.3f}".format):
        print(focus[list(CLUSTER_FEATURES)].T)
    print(f"\n저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()