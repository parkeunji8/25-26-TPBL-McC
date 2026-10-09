"""
출력  : 1_effect_mcc_partner_output/
  - fig1_pair_net.png    : 외국선수 조합별 팀 넷레이팅 (상대 보정 + 경기 단위 부트스트랩 95% CI)
  - fig2_mechanism.png   : 조합별로 팀이 어떻게 이겼나(위) / 맥컬러 역할이 어떻게 바뀌었나(아래)
  수치 결과(A~F)는 CSV로 저장하지 않고 콘솔에만 출력
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "1_effect_mcc_partner_input"
OUT_DIR = BASE_DIR / "1_effect_mcc_partner_output"

TEAM_ID = 5 # Taoyuan Leopards
MCC = 10860
PARTNERS = {16: "Miller", 68: "Kromah", 10861: "Diallo"}
IMPORTS = {MCC: "McCullough", **PARTNERS}
PHASES = ["regular_season", "play_in", "playoffs", "finals"]   # 프리시즌 제외

GARBAGE_SEC = 300          # 4쿼터 이후 남은 시간(초) 기준
GARBAGE_MARGIN = 20        # 점수차 기준
QUARTER_SEC = 720          # 쿼터 길이 (12분)
N_BOOT = 2000
SEED = 42
RIDGE_ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
PARTNER_UNPENALIZED = 100.0  # 동반 외국선수 더미는 사실상 벌점 없이 (스케일 키워 벌점 무력화)
MIN_LOCAL_MIN = 30         # Ridge 통제변수로 넣을 국내선수 최소 동반 출전(분)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
PAIR_COLORS = {"McC+Miller": SERIES[0], "McC+Kromah": SERIES[1], "McC+Diallo": SERIES[2]}
REF_COLOR = "#b5b4ae"


def setup_style():
    korean = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
              "Noto Sans CJK JP", "Noto Sans CJK SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": [f for f in korean if f in installed] + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    })


def load():
    st = pd.read_csv(DATA_DIR / "stints.csv")
    sp = pd.read_csv(DATA_DIR / "stint_players.csv")
    tg = pd.read_csv(DATA_DIR / "team_game_logs.csv")
    pl = pd.read_csv(DATA_DIR / "players.csv")
    ev = pd.read_csv(DATA_DIR / "events.csv")
    return st, sp, tg, pl, ev


def score_at_start(st, ev):

    e = ev[["game_id", "elapsed_sec", "order", "home_score", "away_score"]].sort_values(
        ["elapsed_sec", "order"])
    s = st[["stint_id", "game_id", "start_sec"]].sort_values("start_sec")
    m = pd.merge_asof(s, e.drop(columns="order"), left_on="start_sec", right_on="elapsed_sec",
                      by="game_id", direction="backward")
    return m.set_index("stint_id")[["home_score", "away_score"]].fillna(0)


def build_stints(st, ev, tg):
    st = st[st["valid"] & (st["duration_sec"] > 0) & st["competition_phase"].isin(PHASES)].copy()
    st = st[(st.home_team_id == TEAM_ID) | (st.away_team_id == TEAM_ID)]
    home = st.home_team_id == TEAM_ID
    pick = lambda h, a: st[h].where(home, st[a])
    st["is_home"] = home.astype(int)
    st["opp_id"] = st.away_team_id.where(home, st.home_team_id)
    st["lineup"] = pick("home_ids", "away_ids")
    st["pf"], st["pa"] = pick("home_pts", "away_pts"), pick("away_pts", "home_pts")
    st["poss"] = pick("home_poss_est", "away_poss_est")
    st["opp_poss"] = pick("away_poss_est", "home_poss_est")
    st["poss_avg"] = (st["poss"] + st["opp_poss"]) / 2

    ids = st["lineup"].str.split("|").apply(lambda x: {int(i) for i in x})
    st["lineup_set"] = ids
    st["mcc_on"] = ids.apply(lambda s: MCC in s)
    st["imports_on"] = ids.apply(lambda s: tuple(sorted(IMPORTS[i] for i in s if i in IMPORTS)))
    st["pair"] = st["imports_on"].apply(label_pair)

    sc = score_at_start(st, ev)
    margin = (sc["home_score"] - sc["away_score"]).abs().reindex(st["stint_id"]).values
    remaining = 4 * QUARTER_SEC - st["start_sec"]
    st["garbage"] = (st["quarter"] >= 4) & (remaining <= GARBAGE_SEC) & (margin >= GARBAGE_MARGIN)

    rs = tg[tg.competition_phase == "regular_season"].groupby("team_id")[["ORtg", "DRtg"]].mean()
    st = st.join(rs.rename(columns={"ORtg": "opp_ortg", "DRtg": "opp_drtg"}), on="opp_id")
    st["xpf"] = st["poss_avg"] * st["opp_drtg"] / 100
    st["xpa"] = st["poss_avg"] * st["opp_ortg"] / 100
    return st


def label_pair(imports):
    s = set(imports)
    if s == {"McCullough"}:
        return "McC 단독"
    if "McCullough" in s and len(s) == 2:
        other = (s - {"McCullough"}).pop()
        return f"McC+{other}"
    if "McCullough" not in s and len(s) == 2:
        return "+".join(sorted(s)) + " (McC 없음)"
    return "기타 (" + ",".join(sorted(s)) + ")" if s else "외국선수 없음"

def rate(d):
    p = d["poss_avg"].sum()
    return pd.Series({
        "minutes": d["duration_sec"].sum() / 60, "poss": p, "games": d["game_id"].nunique(),
        "ORtg": d["pf"].sum() / p * 100, "DRtg": d["pa"].sum() / p * 100,
        "Net": (d["pf"].sum() - d["pa"].sum()) / p * 100,
        "Net_adj": ((d["pf"] - d["xpf"]).sum() - (d["pa"] - d["xpa"]).sum()) / p * 100,
        "ORtg_adj": (d["pf"] - d["xpf"]).sum() / p * 100,
        "DRtg_adj": (d["pa"] - d["xpa"]).sum() / p * 100,
    })


def game_bootstrap(d, stat_fn, n=N_BOOT, seed=SEED):

    rng = np.random.default_rng(seed)
    games = d["game_id"].unique()
    by_game = {g: x for g, x in d.groupby("game_id")}
    out = []
    for _ in range(n):
        pick = rng.choice(games, size=len(games), replace=True)
        out.append(stat_fn(pd.concat([by_game[g] for g in pick])))
    return np.array(out)


def pair_summary(st):
    rows = []
    for pair, d in st.groupby("pair"):
        r = rate(d)
        if r["minutes"] >= 20:
            b = game_bootstrap(d, lambda x: rate(x)["Net_adj"], n=N_BOOT // 2)
            r["Net_adj_lo"], r["Net_adj_hi"] = np.percentile(b, [2.5, 97.5])
        r["pair"] = pair
        rows.append(r)
    out = pd.DataFrame(rows).set_index("pair").sort_values("minutes", ascending=False)
    return out


def team_box(st, sp):
    sp = sp[sp["stint_id"].isin(st["stint_id"])]
    cols = ["PTS", "FGA", "FGM", "3PM", "3PA", "FTA", "FTM", "OREB", "DREB", "TOV", "AST"]
    own = sp[sp.team_id == TEAM_ID].groupby("stint_id")[cols].sum().add_prefix("tm_")
    opp = sp[sp.team_id != TEAM_ID].groupby("stint_id")[cols].sum().add_prefix("op_")
    return st.join(own, on="stint_id").join(opp, on="stint_id").fillna(
        {c: 0 for c in list(own.columns) + list(opp.columns)})


def four_factors(d):
    s = d.sum(numeric_only=True)
    return pd.Series({
        "off_eFG": (s.tm_FGM + 0.5 * s["tm_3PM"]) / s.tm_FGA,
        "off_TOV%": s.tm_TOV / (s.tm_FGA + 0.44 * s.tm_FTA + s.tm_TOV),
        "off_ORB%": s.tm_OREB / (s.tm_OREB + s.op_DREB),
        "off_FTr": s.tm_FTA / s.tm_FGA,
        "def_eFG": (s.op_FGM + 0.5 * s["op_3PM"]) / s.op_FGA,
        "def_TOV%": s.op_TOV / (s.op_FGA + 0.44 * s.op_FTA + s.op_TOV),
        "def_DRB%": s.tm_DREB / (s.tm_DREB + s.op_OREB),
        "def_FTr": s.op_FTA / s.op_FGA,
        "off_3PAr": s["tm_3PA"] / s.tm_FGA,
        "def_3PAr": s["op_3PA"] / s.op_FGA,
    })

ZONES = ["rim", "paint", "mid", "c3", "ab3"]


def individual(st, sp, pid):
    rows = []
    sp = sp[sp["stint_id"].isin(st["stint_id"])]
    for pair, d in st.groupby("pair"):
        me = sp[(sp.player_id == pid) & sp.stint_id.isin(d.stint_id)].sum(numeric_only=True)
        if me.get("FGA", 0) + me.get("FTA", 0) == 0:
            continue
        tm = d[[c for c in d.columns if c.startswith("tm_") or c.startswith("op_")]].sum()
        p = d["poss_avg"].sum()
        plays = me.FGA + 0.44 * me.FTA + me.TOV
        tm_plays = tm.tm_FGA + 0.44 * tm.tm_FTA + tm.tm_TOV
        r = {"pair": pair, "minutes": d["duration_sec"].sum() / 60, "FGA": me.FGA,
             "PTS_per100": me.PTS / p * 100,
             "TS%": me.PTS / (2 * (me.FGA + 0.44 * me.FTA)),
             "usage_share": plays / tm_plays,
             "AST_per100": me.AST / p * 100,
             "TOV_per100": me.TOV / p * 100,
             "assisted_FGM%": me.FGM_assisted / me.FGM if me.FGM else np.nan,
             "ORB%": me.OREB / (tm.tm_OREB + tm.op_DREB),
             "DRB%": me.DREB / (tm.tm_DREB + tm.op_OREB),
             "3PA_rate": me["3PA"] / me.FGA if me.FGA else np.nan,
             "FT_rate": me.FTA / me.FGA if me.FGA else np.nan}
        for z in ZONES:
            r[f"zone_{z}%"] = me[f"{z}_FGA"] / me.FGA if me.FGA else np.nan
        for z in ["rim", "paint", "mid"]:
            r[f"zone_{z}_FG%"] = me[f"{z}_FGM"] / me[f"{z}_FGA"] if me[f"{z}_FGA"] else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("pair")


def assist_network(st, ev, pl):
    a = ev[(ev.event_type == "Assist") & ev.assist_to.notna()].copy()
    s = st[["stint_id", "game_id", "start_sec", "end_sec", "pair", "poss_avg"]].sort_values("start_sec")
    a = pd.merge_asof(a.sort_values("elapsed_sec"), s, left_on="elapsed_sec", right_on="start_sec",
                      by="game_id", direction="backward", suffixes=("_ev", ""))
    a = a[a["stint_id"].notna() & (a["elapsed_sec"] <= a["end_sec"])]
    a = a[a.team_id == TEAM_ID]
    names = pl.set_index("player_id")["player_name_english"].to_dict()
    rows = []
    for pair, d in a.groupby("pair"):
        p = st.loc[st.pair == pair, "poss_avg"].sum()
        to_mcc = d[d.assist_to == MCC]
        from_mcc = d[d.player_id == MCC]
        r = {"pair": pair, "poss": p,
             "assists_to_McC_per100": len(to_mcc) / p * 100,
             "assists_from_McC_per100": len(from_mcc) / p * 100}
        top = to_mcc["player_id"].map(names).value_counts().head(3)
        r["top_assisters_to_McC"] = ", ".join(f"{k}({v})" for k, v in top.items())
        for pid, nm in PARTNERS.items():
            r[f"{nm}→McC"] = int(((d.player_id == pid) & (d.assist_to == MCC)).sum())
            r[f"McC→{nm}"] = int(((d.player_id == MCC) & (d.assist_to == pid)).sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("pair")


def ridge_design(d, locals_):
    X = pd.DataFrame(index=d.index)
    for nm in PARTNERS.values():
        X[f"partner_{nm}"] = (d["pair"] == f"McC+{nm}").astype(float)
    for pid in locals_:
        X[f"local_{pid}"] = d["lineup_set"].apply(lambda s: float(pid in s))
    X["opp_net"] = (d["opp_ortg"] - d["opp_drtg"])           # 상대 강도
    X["home"] = d["is_home"].astype(float)
    y = (d["pf"] - d["pa"]) / d["poss_avg"] * 100
    w = d["poss_avg"]
    return X, y, w


def fit_ridge(X, y, w, alpha):

    m = Ridge(alpha=alpha, fit_intercept=False)
    Xc = X.copy()
    part = [c for c in X.columns if c.startswith("partner_")]
    other = [c for c in X.columns if c not in part]
    mu = np.average(Xc[other], axis=0, weights=w)
    Xc[other] = Xc[other] - mu
    Xc[part] = Xc[part] * PARTNER_UNPENALIZED
    m.fit(Xc.values, y.values, sample_weight=w.values)
    coef = pd.Series(m.coef_, index=X.columns)
    coef[part] = coef[part] * PARTNER_UNPENALIZED
    return coef


def ridge_effects(st, sp):
    d = st[(st.pair.isin([f"McC+{n}" for n in PARTNERS.values()]))].copy()
    d = d[d["poss_avg"] > 0]
    tm = sp[(sp.team_id == TEAM_ID) & sp.stint_id.isin(d.stint_id) & ~sp.player_id.isin(IMPORTS)]
    mins = tm.groupby("player_id")["duration_sec"].sum() / 60
    locals_ = mins[mins >= MIN_LOCAL_MIN].index.tolist()
    X, y, w = ridge_design(d, locals_)

    gkf = GroupKFold(n_splits=5)
    cv = {}
    for a in RIDGE_ALPHAS:
        err = []
        for tr, te in gkf.split(X, y, d["game_id"]):
            coef = fit_ridge(X.iloc[tr], y.iloc[tr], w.iloc[tr], a)
            other = [c for c in X.columns if not c.startswith("partner_")]
            mu = np.average(X.iloc[tr][other], axis=0, weights=w.iloc[tr])
            Xte = X.iloc[te].copy(); Xte[other] = Xte[other] - mu
            pred = Xte.values @ coef.values
            err.append(np.average((y.iloc[te] - pred) ** 2, weights=w.iloc[te]))
        cv[a] = np.mean(err)
    alpha = min(cv, key=cv.get)
    coef = fit_ridge(X, y, w, alpha)

    rng = np.random.default_rng(SEED)
    games = d["game_id"].unique()
    idx_by_game = {g: np.where(d["game_id"].values == g)[0] for g in games}
    boots = []
    for _ in range(N_BOOT // 4):
        idx = np.concatenate([idx_by_game[g] for g in rng.choice(games, len(games), replace=True)])
        boots.append(fit_ridge(X.iloc[idx], y.iloc[idx], w.iloc[idx], alpha))
    B = pd.DataFrame(boots)

    rows = []
    for nm in PARTNERS.values():
        c = f"partner_{nm}"
        rows.append({"partner": nm, "effect_net100": coef[c],
                     "ci_lo": B[c].quantile(.025), "ci_hi": B[c].quantile(.975)})
    eff = pd.DataFrame(rows).set_index("partner")

    names = list(PARTNERS.values())
    diffs = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = names[i], names[j]
            dd = B[f"partner_{a}"] - B[f"partner_{b}"]
            diffs.append({"comparison": f"{a} − {b}", "diff": coef[f"partner_{a}"] - coef[f"partner_{b}"],
                          "ci_lo": dd.quantile(.025), "ci_hi": dd.quantile(.975),
                          "P(diff>0)": (dd > 0).mean()})
    return eff, pd.DataFrame(diffs), alpha, cv, locals_


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def fig_pair_net(A):
    """그림 1: 조합별 팀 넷레이팅 (결과)"""
    focus = [f"McC+{n}" for n in PARTNERS.values()]
    ref = [p for p in A.index if "McC 없음" in p and A.loc[p, "minutes"] >= 20]
    rows = [p for p in focus if p in A.index] + ref
    d = A.loc[rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.8))
    y = np.arange(len(rows))[::-1]
    for yi, (p, r) in zip(y, d.iterrows()):
        c = PAIR_COLORS.get(p, REF_COLOR)
        if pd.notna(r.get("Net_adj_lo")):
            ax.plot([r.Net_adj_lo, r.Net_adj_hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.scatter(r.Net_adj, yi, s=90, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(r.Net, yi, s=40, facecolor="none", edgecolor=c, linewidth=1.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [f"{p}\n{int(r.minutes)}분 · {int(r.poss)}포제션" for p, r in d.iterrows()], fontsize=9)
    for yi, (p, r) in zip(y, d.iterrows()):
        ax.annotate(f"{r.Net_adj:+.1f}", (r.Net_adj, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("100포제션당 넷레이팅 (● 상대 보정, ○ 원값, 선 = 경기 단위 부트스트랩 95% CI)")
    ax.set_title("외국선수 조합별 팀 득실 (회색 = 맥컬러 없는 참고 조합)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_pair_net.png")
    plt.close(fig)


def fig_mechanism(B, C):
    """그림 2: 왜 그런 결과가 나왔나 — 위 줄은 팀 지표, 아래 줄은 맥컬러 개인 지표"""
    focus = [p for p in [f"McC+{n}" for n in PARTNERS.values()] if p in B.index and p in C.index]
    panels = [
        # (데이터, 컬럼, 제목, 좋은 방향: True=높을수록, False=낮을수록, None=방향 없음)
        (B, "def_eFG",       "[팀] 상대 eFG%",              False),
        (B, "off_ORB%",      "[팀] 공격 리바운드%",          True),
        (B, "off_FTr",       "[팀] 자유투 유도 (FTA/FGA)",   True),
        (C, "TS%",           "[맥컬러] TS%",                True),
        (C, "usage_share",   "[맥컬러] 사용 비중",           None),
        (C, "assisted_FGM%", "[맥컬러] 어시스트받은 득점 비율", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
    yy = np.arange(len(focus))[::-1]
    for ax, (src, col, title, good_high) in zip(axes.flat, panels):
        vals = src.loc[focus, col].astype(float)
        ax.barh(yy, vals.values, color=[PAIR_COLORS[p] for p in focus], height=0.6)
        for yi, v in zip(yy, vals.values):
            ax.text(v, yi, f" {v:.1%}", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(yy, [p.replace("McC+", "with ") for p in focus])
        ax.set_xlim(0, vals.max() * 1.3)
        ax.xaxis.set_visible(False)
        ax.grid(False)
        arrow = "" if good_high is None else (" ↑좋음" if good_high else " ↓좋음")
        ax.set_title(title + arrow, fontsize=10.5)
    fga = " · ".join(f"{p.replace('McC+', '')} {int(C.loc[p, 'FGA'])}" for p in focus)
    fig.suptitle("맥컬러 + 동반 외국선수: 팀은 어떻게 이겼나(위) / 맥컬러 역할은 어떻게 바뀌었나(아래)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, f"아래 줄 표본(맥컬러 야투 시도): {fga}", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_DIR / "fig2_mechanism.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    setup_style()
    st, sp, tg, pl, ev = load()
    st = build_stints(st, ev, tg)
    n0 = len(st)
    garbage_min = st.loc[st.garbage, "duration_sec"].sum() / 60
    st = st[~st.garbage]
    st = team_box(st, sp)
    print(f"[1] Leopards stint {n0}개 중 가비지 타임 {n0 - len(st)}개({garbage_min:.0f}분) 제외 → {len(st)}개")

    A = pair_summary(st)
    big = A.index[A["minutes"] >= 20]
    B = pd.DataFrame({p: four_factors(st[st.pair == p]) for p in big}).T
    C = individual(st, sp, MCC)
    D = pd.concat({nm: individual(st[st.pair == f"McC+{nm}"], sp, pid) for pid, nm in PARTNERS.items()})
    E = assist_network(st, ev, pl)
    eff, diffs, alpha, cv, locals_ = ridge_effects(st, sp)

    fig_pair_net(A)
    fig_mechanism(B, C)

    focus = [f"McC+{n}" for n in PARTNERS.values()]
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\n[A] 조합별 득실 (100포제션당)")
        print(A[["minutes", "poss", "games", "ORtg", "DRtg", "Net", "Net_adj", "Net_adj_lo", "Net_adj_hi"]])
        print("\n[B] 4 factors")
        print(B.loc[[p for p in focus if p in B.index]].T)
        print("\n[C] 맥컬러 개인")
        print(C.loc[[p for p in focus if p in C.index]].T)
        print("\n[D] 동반 외국선수 개인 (맥컬러와 함께 뛸 때)")
        print(D.droplevel(1)[["FGA", "PTS_per100", "TS%", "usage_share", "AST_per100", "ORB%", "DRB%"]])
        print("\n[E] 어시스트 흐름")
        print(E.loc[[p for p in focus if p in E.index]].T)
        print(f"\n[F] Ridge (α={alpha}, 통제 국내선수 {len(locals_)}명)")
        print(eff)
        print(diffs)
    print(f"\n그림 저장 위치: {OUT_DIR}  (fig1_pair_net.png, fig2_mechanism.png)")


if __name__ == "__main__":
    main()