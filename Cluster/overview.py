from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "1_effect_mcc_partner_input"
OUT_DIR = BASE_DIR / "0_overview_output"

TEAM_ID = 5                                           # Taoyuan Leopards
MCC = 10860
PARTNERS = {10861: "Diallo", 16: "Miller", 68: "Kromah"}   # 표에 나올 순서
IMPORTS = {MCC, *PARTNERS}
PHASES = ["regular_season", "play_in", "playoffs", "finals"]   # 프리시즌 포함하려면 "preseason" 추가
MIN_RELIABLE_MIN = 30      # 이 시간 미만 조합은 효율을 "해석 불가"로 표시
TOP_LOCALS = 10            # 히트맵에 보여줄 국내선수 수
COMBOS = [f"McC+{n}" for n in PARTNERS.values()]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
HEADER_BG, ROW_ALT = "#eceae4", "#f5f4f0"
POS, NEG = "#1a7f4b", "#c23b3a"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f5fa", "#2a78d6"])


def setup_style():
    korean = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
              "Noto Sans CJK JP", "Noto Sans CJK SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": [f for f in korean if f in installed] + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 110, "savefig.dpi": 200,
    })


# ─────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────
def load_stints():
    st = pd.read_csv(DATA_DIR / "stints.csv")
    st = st[st["valid"] & (st["duration_sec"] > 0) & st["competition_phase"].isin(PHASES)]
    st = st[(st.home_team_id == TEAM_ID) | (st.away_team_id == TEAM_ID)].copy()
    home = st.home_team_id == TEAM_ID
    st["lineup"] = st["home_ids"].where(home, st["away_ids"]).str.split("|").apply(
        lambda x: {int(i) for i in x})
    st["pf"] = st["home_pts"].where(home, st["away_pts"])       # Leopards 득점
    st["pa"] = st["away_pts"].where(home, st["home_pts"])       # Leopards 실점
    st["poss"] = (st["home_poss_est"] + st["away_poss_est"]) / 2
    st["minutes"] = st["duration_sec"] / 60
    st["combo"] = st["lineup"].apply(combo_label)
    return st


def combo_label(lineup):
    if MCC not in lineup:
        return None
    others = [nm for pid, nm in PARTNERS.items() if pid in lineup]
    if len(others) == 1:
        return f"McC+{others[0]}"
    return "McC 단독" if not others else None


def combo_summary(st):
    d = st[st["combo"].notna()]
    g = d.groupby("combo").agg(games=("game_id", "nunique"), minutes=("minutes", "sum"),
                               poss=("poss", "sum"), pf=("pf", "sum"), pa=("pa", "sum"))
    g["ORtg"] = g["pf"] / g["poss"] * 100
    g["DRtg"] = g["pa"] / g["poss"] * 100
    g["Net"] = g["ORtg"] - g["DRtg"]
    order = COMBOS + ["McC 단독"]
    return g.reindex([o for o in order if o in g.index])


def local_minutes(st, names):
    d = st[st["combo"].isin(COMBOS)]
    rows = [(pid, r.combo, r.minutes) for r in d.itertuples() for pid in r.lineup if pid not in IMPORTS]
    m = (pd.DataFrame(rows, columns=["pid", "combo", "minutes"])
           .pivot_table(index="pid", columns="combo", values="minutes", aggfunc="sum", fill_value=0))
    m = m.reindex(columns=[c for c in COMBOS if c in m.columns])
    m = m.loc[m.sum(axis=1).sort_values(ascending=False).index].head(TOP_LOCALS)
    m.index = [names.get(p, str(p)) for p in m.index]
    return m


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────
def fig_combo_table(g):
    header = ["조합", "경기", "동시 출전", "포제션", "공격 효율", "수비 효율", "넷레이팅"]
    cells, net_colors = [], []
    for combo, r in g.iterrows():
        ok = r.minutes >= MIN_RELIABLE_MIN
        cells.append([combo, f"{int(r.games)}", f"{r.minutes:.0f}분", f"{r.poss:.0f}",
                      f"{r.ORtg:.1f}" if ok else "–", f"{r.DRtg:.1f}" if ok else "–",
                      f"{r.Net:+.1f}" if ok else "-"])
        net_colors.append((POS if r.Net > 0 else NEG) if ok else MUTED)

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(cells) + 1.3))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center",
                   colWidths=[0.2, 0.08, 0.13, 0.11, 0.13, 0.13, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(ROW_ALT if row % 2 == 0 else SURFACE)
            if col == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.05
            if col == 6:
                cell.get_text().set_color(net_colors[row - 1])
                cell.get_text().set_fontweight("bold")
    ax.set_title("맥컬러 + 동반 외국선수 조합별 득실 (Leopards 관점, 100포제션당)", pad=4)
    fig.text(0.01, 0.02, "공격 효율 = 100포제션당 득점, 수비 효율 = 100포제션당 실점, "
             "넷레이팅 = 공격 − 수비.  상대 보정 전 원값", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "fig1_combo_table.png")
    plt.close(fig)


def fig_local_minutes(m, g):
    share = m / g.loc[m.columns, "minutes"].values          # 조합 출전 시간 중 비율
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(m) + 1.8))
    ax.imshow(share.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v, s = m.iat[i, j], share.iat[i, j]
            ax.text(j, i, f"{v:.0f}분\n({s:.0%})", ha="center", va="center", fontsize=9,
                    color="white" if s > 0.6 else INK)
    ax.set_xticks(range(m.shape[1]),
                  [f"{c}\n(총 {g.loc[c, 'minutes']:.0f}분)" for c in m.columns], fontsize=10)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("조합별로 함께 뛴 국내선수 출전 시간 (괄호 = 그 조합 출전 시간 중 비율)", pad=40)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_local_minutes.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    setup_style()
    pl = pd.read_csv(DATA_DIR / "players.csv")
    names = pl.set_index("player_id")["player_name_english"].fillna(
        pl.set_index("player_id")["player_name_original"]).to_dict()

    st = load_stints()
    g = combo_summary(st)
    m = local_minutes(st, names)

    fig_combo_table(g)
    fig_local_minutes(m, g)

    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print("[1] 조합별 득실\n", g[["games", "minutes", "poss", "ORtg", "DRtg", "Net"]])
        print("\n[2] 조합별 국내선수 동시 출전(분)\n", m)
    print(f"\n저장 위치: {OUT_DIR}  (fig1_combo_table.png, fig2_local_minutes.png)")


if __name__ == "__main__":
    main()