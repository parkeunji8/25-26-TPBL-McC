"""
출력  : 2_cluster_output/
  - fig1_pca_map.png          : 외국선수 역할 지도 (PCA 1·2축, 군집 색/모양, 관심 선수 표시)
  - fig2_cluster_profile.png  : 군집 × 피처 히트맵 (z-score)
  - cluster_model.joblib      : 3단계 KBL 매핑용 모델 (SAVE_MODEL=False면 저장 안 함)
  k 진단, 군집 요약, 관심 선수, 맥컬러와의 축별 차이, 공동 군집 비율은 CSV 없이 콘솔에만 출력

k 후보별로 Ward (k 후보 탐지용) -> K-means (베이스라인용)-> GMM 실행
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "2_build_feature_output" / "cluster_input_imports.csv"
OUT_DIR = BASE_DIR / "2_cluster_output"

FEATURES = ["usg_pct", "ast_pct", "tov_pct",
            "three_par", "ft_rate",
            "orb_pct", "drb_pct",
            "stl_pct", "blk_pct", "pf_per36"]

AXES = {
    "볼 점유":   ["usg_pct", "ast_pct"],
    "외곽 성향": ["three_par"],
    "골밑 접촉": ["ft_rate"],
    "리바운드":  ["orb_pct", "drb_pct"],
    "수비 활동": ["stl_pct", "blk_pct"],
}

K_RANGE = range(2, 8)
K_FINAL = None            # None이면 자동 선택. 콘솔의 k 진단을 보고 숫자로 고정 권장 (예: 4)
K_AUTO_RANGE = (4, 6)     # 자동 선택 시 고려할 범위 (해석 가능한 범위로 제한)
PCA_VAR = 0.80            # PCA 누적 설명력 기준
GMM_COV = "diag"          # 표본이 작아 대각 공분산 사용
N_BOOT = 200              # 안정성 부트스트랩 반복 수
BOOT_FRAC = 0.8           # 부트스트랩 시 사용할 선수 비율
SEED = 42
SAVE_MODEL = True         # KBL 매핑 단계에서 쓸 모델 저장 여부

MCC_ID = 10860
FOCUS = {10860: "McCullough", 16: "Miller", 68: "Kromah", 10861: "Diallo"}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]   # 색만으로 구분하지 않도록 모양 병행
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def setup_style():
    korean = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
              "Noto Sans CJK JP", "Noto Sans CJK SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in korean if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    })


def short_name(full: str) -> str:
    parts = [p for p in str(full).split() if p.rstrip(".").upper() not in {"JR", "SR", "II", "III", "IV"}]
    return parts[-1] if parts else str(full)


def prepare():
    df = pd.read_csv(IN_FILE).reset_index(drop=True)
    df["short"] = df["player_name_english"].apply(short_name)
    scaler = StandardScaler().fit(df[FEATURES])
    Z = scaler.transform(df[FEATURES])
    pca = PCA(random_state=SEED).fit(Z)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), PCA_VAR) + 1)
    pca = PCA(n_components=n_pc, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    print(f"[1] 선수 {len(df)}명 / 피처 {len(FEATURES)}개 → PCA {n_pc}개 성분 "
          f"(누적 설명력 {pca.explained_variance_ratio_.sum():.1%})")
    return df, scaler, Z, pca, P


def fit_kmeans(X, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)


def fit_gmm(X, k, seed=SEED):
    return GaussianMixture(n_components=k, covariance_type=GMM_COV, n_init=20,
                           reg_covar=1e-3, random_state=seed).fit(X)


def bootstrap_stability(X, k, rng):
    """선수 80%를 뽑아 K-means를 다시 돌렸을 때, 전체 데이터 결과와 얼마나 일치하나 (ARI 평균)"""
    full = fit_kmeans(X, k).labels_
    n = len(X)
    scores = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=int(n * BOOT_FRAC), replace=False)
        sub = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9))).fit(X[idx]).labels_
        scores.append(adjusted_rand_score(full[idx], sub))
    return float(np.mean(scores))


def diagnose(P, Lward):
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_RANGE:
        ward = fcluster(Lward, k, criterion="maxclust") - 1
        km = fit_kmeans(P, k).labels_
        gm = fit_gmm(P, k)
        gl = gm.predict(P)
        rows.append({
            "k": k,
            "sil_ward": silhouette_score(P, ward),
            "sil_kmeans": silhouette_score(P, km),
            "sil_gmm": silhouette_score(P, gl) if len(set(gl)) > 1 else np.nan,
            "bic_gmm": gm.bic(P),
            "stability_kmeans": bootstrap_stability(P, k, rng),
            "ari_kmeans_vs_ward": adjusted_rand_score(km, ward),
            "ari_kmeans_vs_gmm": adjusted_rand_score(km, gl),
            "min_cluster_size_kmeans": int(np.bincount(km).min()),
        })
        print(f"    k={k}  실루엣 W/K/G={rows[-1]['sil_ward']:.2f}/{rows[-1]['sil_kmeans']:.2f}/"
              f"{rows[-1]['sil_gmm']:.2f}  BIC={rows[-1]['bic_gmm']:.0f}  "
              f"안정성={rows[-1]['stability_kmeans']:.2f}  K-means↔GMM={rows[-1]['ari_kmeans_vs_gmm']:.2f}  "
              f"최소 군집={rows[-1]['min_cluster_size_kmeans']}명")
    return pd.DataFrame(rows)


def choose_k(diag):
    if K_FINAL is not None:
        return K_FINAL
    lo, hi = K_AUTO_RANGE
    d = diag[(diag.k >= lo) & (diag.k <= hi)].copy()
    # 실루엣(K-means)과 안정성 순위를 합산, 군집이 2명 미만이면 제외
    d = d[d.min_cluster_size_kmeans >= 2]
    d["score"] = d["sil_kmeans"].rank() + d["stability_kmeans"].rank() + d["ari_kmeans_vs_gmm"].rank()
    return int(d.sort_values(["score", "k"], ascending=[False, True]).iloc[0]["k"])


def final_model(df, P, k, Lward):
    km_model = fit_kmeans(P, k)
    raw = km_model.labels_
    order = (pd.Series(df["usg_pct"].values).groupby(raw).mean()
               .sort_values(ascending=False).index.tolist())
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in raw])
    centers = km_model.cluster_centers_[order]
    sigma2 = km_model.inertia_ / (P.shape[0] * P.shape[1])

    soft = {"centers": centers, "sigma2": sigma2}
    proba = soft_membership(P, soft)
    ward = fcluster(Lward, k, criterion="maxclust") - 1
    print(f"[4] 최종 k={k} (K-means 하드 라벨 + 소프트 확률)  |  "
          f"Ward 일치도 ARI={adjusted_rand_score(labels, ward):.2f}")
    return soft, km_model, order, labels, proba, ward


def soft_membership(P, soft):

    d2 = ((P[:, None, :] - soft["centers"][None, :, :]) ** 2).sum(axis=2)
    logit = -d2 / (2 * soft["sigma2"])
    logit -= logit.max(axis=1, keepdims=True)
    e = np.exp(logit)
    return e / e.sum(axis=1, keepdims=True)


def co_clustering(P, k, df):

    rng = np.random.default_rng(SEED + 1)
    n = len(P)
    together = np.zeros((n, n))
    seen = np.zeros((n, n))
    for _ in range(N_BOOT):
        idx = rng.choice(n, size=int(n * BOOT_FRAC), replace=False)
        lab = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9))).fit(P[idx]).labels_
        same = (lab[:, None] == lab[None, :]).astype(float)
        together[np.ix_(idx, idx)] += same
        seen[np.ix_(idx, idx)] += 1
    co = np.divide(together, seen, out=np.full_like(together, np.nan), where=seen > 0)
    return pd.DataFrame(co, index=df["short"], columns=df["short"])


def describe_clusters(df, Z, labels):
    zdf = pd.DataFrame(Z, columns=FEATURES)
    zmean = zdf.groupby(labels).mean()
    rawmean = df[FEATURES].groupby(labels).mean()
    size = pd.Series(labels).value_counts().sort_index()

    desc = {}
    for c in zmean.index:
        s = zmean.loc[c].sort_values()
        hi = ", ".join(f"{f}↑" for f in s.index[-2:][::-1] if s[f] > 0.3)
        lo = ", ".join(f"{f}↓" for f in s.index[:2] if s[f] < -0.3)
        desc[c] = " / ".join(x for x in [hi, lo] if x)

    prof = (zmean.add_prefix("z_")
              .join(rawmean.add_prefix("raw_"))
              .assign(n_players=size, auto_desc=pd.Series(desc),
                      members=pd.Series(df["short"].values).groupby(labels).agg(", ".join)))
    prof.index.name = "cluster"
    return prof, zmean


def focus_table(df, Z, labels, proba):
    rows = []
    for pid, name in FOCUS.items():
        i = df.index[df["player_id"] == pid]
        if len(i) == 0:
            print(f"[경고] {name}({pid}) 이(가) 군집 대상에 없음")
            continue
        i = i[0]
        dist = np.linalg.norm(Z - Z[i], axis=1)
        nn = [j for j in np.argsort(dist) if j != i][:3]
        rows.append({
            "player": name, "cluster": labels[i],
            **{f"p_cluster{c}": round(p, 3) for c, p in enumerate(proba[i])},
            "nearest_3": ", ".join(f"{df.loc[j, 'short']}({dist[j]:.2f})" for j in nn),
        })
    return pd.DataFrame(rows)


def mccullough_relation(df, Z):
    zdf = pd.DataFrame(Z, columns=FEATURES, index=df["player_id"])
    if MCC_ID not in zdf.index:
        return pd.DataFrame()
    mcc = zdf.loc[MCC_ID]
    rows = []
    for pid, name in FOCUS.items():
        if pid == MCC_ID or pid not in zdf.index:
            continue
        diff = zdf.loc[pid] - mcc
        row = {"partner": name, "euclid_dist": float(np.linalg.norm(diff))}
        row.update({f"axis_{a}": diff[f].mean() for a, f in AXES.items()})
        row.update({f"diff_{f}": diff[f] for f in FEATURES})
        rows.append(row)
    return pd.DataFrame(rows).round(3)


# ─────────────────────────────────────────────────────────────
# 시각화 (2장)
# ─────────────────────────────────────────────────────────────
def fig_pca_map(df, P, pca, labels, k):
    """그림 1: 선수들이 역할 공간 어디에 있고, 관심 4명이 어느 군집에 속하나"""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for c in range(k):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=70, color=SERIES[c % 8], marker=MARKERS[c % 8],
                   edgecolor=SURFACE, linewidth=1.5, label=f"군집 {c} (n={m.sum()})", zorder=3)
        cx, cy = P[m, 0].mean(), P[m, 1].mean()
        ax.text(cx, cy, f"C{c}", fontsize=13, fontweight="bold", color=INK2, alpha=0.35,
                ha="center", va="center", zorder=2)
    for pid, name in FOCUS.items():
        i = df.index[df.player_id == pid]
        if len(i):
            i = i[0]
            ax.scatter(P[i, 0], P[i, 1], s=230, facecolor="none", edgecolor=INK, linewidth=1.8, zorder=4)
            ax.annotate(name, (P[i, 0], P[i, 1]), xytext=(9, 9), textcoords="offset points",
                        fontsize=10.5, fontweight="bold", color=INK, zorder=5)

    for i in df.index[~df.player_id.isin(FOCUS)]:
        ax.annotate(df.loc[i, "short"], (P[i, 0], P[i, 1]), xytext=(5, -9), textcoords="offset points",
                    fontsize=7, color=MUTED)

    load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
    scale = 0.8 * min(np.abs(P[:, 0]).max() / np.abs(load[:, 0]).max(),
                      np.abs(P[:, 1]).max() / np.abs(load[:, 1]).max())
    for f, (lx, ly) in zip(FEATURES, load):
        ax.annotate("", xy=(lx * scale, ly * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        ax.text(lx * scale * 1.08, ly * scale * 1.08, f, fontsize=8, color=INK2, ha="center", va="center")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.set_title("TPBL 외국선수 역할 지도 (PCA 1·2축, 화살표 = 피처 방향)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_pca_map.png")
    plt.close(fig)


def fig_cluster_profile(zmean, prof):
    """그림 2: 각 군집이 어떤 역할인가"""
    k = len(zmean)
    fig, ax = plt.subplots(figsize=(11, 0.75 * k + 2))
    lim = max(1.5, np.abs(zmean.values).max())
    im = ax.imshow(zmean.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for (r, c), v in np.ndenumerate(zmean.values):
        ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                color=INK if abs(v) < lim * 0.6 else "white")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=30, ha="right")
    ax.set_yticks(range(k), [f"C{c} (n={prof.loc[c, 'n_players']})" for c in zmean.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("외국선수 평균 대비 z-score", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("군집별 프로필 (빨강 = 평균보다 높음, 파랑 = 낮음)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_cluster_profile.png")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    setup_style()

    df, scaler, Z, pca, P = prepare()
    Lward = linkage(P, method="ward")

    print("[2] k 진단")
    diag = diagnose(P, Lward)

    k = choose_k(diag)
    print(f"[3] 선택 k = {k}" + ("  (자동 선택: 위 k 진단을 보고 K_FINAL로 고정 권장)" if K_FINAL is None else ""))

    soft, km_model, order, labels, proba, ward = final_model(df, P, k, Lward)
    co = co_clustering(P, k, df)

    prof, zmean = describe_clusters(df, Z, labels)
    focus = focus_table(df, Z, labels, proba)
    rel = mccullough_relation(df, Z)

    mcc_short = df.loc[df.player_id == MCC_ID, "short"].iloc[0]
    mcc_co = co[mcc_short].drop(mcc_short).sort_values(ascending=False)

    if SAVE_MODEL:
        joblib.dump({"features": FEATURES, "scaler": scaler, "pca": pca, "kmeans": km_model, "soft": soft,
                     "cluster_order": order, "k": k}, OUT_DIR / "cluster_model.joblib")

    fig_pca_map(df, P, pca, labels, k)
    fig_cluster_profile(zmean, prof)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 80):
        print("\n[5] 군집 요약")
        print(prof[["n_players", "auto_desc", "members"]])
        print("\n[6] 관심 선수")
        print(focus)
        print("\n[7] 맥컬러와의 축별 차이 (파트너 − 맥컬러, z 단위. +면 파트너가 더 높음)")
        print(rel[["partner", "euclid_dist"] + [c for c in rel.columns if c.startswith("axis_")]])
        print("\n[8] 부트스트랩에서 맥컬러와 같은 군집에 묶인 비율 (상위 8명 + 동반 외국선수)")
        partners = [df.loc[df.player_id == p, "short"].iloc[0] for p in FOCUS if p != MCC_ID]
        print(mcc_co.head(8).round(2).to_string())
        print("  동반 외국선수:", ", ".join(f"{p} {mcc_co[p]:.2f}" for p in partners))
    saved = "fig1_pca_map.png, fig2_cluster_profile.png" + (", cluster_model.joblib" if SAVE_MODEL else "")
    print(f"\n저장 위치: {OUT_DIR}  ({saved})")


if __name__ == "__main__":
    main()