"""
폐업수 기준 EDA 그림 (c1~c3). 실행: python 폐업수_그림.py <새로해보는3 csv>  (먼저 폐업수_분석.py 실행해 tables/ 생성)
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "WenQuanYi Zen Hei"]:
    if any(f.name == name for f in fm.fontManager.ttflist): plt.rcParams["font.family"] = name; break
INK, INK2, GRID, MUTE = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
C = {"음식점": "#2a78d6", "카페": "#eb6834"}
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "savefig.dpi": 160, "savefig.bbox": "tight"})
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); Y, N = "다음분기_폐업수", "읍면동업태_영업점포수"

# c1 분포 + 점포수와의 관계
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
v = d[Y].clip(upper=15).value_counts().sort_index()
a.bar(v.index, v.values, width=0.8, color=[C["음식점"] if i else INK2 for i in v.index])
a.set_xticks([0, 5, 10, 15]); a.set_xticklabels(["0", "5", "10", "15+"])
a.text(0.6, v.iloc[0] * 0.97, f"0건 칸 {(d[Y]==0).mean():.0%}", fontsize=10, color=INK, va="top")
a.set_xlabel("다음 분기 폐업수(건)"); a.set_ylabel("칸 수"); a.set_title("대부분 0~3건, 소수 칸만 크다", loc="left"); a.grid(axis="x", visible=False)
for u in ["음식점", "카페"]:
    s = d[d.업태구분명 == u]; b.scatter(s[N], s[Y], s=9, alpha=0.45, color=C[u], label=u, linewidths=0)
b.set_xscale("symlog", linthresh=10); b.set_yscale("symlog", linthresh=1)
b.set_xlabel("지금 영업 중인 점포수 (로그 눈금)"); b.set_ylabel("다음 분기 폐업수 (로그 눈금)")
b.set_title("점포가 많을수록 폐업수도 많다 (상관 0.90)", loc="left"); b.legend(frameon=False, loc="upper left", markerscale=2)
fig.savefig("figures/c1_폐업수_분포.png"); plt.close()

# c2 분기별 폐업수 합
q = pd.read_csv("tables/c2_분기별_폐업수합.csv", index_col=0, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(q))
ax.bar(x, q.음식점, 0.7, color=C["음식점"], label="음식점"); ax.bar(x, q.카페, 0.7, bottom=q.음식점, color=C["카페"], label="카페")
for i, (k, r) in enumerate(q.iterrows()):
    ax.text(i, r.sum() + 15, f"{int(r.sum()):,}", ha="center", fontsize=8.5, color=INK if k.endswith("Q4") else INK2,
            fontweight="bold" if k.endswith("Q4") else "normal")
te = [i for i, k in enumerate(q.index) if k in ("2026Q2", "2026Q3")]
ax.axvspan(te[0] - 0.45, te[-1] + 0.45, color="#f3f2ee", zorder=0); ax.text(np.mean(te), q.sum(1).max() * 0.98, "테스트", ha="center", color=INK2, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(q.index, rotation=45, fontsize=9); ax.set_xlabel("폐업이 일어난 분기 (예측대상분기)")
ax.set_ylabel("인천 전체 폐업수"); ax.set_title("4분기마다 폐업수가 다른 분기의 1.3~1.7배로 뛴다", loc="left"); ax.legend(frameon=False, loc="upper left", ncol=2)
ax.grid(axis="x", visible=False); fig.savefig("figures/c2_분기별_폐업수.png"); plt.close()

# c3 변수 상관: 폐업수와 vs 폐업률과
cr = pd.read_csv("tables/c4_변수_폐업수상관.csv", index_col=0, encoding="utf-8-sig")
cr = cr[~cr.index.isin(["읍면동업태_직전분기말점포수"])].head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.6, 5.2)); y = np.arange(len(cr)); h = 0.38
ax.barh(y + h / 2, cr.폐업수와, h, color=C["음식점"], label="폐업수(타깃)와의 상관")
ax.barh(y - h / 2, cr.폐업률과, h, color=MUTE, label="폐업률(점포 1곳당 위험)과의 상관")
for i, (p, r) in enumerate(zip(cr.폐업수와, cr.폐업률과)):
    ax.text(p + (0.01 if p >= 0 else -0.01), i + h / 2, f"{p:.2f}", va="center", ha="left" if p >= 0 else "right", fontsize=8.5, color=INK2)
ax.set_xlim(-0.45, 0.9); ax.set_yticks(y); ax.set_yticklabels([s.replace("읍면동업태_", "") for s in cr.index]); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("Spearman 상관 (학습 2023~2025, 전체 칸)"); ax.set_title("폐업수와 강하게 묶인 변수는 대부분 '동네 크기' 변수", loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=9); ax.grid(axis="y", visible=False)
fig.savefig("figures/c3_변수_폐업수상관.png"); plt.close()
