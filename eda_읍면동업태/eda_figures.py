"""
인천 분기×읍면동×업태 데이터 EDA (PPT 2장용 그래프 4개)
실행: python eda_figures.py <데이터.csv> <저장폴더>
- 집계 폐업률은 모두 '폐업수 합 ÷ 점포수 합'(점포수 가중)으로 계산
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import spearmanr

src = sys.argv[1] if len(sys.argv) > 1 else "인천_분기x읍면동x업태_다음분기폐업률.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(out, exist_ok=True)

# ---- 한글 폰트 (있는 것 중 첫 번째) ----
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "WenQuanYi Zen Hei"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- 색상/스타일 ----
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"음식점": "#2a78d6", "카페": "#eb6834"}
NEG = "#eb6834"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
pct1 = matplotlib.ticker.PercentFormatter(1.0, decimals=1)

d = pd.read_csv(src, encoding="utf-8-sig")
rate = lambda g: g["다음분기_폐업수"].sum() / g["읍면동업태_영업점포수"].sum()

# =========================================================
# 그림 1. 점포수가 적은 칸일수록 폐업률이 0 아니면 크게 튄다
# =========================================================
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for u in ["음식점", "카페"]:
    s = d[d["업태구분명"] == u]
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(s))
    ax.scatter(s["읍면동업태_영업점포수"] * np.exp(jitter * 0.2), s["다음분기_폐업률"],
               s=9, alpha=0.35, color=C[u], edgecolors="none", label=u)
ax.set_xscale("log")
ax.set_ylim(-0.02, 0.5)
ax.yaxis.set_major_formatter(pct)
ax.set_xlabel("읍면동·업태 영업점포수 (로그 눈금)")
ax.set_ylabel("다음분기 폐업률")
ax.set_title("점포가 적은 칸은 폐업률이 0 아니면 크게 튄다", loc="left")
bins = [(1, 9, "1~9개"), (10, 29, "10~29개"), (30, 99, "30~99개"), (100, 10**5, "100개 이상")]
txt = "폐업률 0인 칸 비율\n" + "\n".join(
    f"{lab}: {(d.loc[d['읍면동업태_영업점포수'].between(a, b), '다음분기_폐업률'] == 0).mean():.0%}"
    for a, b, lab in bins)
ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
ax.legend(loc="upper center", frameon=False, markerscale=2, ncol=2)
ax.text(0, -0.2, "※ y축 50% 위 칸 일부 생략(최대 100%). 점 1개 = 한 분기의 읍면동×업태 한 칸",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(f"{out}/fig1_점포수_vs_폐업률.png", dpi=200); plt.close(fig)

# =========================================================
# 그림 2. 분기별 폐업률 추이 (4분기 급증)
# =========================================================
q = d.groupby(["예측대상분기", "업태구분명"]).apply(rate).unstack()
x = np.arange(len(q))
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for i, lab in enumerate(q.index):
    if lab.endswith("Q4"):
        ax.axvspan(i - 0.4, i + 0.4, color="#f1f0ed", zorder=0)
for u in ["음식점", "카페"]:
    ax.plot(x, q[u], color=C[u], lw=2, marker="o", ms=5, label=u)
ax.set_xticks(x); ax.set_xticklabels(q.index, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(pct1)
ax.set_ylim(0, None); ax.set_xlim(-0.5, len(q) - 0.2)
ax.set_ylabel("폐업률 (폐업수 합 ÷ 점포수 합)")
ax.set_title("매년 4분기에 폐업률이 뛴다", loc="left")
ax.legend(loc="lower right", frameon=False, ncol=2)
ax.text(0, -0.3, "※ 회색 = 4분기. x축은 폐업이 일어난 분기(예측대상분기)", transform=ax.transAxes,
        fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(f"{out}/fig2_분기별_폐업률.png", dpi=200); plt.close(fig)

# =========================================================
# 그림 3. 다음분기 폐업률과 순위상관이 큰 변수 (점포 30개 이상 칸)
# =========================================================
big = d[d["읍면동업태_영업점포수"] >= 30]
excl = {"기준연도", "다음분기_폐업수", "다음분기_폐업률", "업태구분_2진수"}
num = [c for c in d.select_dtypes("number").columns if c not in excl]
r = pd.Series({c: spearmanr(big[c], big["다음분기_폐업률"], nan_policy="omit")[0] for c in num})
top = r.reindex(r.abs().sort_values(ascending=False).index).head(10)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.barh(top.index, top.values, color=[C["음식점"] if v > 0 else NEG for v in top.values], height=0.6)
for i, v in enumerate(top.values):
    ax.text(v + (0.01 if v > 0 else -0.01), i, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9, color=INK)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(-0.48, 0.48); ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=9)
ax.set_xlabel("Spearman 순위상관  (파랑: 클수록 폐업↑ / 주황: 클수록 폐업↓)", fontsize=9.5)
ax.set_title("폐업률과 가장 관련 큰 변수 10개", loc="left")
ax.text(0, -0.2, f"※ 점포 30개 이상 칸만 사용(n={len(big):,}), 점포수 효과를 줄이기 위해",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(f"{out}/fig3_변수_순위상관.png", dpi=200); plt.close(fig)

# =========================================================
# 그림 4. 핵심 변수 5분위별 폐업률
# =========================================================
vars4 = [("평균업력개월", "평균 업력이 짧은 동네일수록"),
         ("읍면동업태_폐업률_4Q평균", "최근 1년 폐업률이 높았던 동네일수록")]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.2), sharey=True)
for ax, (c, t) in zip(axes, vars4):
    qt = pd.qcut(d[c].rank(method="first"), 5, labels=["1\n(낮음)", "2", "3", "4", "5\n(높음)"])
    v = d.groupby(qt, observed=True).apply(rate)
    ax.bar(v.index.astype(str), v.values, color=C["음식점"], width=0.6)
    for i, y in enumerate(v.values):
        ax.text(i, y + 0.0007, f"{y:.1%}", ha="center", fontsize=9, color=INK)
    ax.set_title(f"{t}\n다음분기 폐업률↑", loc="left", fontsize=11)
    ax.set_xlabel(f"{c} 5분위", fontsize=9)
    ax.grid(axis="x", visible=False); ax.yaxis.set_major_formatter(pct1)
axes[0].set_ylabel("다음분기 폐업률")
fig.text(0.01, 0.01, "※ 5분위 = 값 크기순으로 칸을 5등분. 폐업률은 폐업수 합 ÷ 점포수 합", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{out}/fig4_핵심변수_5분위.png", dpi=200); plt.close(fig)
print("저장:", out)
