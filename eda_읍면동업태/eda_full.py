"""
인천 분기×읍면동×업태 데이터 전체 EDA (표 + 그림)
실행: python eda_full.py <데이터.csv> <저장폴더>
결과: <저장폴더>/tables/*.csv, <저장폴더>/figures/fig5~8*.png
(그림 1~4는 eda_figures.py)
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import spearmanr

src = sys.argv[1] if len(sys.argv) > 1 else "인천_분기x읍면동x업태_다음분기폐업률.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "."
T, F = f"{out}/tables", f"{out}/figures"
os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)

for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "WenQuanYi Zen Hei"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"음식점": "#2a78d6", "카페": "#eb6834"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.edgecolor": GRID, "axes.axisbelow": True,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "font.size": 10})

d = pd.read_csv(src, encoding="utf-8-sig")
K = ["군구명_개편후", "읍면동_개편후", "업태구분명"]
rate = lambda g: g["다음분기_폐업수"].sum() / g["읍면동업태_영업점포수"].sum()
save = lambda df, n: df.to_csv(f"{T}/{n}.csv", encoding="utf-8-sig")

# ---------------------------------------------------------
# 1. 읍면동_ vs 읍면동업태_ 짝 변수 상관
# ---------------------------------------------------------
pairs = ["영업점포수", "분기개업수", "분기폐업수", "개업률", "폐업률", "점포수증감률",
         "폐업률_lag1", "폐업률_lag4", "폐업률_4Q평균", "폐업률_전분기대비증감"]
rows = []
for p in pairs:
    a, b = f"읍면동_{p}", f"읍면동업태_{p}"
    r = {"변수": p}
    for lab, sub in [("전체", d), ("음식점", d[d.업태구분명 == "음식점"]), ("카페", d[d.업태구분명 == "카페"])]:
        s = sub[[a, b]].dropna()
        r[f"{lab}_Pearson"] = s[a].corr(s[b])
        r[f"{lab}_Spearman"] = spearmanr(s[a], s[b])[0]
    rows.append(r)
pair_df = pd.DataFrame(rows).set_index("변수").round(3)
save(pair_df, "01_읍면동_vs_읍면동업태_상관")

fig, ax = plt.subplots(figsize=(8.5, 4.6))
y = np.arange(len(pairs))
ax.barh(y + 0.2, pair_df["음식점_Spearman"], height=0.38, color=C["음식점"], label="음식점 칸")
ax.barh(y - 0.2, pair_df["카페_Spearman"], height=0.38, color=C["카페"], label="카페 칸")
for i, p in enumerate(pairs):
    ax.text(pair_df.loc[p, "음식점_Spearman"] + 0.01, i + 0.2, f"{pair_df.loc[p, '음식점_Spearman']:.2f}", va="center", fontsize=8.5)
    ax.text(pair_df.loc[p, "카페_Spearman"] + 0.01, i - 0.2, f"{pair_df.loc[p, '카페_Spearman']:.2f}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(pairs); ax.invert_yaxis()
ax.set_xlim(0, 1.08); ax.grid(axis="y", visible=False)
ax.set_xlabel("Spearman 상관: 읍면동_X ↔ 읍면동업태_X")
ax.set_title("음식점 칸은 읍면동 전체 값과 거의 같고, 카페 칸은 다르다", loc="left", fontsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/fig5_읍면동_vs_읍면동업태_상관.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------
# 2. 변수 간 상관 행렬 (Spearman) + 높은 쌍 목록
# ---------------------------------------------------------
drop = {"기준연도", "다음분기_폐업수"}
num = [c for c in d.select_dtypes("number").columns if c not in drop]
corr = d[num].corr(method="spearman").round(3)
save(corr, "02_변수간_상관행렬_Spearman")
save(d[num].corr().round(3), "02b_변수간_상관행렬_Pearson")
hi = (corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("Spearman").reset_index()
      .rename(columns={"level_0": "변수1", "level_1": "변수2"}))
hi = hi[(hi.Spearman.abs() >= 0.8) & (hi.변수1 != "다음분기_폐업률") & (hi.변수2 != "다음분기_폐업률")]
hi = hi.sort_values("Spearman", key=abs, ascending=False).reset_index(drop=True)
save(hi, "03_상관_0.8이상_변수쌍")

fig, ax = plt.subplots(figsize=(11, 9.5))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3f0", "#2a78d6"])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=90, fontsize=7)
ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=7)
ax.grid(False)
for i in range(len(num)):
    for j in range(len(num)):
        v = corr.values[i, j]
        if abs(v) >= 0.8 and i != j:
            ax.text(j, i, f"{v:.1f}".replace("0.", "."), ha="center", va="center", fontsize=5, color="white")
cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("Spearman")
ax.set_title("변수 간 Spearman 상관 (|ρ|≥0.8 칸에 값 표시)", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(f"{F}/fig6_변수간_상관_히트맵.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------
# 3. 변수와 타깃의 상관 (전체 / 점포30+ / 업태별)
# ---------------------------------------------------------
feat = [c for c in num if c not in {"다음분기_폐업률"}]
big = d[d.읍면동업태_영업점포수 >= 30]
tgt = pd.DataFrame({
    "전체_Spearman": {c: spearmanr(d[c], d.다음분기_폐업률, nan_policy="omit")[0] for c in feat},
    "점포30+_Spearman": {c: spearmanr(big[c], big.다음분기_폐업률, nan_policy="omit")[0] for c in feat},
    "음식점30+_Spearman": {c: spearmanr(*big[big.업태구분명 == "음식점"][[c, "다음분기_폐업률"]].T.values, nan_policy="omit")[0] for c in feat},
    "카페30+_Spearman": {c: spearmanr(*big[big.업태구분명 == "카페"][[c, "다음분기_폐업률"]].T.values, nan_policy="omit")[0] for c in feat},
}).round(3).sort_values("점포30+_Spearman", key=abs, ascending=False)
save(tgt, "04_변수_타깃_상관")

# ---------------------------------------------------------
# 4. 시간·지역
# ---------------------------------------------------------
q = d.groupby(["예측대상분기", "업태구분명"]).apply(rate).unstack().round(4)
save(q, "05_분기별_폐업률")
qn = d.groupby("분기번호").apply(rate).round(4); save(qn.to_frame("폐업률"), "05b_기준분기번호별_폐업률")
gg = d.groupby(["군구명_개편후", "업태구분명"]).apply(rate).unstack().round(4)
gg["점포수합"] = d.groupby("군구명_개편후").읍면동업태_영업점포수.sum()
save(gg.sort_values("음식점", ascending=False), "06_군구별_폐업률")
emd = (big.groupby(K).apply(lambda g: pd.Series({"분기수": len(g), "평균점포수": g.읍면동업태_영업점포수.mean(), "폐업률": rate(g)}))
       .reset_index().query("분기수>=10").sort_values("폐업률", ascending=False))
save(pd.concat([emd.head(10), emd.tail(10)]).round(4), "07_읍면동업태_폐업률_상하위10")

order = gg.sort_values("음식점", ascending=False).index
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(order))
ax.bar(x - 0.2, gg.loc[order, "음식점"], 0.38, color=C["음식점"], label="음식점")
ax.bar(x + 0.2, gg.loc[order, "카페"], 0.38, color=C["카페"], label="카페")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=0, fontsize=8.5)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
ax.grid(axis="x", visible=False); ax.legend(frameon=False)
ax.set_ylabel("다음분기 폐업률 (폐업수 합 ÷ 점포수 합)")
ax.set_title("군구별 폐업률: 영종·서해구가 높고 강화군·제물포구가 낮다", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{F}/fig7_군구별_폐업률.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------
# 5. 학습(~2025) vs 테스트(2026) 분포 비교
# ---------------------------------------------------------
d["구간"] = np.where(d.기준연도 <= 2025, "학습 2023~2025", "테스트 2026")
drift = d.groupby(["구간", "업태구분명"]).apply(lambda g: pd.Series({
    "폐업률": rate(g), "0인칸비율": (g.다음분기_폐업률 == 0).mean(),
    "평균업력개월(가중)": np.average(g.평균업력개월, weights=g.읍면동업태_영업점포수),
    "점포수합": g.읍면동업태_영업점포수.sum()})).round(4)
save(drift, "08_학습_테스트_분포비교")

# ---------------------------------------------------------
# 6. 타깃 정의 점검: 다음분기_폐업수 vs 다음 행의 분기폐업수
# ---------------------------------------------------------
s = d.sort_values(K + ["시간순번"]).copy()
s["다음행_분기폐업수"] = s.groupby(K).읍면동업태_분기폐업수.shift(-1)
m = s.다음행_분기폐업수.notna()
tchk = pd.Series({
    "비교 가능 행": int(m.sum()),
    "일치 행 비율": float((s.다음행_분기폐업수 == s.다음분기_폐업수)[m].mean()),
    "타깃 폐업수 합": int(s.loc[m, "다음분기_폐업수"].sum()),
    "다음 행 분기폐업수 합": int(s.loc[m, "다음행_분기폐업수"].sum()),
    "타깃 > 다음행 인 행": int((s.다음분기_폐업수 > s.다음행_분기폐업수)[m].sum()),
})
save(tchk.to_frame("값"), "09_타깃정의_점검")

# 점포수 구간별 타깃
b = pd.cut(d.읍면동업태_영업점포수, [0, 9, 29, 99, 1e9], labels=["1~9", "10~29", "30~99", "100+"])
size = d.groupby([b, "업태구분명"], observed=True).agg(칸수=("다음분기_폐업률", "size"),
        폐업률0비율=("다음분기_폐업률", lambda s: (s == 0).mean()),
        폐업률표준편차=("다음분기_폐업률", "std"), 점포수합=("읍면동업태_영업점포수", "sum")).round(3)
save(size, "10_점포수구간별_타깃")
print(pair_df, hi.head(30), tgt.head(12), drift, tchk, size, sep="\n\n")
