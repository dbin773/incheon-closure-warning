"""
최종 EDA 노션용 추가 그림 (f08~f13)
실행: python eda_그림.py <새로해보는2.csv> <총매출추가.csv> <원본 통합 raw.csv>
"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.stats import spearmanr
for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "WenQuanYi Zen Hei"]:
    if any(f.name == name for f in fm.fontManager.ttflist): plt.rcParams["font.family"] = name; break
INK, INK2, GRID, MUTE = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
C = {"음식점": "#2a78d6", "카페": "#eb6834"}
SEQ = ["#9ec5f4", "#5a9be6", "#2a78d6", "#1b56a3"]   # 같은 파랑 계열 연→진
plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "savefig.dpi": 160, "savefig.bbox": "tight"})
pct1 = matplotlib.ticker.PercentFormatter(1.0, decimals=1)
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); s = pd.read_csv(sys.argv[2], encoding="utf-8-sig")
T, N, K = "다음분기_폐업률", "읍면동업태_영업점포수", ["군구명_개편후", "읍면동_개편후", "업태구분명"]
tr = d[d.기준연도 <= 2025].copy()
rate = lambda g: g.다음분기_폐업수.sum() / g[N].sum()
os.makedirs("figures", exist_ok=True)

# f08 VIF: 읍면동업태_만 vs 읍면동_까지 (음식점 칸)
v = pd.read_csv("tables/14_VIF.csv", index_col=0, encoding="utf-8-sig")
v = v[v.index.str.startswith("읍면동업태_")][["음식점칸_읍면동업태_만", "음식점칸_읍면동_까지"]].sort_values("음식점칸_읍면동_까지")
fig, ax = plt.subplots(figsize=(7.2, 4.2)); y = np.arange(len(v)); h = 0.38
ax.barh(y + h / 2, v.iloc[:, 0], h, color=C["음식점"], label="읍면동업태_ 변수만")
ax.barh(y - h / 2, v.iloc[:, 1], h, color=MUTE, label="읍면동_ 변수까지 넣으면")
ax.set_xscale("log"); ax.axvline(10, color=INK2, lw=1, ls="--"); ax.text(10.5, len(v) - 0.6, "VIF 10 (위험 기준)", color=INK2, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([i.replace("읍면동업태_", "") for i in v.index])
for i, (a_, b_) in enumerate(v.values): ax.text(b_ * 1.15, i - h / 2, f"{b_:,.0f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("VIF (로그 눈금)"); ax.set_title("읍면동_까지 넣으면 다중공선성이 폭발한다 (음식점 칸)", loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=9); ax.grid(axis="y", visible=False)
fig.savefig("figures/f08_VIF_비교.png"); plt.close()

# f09 연도별 안정성: (a) 평균업력 5분위 (b) 예측분기별 폐업률
a5 = pd.read_csv("tables/17_연도별_평균업력5분위_폐업률.csv", index_col=0, encoding="utf-8-sig")
q4 = pd.read_csv("tables/16_연도별_예측분기_폐업률.csv", index_col=0, encoding="utf-8-sig")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for i, (yr, r) in enumerate(a5.iterrows()):
    axs[0].plot(range(1, 6), r.values / 100, marker="o", ms=6, lw=2, color=SEQ[i + 1], label=str(yr))
axs[0].set_xticks(range(1, 6)); axs[0].set_xticklabels(["1\n(업력 짧음)", "2", "3", "4", "5\n(업력 김)"])
axs[0].yaxis.set_major_formatter(pct1); axs[0].set_title("평균 업력이 짧을수록 폐업률↑ (매년 같음)", loc="left", fontsize=12)
axs[0].set_ylabel("다음분기 폐업률"); axs[0].legend(frameon=False, fontsize=9, title="기준연도", title_fontsize=9)
for i, (yr, r) in enumerate(q4.iterrows()):
    axs[1].plot(range(1, 5), r.values / 100, marker="o", ms=6, lw=2, color=SEQ[i + 1], label=str(yr))
axs[1].set_xticks(range(1, 5)); axs[1].set_xticklabels(["1분기", "2분기", "3분기", "4분기"]); axs[1].set_xlabel("예측대상 분기")
axs[1].yaxis.set_major_formatter(pct1); axs[1].set_title("4분기 폐업 급증 (매년 같음)", loc="left", fontsize=12)
axs[1].legend(frameon=False, fontsize=9, title="기준연도", title_fontsize=9)
fig.tight_layout(); fig.savefig("figures/f09_연도별_안정성.png"); plt.close()

# f10 이상치: 인천 전체 일별 폐업 수
raw = pd.read_csv(sys.argv[3], encoding="utf-8-sig", low_memory=False, usecols=["관리번호", "폐업일자", "인허가일자", "지번주소"])
raw["폐업일자"] = pd.to_datetime(raw.폐업일자, errors="coerce")
daily = raw[raw.폐업일자.between("2023-04-01", "2026-09-30")].groupby("폐업일자").size()
daily = daily.reindex(pd.date_range("2023-04-01", "2026-09-30"), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(daily.index, daily.values, lw=0.8, color=C["음식점"])
ax.set_ylabel("하루 폐업 수 (인천 전체)"); ax.set_title("하루에 몰리는 폐업: 축제 임시 영업과 구청 일괄 정리", loc="left")
notes = [("2023-09-02", "송도 축제 부스", (-60, 14)), ("2024-08-31", "송도 달빛축제공원\n축제 부스 일괄 폐업", (-150, 6)),
         ("2024-11-07", "주안·용현·숭의동\n구청 일괄 정리", (8, 30)), ("2025-12-31", "연말 정리 (매년 12-31)", (8, 4))]
for dt, lab, off in notes:
    x = pd.Timestamp(dt); yv = daily.loc[x]
    ax.annotate(lab, (x, yv), xytext=off, textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.plot([x], [yv], "o", ms=6, color=C["카페"])
ax.set_ylim(0, 140)
ax.text(0.01, 0.92, f"평소 하루 중앙값 {int(daily.median())}건", transform=ax.transAxes, fontsize=9, color=INK2)
fig.savefig("figures/f10_일별폐업.png"); plt.close()

# f11 업태 분류: 음식점으로 들어간 카페 이름 가게의 원본 업태
mv = pd.read_csv("재분류_가게목록.csv", encoding="utf-8-sig").원본업태.value_counts()
mv = pd.concat([mv.head(6), pd.Series({"그 외": mv.iloc[6:].sum()})]).sort_values()
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].barh(mv.index, mv.values, color=C["카페"], height=0.6)
for i, v_ in enumerate(mv.values): axs[0].text(v_ + 15, i, f"{v_:,}", va="center", fontsize=9, color=INK2)
axs[0].set_title("음식점으로 분류된 '카페 이름' 가게 1,419곳의 원본 업태", loc="left", fontsize=12); axs[0].grid(axis="y", visible=False)
axs[1].bar(["원래 분류", "이름으로 보정"], [6758, 8177], color=[MUTE, C["카페"]], width=0.55)
for i, v_ in enumerate([6758, 8177]): axs[1].text(i, v_ + 120, f"{v_:,}곳", ha="center", fontsize=10, color=INK)
axs[1].set_title("카페 가게 수 (+21%)", loc="left", fontsize=12); axs[1].set_ylim(0, 9500); axs[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("figures/f11_업태분류.png"); plt.close()

# f12 매출: (a) 총매출 vs 점포수 (b) 점포당 매출 전년대비 5분위 vs 폐업률
m = pd.read_csv("tables/_매출변수.csv", encoding="utf-8-sig")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for u in ["음식점", "카페"]:
    x = m[m.업태구분명 == u]
    axs[0].scatter(x[N], x.NICE_총매출_만원, s=8, alpha=0.35, color=C[u], edgecolors="none", label=f"{u} (ρ={spearmanr(x[N], x.NICE_총매출_만원)[0]:.2f})")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("영업점포수"); axs[0].set_ylabel("총매출 (만원)")
axs[0].set_title("총매출은 거의 '동네 크기'다", loc="left", fontsize=12); axs[0].legend(frameon=False, fontsize=9, markerscale=2)
b = m[(m[N] >= 30) & m.점포당매출_전년대비.notna() & (m.기준연도 <= 2025)].copy()
b["q"] = pd.qcut(b.점포당매출_전년대비, 5, labels=False) + 1
for u, mk in [("음식점", "o"), ("카페", "s")]:
    r = b[b.업태구분명 == u].groupby("q").apply(rate)
    axs[1].plot(r.index, r.values, marker=mk, ms=6, lw=2, color=C[u], label=u)
axs[1].set_xticks(range(1, 6)); axs[1].set_xticklabels(["1\n(감소)", "2", "3", "4", "5\n(증가)"]); axs[1].set_xlabel("점포당 매출 전년 대비 (5분위)")
axs[1].yaxis.set_major_formatter(pct1); axs[1].set_ylabel("다음분기 폐업률")
axs[1].set_title("점포당 매출 변화와 폐업률: 관계 약함", loc="left", fontsize=12); axs[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/f12_매출.png"); plt.close()

# f13 군구 집계 안정성: 분기별 폐업률 범위
gq = tr.groupby(["군구명_개편후", "기준분기"]).apply(rate).rename("r").reset_index()
st = gq.groupby("군구명_개편후").r.agg(["min", "max", "mean"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(7.5, 4.2)); y = np.arange(len(st))
ax.hlines(y, st["min"], st["max"], color=MUTE, lw=4)
ax.plot(st["mean"], y, "o", ms=8, color=C["음식점"], label="평균")
ax.set_yticks(y); ax.set_yticklabels(st.index); ax.xaxis.set_major_formatter(pct1)
ax.set_xlabel("분기별 다음분기 폐업률 (회색 막대 = 최저~최고)")
ax.set_title("군구별 폐업률과 분기별 흔들림 (2023~2025)", loc="left"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig("figures/f13_군구안정성.png"); plt.close()
print("done", os.listdir("figures"))
