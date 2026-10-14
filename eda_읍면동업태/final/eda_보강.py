"""
최종 EDA 보강: 읍면동_ 변수를 빼고 읍면동업태_ 변수만 쓰는 근거 + 시간 안정성·품질·집계 안정성·VIF
실행: python eda_보강.py <새로해보는2 칸데이터.csv>
결과: tables/ 폴더의 csv들 (학습 구간 2023~2025만 사용, 테스트 2026은 분석에 쓰지 않음)
"""
import sys, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
T, N = "다음분기_폐업률", "읍면동업태_영업점포수"
tr = d[d.기준연도 <= 2025].copy()
PAIRS = ["영업점포수", "분기개업수", "분기폐업수", "개업률", "폐업률", "점포수증감률",
         "폐업률_lag1", "폐업률_lag4", "폐업률_4Q평균", "폐업률_전분기대비증감"]
out = {}

# 1) 항등식: 읍면동_ 개수 변수 = 음식점 칸 + 카페 칸
k = ["기준분기", "군구명_개편후", "읍면동_개편후"]
s = d.groupby(k)[[N, "읍면동업태_분기개업수", "읍면동업태_분기폐업수"]].sum()
f = d.drop_duplicates(k).set_index(k)[["읍면동_영업점포수", "읍면동_분기개업수", "읍면동_분기폐업수"]]
out["11_항등식"] = pd.DataFrame({"변수": ["영업점포수", "분기개업수", "분기폐업수"],
    "읍면동_ = 음식점+카페 일치율": [(s[N] == f.읍면동_영업점포수).mean(), (s.읍면동업태_분기개업수 == f.읍면동_분기개업수).mean(),
                               (s.읍면동업태_분기폐업수 == f.읍면동_분기폐업수).mean()]}).round(3)
# 음식점 비중 분포
share = d[d.업태구분명 == "음식점"].읍면동업태_점포비중
out["11b_음식점비중"] = share.describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_frame("음식점 칸의 점포비중")

# 2) 업태 × 연도별 읍면동_ vs 읍면동업태_ Spearman (안정성)
rows = []
for v in PAIRS:
    for u in ["음식점", "카페"]:
        r = {"변수": v, "업태": u}
        for y in [2023, 2024, 2025]:
            x = tr[(tr.업태구분명 == u) & (tr.기준연도 == y)][["읍면동_" + v, "읍면동업태_" + v]].dropna()
            r[str(y)] = spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]
        rows.append(r)
out["12_업태연도별_쌍상관"] = pd.DataFrame(rows).round(3)

# 3) 추가 정보량: 읍면동업태_를 통제한 뒤 읍면동_가 타깃과 갖는 부분 순위상관 (점포 30개 이상 칸)
def partial_spearman(x, y, z):
    rx, ry, rz = [pd.Series(a).rank().values for a in (x, y, z)]
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]
rows = []
for v in ["폐업률", "폐업률_lag1", "폐업률_lag4", "폐업률_4Q평균", "개업률", "점포수증감률"]:
    for u in ["음식점", "카페"]:
        x = tr[(tr.업태구분명 == u) & (tr[N] >= 30)][["읍면동_" + v, "읍면동업태_" + v, T]].dropna()
        rows.append({"변수": v, "업태": u, "칸수": len(x),
                     "읍면동업태_ vs 타깃": spearmanr(x.iloc[:, 1], x[T])[0],
                     "읍면동_ vs 타깃": spearmanr(x.iloc[:, 0], x[T])[0],
                     "읍면동업태_ 통제 후 읍면동_ (부분상관)": partial_spearman(x.iloc[:, 0], x[T], x.iloc[:, 1])})
out["13_추가정보량_부분상관"] = pd.DataFrame(rows).round(3)

# 4) VIF: 읍면동업태_만 vs 읍면동_까지 (연속형, 학습 구간, 결측행 제외)
def vif(X):
    X = (X - X.mean()) / X.std()
    res = {}
    for c in X.columns:
        y = X[c].values; A = np.c_[np.ones(len(X)), X.drop(columns=c).values]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None); r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res[c] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return pd.Series(res)
# (가) 정확한 계산식 관계 확인: 점포수증감률 = 개업률 − 폐업률, 전분기대비증감 = 폐업률 − lag1
ide = []
for pre in ["읍면동업태_", "읍면동_"]:
    x = tr.dropna(subset=[pre + "점포수증감률", pre + "폐업률_전분기대비증감", pre + "폐업률_lag1"])
    ide.append({"묶음": pre, "점포수증감률 = 개업률 − 폐업률": np.isclose(x[pre + "점포수증감률"], x[pre + "개업률"] - x[pre + "폐업률"]).mean(),
                "전분기대비증감 = 폐업률 − lag1": np.isclose(x[pre + "폐업률_전분기대비증감"], x[pre + "폐업률"] - x[pre + "폐업률_lag1"]).mean()})
out["14a_계산식관계"] = pd.DataFrame(ide).round(3)
# (나) 계산식으로 겹치는 2개를 뺀 뒤 VIF 비교
SUB = [v for v in PAIRS if v not in ("점포수증감률", "폐업률_전분기대비증감")]
UPT = ["읍면동업태_" + v for v in SUB]; DONG = ["읍면동_" + v for v in SUB]
X = tr[UPT + DONG].dropna()
Xr = tr[tr.업태구분명 == "음식점"][UPT + DONG].dropna()
out["14_VIF"] = pd.DataFrame({"전체_읍면동업태_만": vif(X[UPT]), "전체_읍면동_까지": vif(X[UPT + DONG]),
                              "음식점칸_읍면동업태_만": vif(Xr[UPT]), "음식점칸_읍면동_까지": vif(Xr[UPT + DONG])}).round(1)

# 5) 연도별 안정성: 핵심 변수-타깃 Spearman (점포 30개 이상)
rows = []
for v in ["평균업력개월", "읍면동업태_폐업률_4Q평균", "업력1년미만_점포비중", "법정동_인구천명당_동일업태점포수",
          "읍면동업태_폐업률", "읍면동_폐업률_4Q평균", "평균시설규모", "다중이용업소비중"]:
    r = {"변수": v}
    for y in [2023, 2024, 2025]:
        x = tr[(tr.기준연도 == y) & (tr[N] >= 30)][[v, T]].dropna(); r[str(y)] = spearmanr(x[v], x[T])[0]
    rows.append(r)
out["15_연도별_변수타깃상관"] = pd.DataFrame(rows).round(3)
# 4분기 계절성 연도별
tr["예측분기"] = tr.예측대상분기.str[-1].astype(int)
g = tr.groupby(["기준연도", "예측분기"]).apply(lambda x: x.다음분기_폐업수.sum() / x[N].sum()).unstack()
out["16_연도별_예측분기_폐업률"] = (g * 100).round(2)
# 평균업력 5분위 연도별
tr["업력5분위"] = tr.groupby("기준연도").평균업력개월.transform(lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5]))
g = tr[tr[N] >= 30].groupby(["기준연도", "업력5분위"]).apply(lambda x: x.다음분기_폐업수.sum() / x[N].sum()).unstack()
out["17_연도별_평균업력5분위_폐업률"] = (g * 100).round(2)

# 6) 품질: 결측·무한대·범위
num = d.select_dtypes("number")
q = pd.DataFrame({"결측": num.isna().sum(), "무한대": np.isinf(num).sum(), "최소": num.min(), "최대": num.max()})
out["18_품질"] = q[(q.결측 > 0) | (q.무한대 > 0)]

# 7) 군구 집계 안정성 (학습 구간)
gq = tr.groupby(["군구명_개편후", "기준분기"]).agg(점포=(N, "sum"), 폐업=("다음분기_폐업수", "sum"))
gq["폐업률"] = gq.폐업 / gq.점포
cafe = tr[tr.업태구분명 == "카페"].groupby("군구명_개편후")[N].sum() / tr.groupby("군구명_개편후")[N].sum()
small = tr.assign(작은칸=tr[N] < 30).groupby("군구명_개편후").작은칸.mean()
gg = gq.groupby(level=0).agg(분기평균점포=("점포", "mean"), 최소점포=("점포", "min"),
                             평균폐업률=("폐업률", "mean"), 폐업률표준편차=("폐업률", "std"))
gg["카페비중"] = cafe; gg["점포30미만칸비중"] = small
out["19_군구집계안정성"] = gg.sort_values("평균폐업률", ascending=False).round(4)

for name, t in out.items():
    t.to_csv(f"tables/{name}.csv", encoding="utf-8-sig")
    print(f"\n== {name}\n{t.to_string()}")
