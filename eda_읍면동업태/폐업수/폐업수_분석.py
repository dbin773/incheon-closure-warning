"""
최종 타깃(다음분기_폐업수) 기준 EDA 수치·모델 비교
실행: python 폐업수_분석.py <새로해보는3 csv> <업태수정 칸데이터 csv>
모델 = 팀 최종 노트북(lightGBM_폐업수예측.ipynb) 설정: LightGBM poisson, 26변수, 학습 ~2025 / 테스트 2026
"""
import sys, warnings, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); fx = pd.read_csv(sys.argv[2], encoding="utf-8-sig")
Y, N = "다음분기_폐업수", "읍면동업태_영업점포수"
F = ["분기번호","업태구분_2진수",N,"평균업력개월","업력1년미만_점포비중","평균시설규모","중앙시설규모","소규모점포비중",
     "다중이용업소비중","법정동_실제총인구","법정동_인구증감률_QoQ","법정동_인구증감률_YoY","법정동_인구천명당_전체점포수",
     "법정동_인구천명당_동일업태점포수","읍면동업태_분기개업수","읍면동업태_분기폐업수","읍면동업태_점포비중",
     "읍면동업태_직전분기말점포수","읍면동업태_개업률","읍면동업태_폐업률","읍면동업태_점포수증감률","읍면동업태_폐업률_lag1",
     "읍면동업태_폐업률_lag4","읍면동업태_폐업률_4Q평균","읍면동업태_폐업률_전분기대비증감","NICE_총매출_만원"]
DONG = [c for c in d.columns if c.startswith("읍면동_") and c != "읍면동_개편후"]
tr = d[d.기준연도 <= 2025]
out = {}

# 1) 분포
out["0건 비중(전체)"] = (d[Y] == 0).mean(); out["중앙값"] = d[Y].median(); out["상위10%"] = d[Y].quantile(.9); out["최대"] = d[Y].max()
out["상위10% 칸이 차지하는 폐업 비중"] = d[Y][d[Y] >= d[Y].quantile(.9)].sum() / d[Y].sum()
out["폐업수~점포수 Pearson(학습)"] = pearsonr(tr[N], tr[Y])[0]; out["폐업수~점포수 Spearman(학습)"] = spearmanr(tr[N], tr[Y])[0]
bins = [0, 10, 30, 100, 300, 10**6]; lab = ["1~9", "10~29", "30~99", "100~299", "300+"]
b = d.assign(구간=pd.cut(d[N], bins, right=False, labels=lab)).groupby("구간", observed=True).agg(
    칸=(Y, "size"), 평균폐업수=(Y, "mean"), 영건비중=(Y, lambda s: (s == 0).mean()), 폐업합=(Y, "sum"))
b["폐업합비중"] = b.폐업합 / b.폐업합.sum(); b.to_csv("tables/c1_점포구간별_폐업수.csv", encoding="utf-8-sig"); print(b.round(3))

# 2) 분기별 폐업수 합
q = d.groupby(["예측대상분기", "업태구분명"])[Y].sum().unstack(); q.to_csv("tables/c2_분기별_폐업수합.csv", encoding="utf-8-sig"); print(q)
# 군구
g = tr.groupby("군구명_개편후").agg(분기평균폐업수=(Y, lambda s: s.sum() / tr.예측대상분기.nunique()), 점포=(N, "sum"), 폐업=(Y, "sum"))
g["폐업률"] = g.폐업 / g.점포; g = g.sort_values("분기평균폐업수", ascending=False); g.to_csv("tables/c3_군구별_폐업수.csv", encoding="utf-8-sig"); print(g.round(3))

# 3) 변수-타깃 상관: 폐업수 vs 폐업률(점포수 효과 뺀 것)
rows = []
for c in F:
    s = tr[list(dict.fromkeys([c, Y, N, "다음분기_폐업률"]))].dropna()
    rows.append((c, spearmanr(s[c], s[Y])[0], spearmanr(s[c], s[N])[0], spearmanr(s[c], s["다음분기_폐업률"])[0]))
cr = pd.DataFrame(rows, columns=["변수", "폐업수와", "점포수와", "폐업률과"]).set_index("변수").sort_values("폐업수와", key=lambda s: s.abs(), ascending=False)
cr.to_csv("tables/c4_변수_폐업수상관.csv", encoding="utf-8-sig"); print(cr.round(2))

# 4) 모델
P = dict(objective="poisson", learning_rate=0.05, num_leaves=15, min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
         colsample_bytree=0.9, subsample=0.9, n_estimators=120, verbosity=-1, force_col_wise=True, n_jobs=4)
def met(y, p, n):
    m = n >= 30
    return dict(MAE=mean_absolute_error(y, p), R2=r2_score(y, p), Spearman=spearmanr(y, p)[0],
                율Spearman30=spearmanr((y / n)[m], (p / n)[m])[0])
def run(df, feats, test_year, seeds=range(5)):
    a, t = df[df.기준연도 < test_year], df[df.기준연도 == test_year]
    r = [met(t[Y].values, lgb.LGBMRegressor(**P, random_state=s).fit(a[feats], a[Y]).predict(t[feats]), t[N].values) for s in seeds]
    return pd.DataFrame(r).mean()
def naive(df, test_year):
    a, t = df[df.기준연도 < test_year], df[df.기준연도 == test_year]
    r = a.groupby("업태구분명")[Y].sum() / a.groupby("업태구분명")[N].sum()
    return pd.Series(met(t[Y].values, t[N].values * t.업태구분명.map(r).values, t[N].values))
res = {}
for yr in [2026, 2025]:
    res[(yr, "점포수×업태평균 폐업률 (기준)")] = naive(d, yr)
    res[(yr, "최종 모델 (26변수)")] = run(d, F, yr)
    res[(yr, "최종 − 총매출")] = run(d, [f for f in F if f != "NICE_총매출_만원"], yr)
    res[(yr, "최종 + 읍면동_ 10개")] = run(d, F + DONG, yr)
    nf = [f for f in F if f != "NICE_총매출_만원"]
    res[(yr, "업태 원래 (매출 제외)")] = res[(yr, "최종 − 총매출")]
    res[(yr, "업태 수정 (매출 제외)")] = run(fx, nf, yr)
R = pd.DataFrame(res).T; R.to_csv("tables/c5_모델비교.csv", encoding="utf-8-sig"); print(R.round(3))

# 5) 읍면동_ 부분상관 (폐업수 기준): 읍면동업태_ 변수 + 점포수 통제 후 잔차 상관
eu = [f for f in F if f.startswith("읍면동업태_")]
s = tr[eu + DONG + [Y]].dropna(); X = np.c_[np.ones(len(s)), s[eu].values]
res_y = s[Y].values - X @ np.linalg.lstsq(X, s[Y].values, rcond=None)[0]
pc = {c: np.corrcoef(s[c].values - X @ np.linalg.lstsq(X, s[c].values, rcond=None)[0], res_y)[0, 1] for c in DONG}
pc = pd.Series(pc); pc.to_csv("tables/c6_읍면동_부분상관_폐업수.csv", encoding="utf-8-sig"); print(pc.round(3))
# 매출 상관
out["총매출~폐업수 Spearman(학습)"] = spearmanr(tr.NICE_총매출_만원, tr[Y], nan_policy="omit")[0]
out["총매출~점포수 Spearman(학습)"] = spearmanr(tr.NICE_총매출_만원, tr[N], nan_policy="omit")[0]
pd.Series(out).to_csv("tables/c0_요약수치.csv", encoding="utf-8-sig"); print(pd.Series(out).round(3))
