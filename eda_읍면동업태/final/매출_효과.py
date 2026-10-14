"""
총매출(NICE) 변수를 M1에 넣으면 성능이 달라지는지 (팀원 매출 결합본 새로해보는2_총매출추가.csv 사용)
매출은 2024Q1부터라 같은 행(2024~)으로 M1 vs M1+매출 비교. 팀원 M1 설정, 시드 5개 평균.
실행: python 매출_효과.py <새로해보는2_총매출추가.csv>
"""
import sys, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
K = ["군구명_개편후", "읍면동_개편후", "업태구분명"]; T, N = "다음분기_폐업률", "읍면동업태_영업점포수"
d = d.sort_values(K + ["시간순번"]); g = d.groupby(K).NICE_총매출_만원
d["총매출_log"] = np.log1p(d.NICE_총매출_만원)
d["점포당매출_log"] = np.log1p(d.NICE_총매출_만원 / d[N])
d["매출_전분기대비"] = d.NICE_총매출_만원 / g.shift(1) - 1
d["매출_전년대비"] = d.NICE_총매출_만원 / g.shift(4) - 1
d["점포당매출_전년대비"] = d.점포당매출_log - d.groupby(K).점포당매출_log.shift(4)
SALES = ["총매출_log", "점포당매출_log", "매출_전분기대비", "매출_전년대비", "점포당매출_전년대비"]
BASE = ["분기번호","시간순번","업태구분_2진수","읍면동업태_영업점포수","평균업력개월","업력1년미만_점포비중",
        "평균시설규모","중앙시설규모","소규모점포비중","다중이용업소비중","법정동_실제총인구",
        "법정동_인구증감률_QoQ","법정동_인구증감률_YoY","법정동_인구천명당_전체점포수","법정동_인구천명당_동일업태점포수",
        "읍면동업태_분기개업수","읍면동업태_분기폐업수","읍면동업태_점포비중","읍면동업태_직전분기말점포수",
        "읍면동업태_개업률","읍면동업태_폐업률","읍면동업태_점포수증감률","읍면동업태_폐업률_lag1",
        "읍면동업태_폐업률_lag4","읍면동업태_폐업률_4Q평균","읍면동업태_폐업률_전분기대비증감"]
P = dict(objective="regression_l1", learning_rate=0.05, num_leaves=15, min_child_samples=70, reg_alpha=0.1,
         reg_lambda=1.0, colsample_bytree=0.9, subsample=0.9, n_estimators=120, verbosity=-1, n_jobs=4)
rows = []
for ty in [2026, 2025]:
    tr, te = d[d.기준연도 < ty], d[d.기준연도 == ty]
    for nm, F in [("M1", BASE), ("M1+매출", BASE + SALES)]:
        p = np.mean([np.clip(lgb.LGBMRegressor(**P, random_state=s).fit(tr[F], tr[T], sample_weight=np.sqrt(tr[N]))
                             .predict(te[F]), 0, 1) for s in range(5)], axis=0)
        y, n = te[T].values, te[N].values; big = n >= 30
        rows.append({"테스트": ty, "학습": f"2024~{ty-1}", "모델": nm, "MAE": mean_absolute_error(y, p),
                     "WMAE": np.average(np.abs(y - p), weights=np.sqrt(n)), "R2": r2_score(y, p),
                     "가중R2": 1 - np.sum(n * (y - p) ** 2) / np.sum(n * (y - np.average(y, weights=n)) ** 2),
                     "Spearman_점포30이상": spearmanr(y[big], p[big])[0]})
res = pd.DataFrame(rows).round(4); res.to_csv("tables/20_매출_모델효과.csv", index=False, encoding="utf-8-sig"); print(res.to_string(index=False))
d.to_csv("tables/_매출변수.csv", index=False, encoding="utf-8-sig")
