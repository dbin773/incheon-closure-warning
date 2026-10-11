"""LightGBM 예측 결과 CSV 만들기
노트북과 같은 LightGBM(26개 변수, 같은 분할·파라미터, Isotonic 보정)으로
테스트 기간(2026Q1~Q2) 사업체별 다음분기 폐업확률을 계산해 results/LightGBM_예측결과_2026.csv로 저장한다.
실행: python 예측결과_만들기.py "[전처리10_최종]인천_사업체-분기_패널_군구인구_YoY보완 (2).zip"
"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
FEATURES = ["업태구분_2진수","업력개월","시설규모_log","다중이용업소여부","분기번호","시간순번",
    "군구_총인구","군구_인구증감률_QoQ","군구_인구증감률_YoY","군구_인구천명당_전체점포수","군구_인구천명당_동일업태점포수",
    "군구업태_개업률","군구업태_폐업률","군구업태_폐업률_lag1","군구업태_폐업률_lag4","군구업태_폐업률_4Q평균","군구업태_공급수요격차",
    "읍면동_영업점포수","읍면동_개업률","읍면동_폐업률","읍면동_점포수증감률","읍면동_폐업률_lag1","읍면동_폐업률_lag4","읍면동_폐업률_4Q평균",
    "읍면동업태_영업점포수","읍면동업태_점포비중"]
Y="다음분기_폐업여부"; T="시간순번"
df = pd.read_csv(sys.argv[1], low_memory=False)
df = df[["관리번호","기준분기","군구명_개편후","읍면동_개편후","업태구분명"]+FEATURES+[Y]].copy()
df["다중이용업소여부"]=df["다중이용업소여부"].map({"Y":1,"N":0}).astype(float)
tr,va,te = df[df[T]<=8].copy(), df[df[T].between(9,12)].copy(), df[df[T]>=13].copy()
P=dict(objective="binary",metric="average_precision",learning_rate=0.03,num_leaves=63,min_child_samples=200,
 feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,seed=42,verbose=-1)
m=lgb.train(P,lgb.Dataset(tr[FEATURES],tr[Y]),3000,valid_sets=[lgb.Dataset(va[FEATURES],va[Y])],callbacks=[lgb.early_stopping(100,verbose=False)])
print("trees",m.best_iteration)
iso=IsotonicRegression(out_of_bounds="clip",y_min=1e-6,y_max=1-1e-6).fit(m.predict(va[FEATURES]),va[Y])
va["p"]=iso.predict(m.predict(va[FEATURES])); te["p"]=iso.predict(m.predict(te[FEATURES]))

pred=te[["관리번호","기준분기","군구명_개편후","읍면동_개편후","업태구분명","p",Y]].rename(columns={"p":"예측_폐업확률",Y:"실제_다음분기_폐업여부"})
pred["예측_폐업확률"]=pred["예측_폐업확률"].round(5)
pred=pred.sort_values(["기준분기","예측_폐업확률"],ascending=[True,False])
pred.to_csv("results/LightGBM_예측결과_2026.csv",index=False,encoding="utf-8-sig")
print(pred.shape)
