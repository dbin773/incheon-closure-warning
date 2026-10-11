"""3단계 경보(관심·주의·경고) 검증
노트북과 같은 LightGBM(26개 변수, 같은 분할·파라미터)으로 사업체별 다음분기 폐업확률을 구한 뒤,
2025년 검증 데이터 분포로 기준선을 정하고(상위 5% 경고, 5~15% 주의, 15~30% 관심),
2026년 테스트 데이터에서 단계별 실제 폐업률을 확인한다. 읍면동×업태 평균확률 기준 단계도 함께 확인.
실행: python 경보단계_검증.py "[전처리10_최종]인천_사업체-분기_패널_군구인구_YoY보완 (2).zip"
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
LAB=["정상","관심","주의","경고"]
def stage(p, cuts): return pd.cut(p,[-1]+cuts+[2],labels=LAB)
def table(d, col_n, col_y, name):
    g=d.groupby("단계",observed=False).agg(대상수=(col_n,"sum") if col_n else ("p","size"), 실제폐업=(col_y,"sum"))
    g["비중%"]=(g["대상수"]/g["대상수"].sum()*100).round(1)
    g["실제폐업률%"]=(g["실제폐업"]/g["대상수"]*100).round(2)
    g["폐업중몫%"]=(g["실제폐업"]/g["실제폐업"].sum()*100).round(1)
    g["평균대비배"]=(g["실제폐업률%"]/(g["실제폐업"].sum()/g["대상수"].sum()*100)).round(2)
    print("\n==",name); print(g.loc[LAB[::-1]]); return g.loc[LAB[::-1]]
# 1) 사업체 단위: 검증 분포 상위 5% 경고, 5~15% 주의, 15~30% 관심
q=va["p"].quantile([0.70,0.85,0.95]).tolist(); print("사업체 cut",[round(x,4) for x in q])
te["단계"]=stage(te["p"],q); out={}
out["사업체_테스트전체"]=table(te,None,Y,"사업체 단위 (테스트 2026Q1~Q2)")
for u in ["음식점","카페"]:
    sub=te[te["업태구분_2진수"]==(1 if u=="카페" else 0)]
    out[f"사업체_{u}"]=table(sub,None,Y,f"사업체 단위 - {u}")
# 2) 읍면동×업태 단위 예상폐업률
def agg(d):
    a=d.groupby(["기준분기","군구명_개편후","읍면동_개편후","업태구분명"]).agg(n=("p","size"),exp=("p","mean"),y=(Y,"sum")).reset_index()
    return a[a.n>=20]
av,at=agg(va),agg(te)
q2=av["exp"].quantile([0.50,0.75,0.90]).tolist(); print("\n읍면동 cut",[round(x,4) for x in q2])
at["단계"]=stage(at["exp"],q2)
out["읍면동업태_테스트"]=table(at,"n","y","읍면동×업태 단위 (점포 20개 이상, 테스트)")
print("지역수",len(at))
pd.concat(out).to_csv("경보단계_검증.csv",encoding="utf-8-sig")
at.sort_values("exp",ascending=False).head(15).to_csv("경고지역_예시.csv",encoding="utf-8-sig",index=False)
print(at.sort_values("exp",ascending=False).head(10)[["기준분기","군구명_개편후","읍면동_개편후","업태구분명","n","exp","y","단계"]])
