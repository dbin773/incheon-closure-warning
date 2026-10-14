"""
업태구분명 재분류가 데이터·모델 성능을 바꾸는지 확인
실행: python 업태재분류_실험.py <사업체패널.pkl> <새로해보는2 칸데이터.csv> <원본 통합 raw.csv>
 1) 팀원 규칙: 카페 = 원본 업태구분명이 까페·전통찻집·다방·커피숍, 나머지 일반음식점(+휴게 패스트푸드) = 음식점
 2) 문제: 일반음식점으로 등록된 카페(업태 '기타', '한식', '경양식' 등)가 음식점으로 들어감
 3) 수정 규칙: 음식점 중 사업장명에 커피·카페·디저트·베이커리 등이 들어가면 카페로 (호프·맥주·주점·사주·보드게임 등은 제외)
 4) 두 분류로 칸 데이터를 똑같은 방법으로 다시 만들고, 팀원 M1 설정으로 성능 비교
"""
import sys, warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 칸데이터_재구성 import build, raw_label, same_quarter, K
warnings.filterwarnings("ignore")

CAFE_KW = (r"커피|카페|까페|cafe|café|coffee|스타벅스|starbucks|이디야|투썸|메가엠지씨|메가커피|빽다방|컴포즈|탐앤탐스|"
           r"할리스|엔제리너스|파스쿠찌|매머드|더벤티|공차|티하우스|찻집|디저트|dessert|빙수|설빙|베이커리|bakery|제과|브런치|brunch")
NOT_KW = r"호프|맥주|비어|beer|주점|포차|소주|사주|타로|보드게임|만화|노래|pc|키즈|라이브"

def fix_label(names, label, raw_type=None):
    nm = names.fillna("").str.lower()
    hit = nm.str.contains(CAFE_KW) & ~nm.str.contains(NOT_KW)
    if raw_type is not None: hit &= raw_type.ne("라이브카페")
    out = label.copy(); out[(label == "음식점") & hit] = "카페"
    return out

p = pd.read_pickle(sys.argv[1]); area = pd.read_csv(sys.argv[2], encoding="utf-8-sig")
raw = pd.read_csv(sys.argv[3], encoding="utf-8-sig", low_memory=False,
                  usecols=["시설유형", "관리번호", "업태구분명", "사업장명", "인허가일자", "폐업일자", "지번주소"])
ids = set(p.관리번호)
rt = raw.set_index("관리번호").업태구분명
p["원본업태"] = p.관리번호.map(rt)
p["업태_원래"] = p.업태구분명
p["업태_수정"] = fix_label(p.사업장명, p.업태구분명, p.원본업태)
lab0 = raw_label(raw); lab1 = fix_label(raw.사업장명, lab0, raw.업태구분명)

# ---- 재분류 규모 ----
st = p.drop_duplicates("관리번호")
moved = st[st.업태_원래 != st.업태_수정]
print("패널 가게:", len(st), "| 음식점→카페로 옮긴 가게:", len(moved),
      f"| 원래 카페 {int((st.업태_원래=='카페').sum())} → 수정 후 {int((st.업태_수정=='카페').sum())}")
print("옮긴 가게의 원본 업태:", moved.원본업태.value_counts().head(8).to_dict())
moved[["관리번호", "사업장명", "원본업태", "업태_원래", "업태_수정"]].to_csv("재분류_가게목록.csv", index=False, encoding="utf-8-sig")
nopanel = raw[~raw.관리번호.isin(ids) & raw.시설유형.eq("휴게음식점") & lab0.isna() &
              raw.사업장명.fillna("").str.lower().str.contains(CAFE_KW) & (raw.폐업일자.fillna("9999") >= "2023")]
print("참고: 패널에 아예 없는 휴게음식점 중 카페 이름 가게(기타 휴게음식점 등):", len(nopanel))

# ---- 칸 데이터 두 벌 ----
c0 = build(p, "업태_원래", area, same_quarter(raw, ids, lab0))
c1 = build(p, "업태_수정", area, same_quarter(raw, ids, lab1))
c1.to_csv("인천_분기x읍면동x업태_업태수정.csv", index=False, encoding="utf-8-sig")
for nm, c in [("원래", c0), ("수정", c1)]:
    s = c.groupby("업태구분명").agg(칸=("읍면동업태_영업점포수", "size"), 점포=("읍면동업태_영업점포수", "sum"), 폐업=("다음분기_폐업수", "sum"))
    s["폐업률"] = (s.폐업 / s.점포).round(4); print(nm, s.to_dict("index"))

BASE = ["분기번호","시간순번","업태구분_2진수","읍면동업태_영업점포수","평균업력개월","업력1년미만_점포비중",
        "평균시설규모","중앙시설규모","소규모점포비중","다중이용업소비중","법정동_실제총인구",
        "법정동_인구증감률_QoQ","법정동_인구증감률_YoY","법정동_인구천명당_전체점포수","법정동_인구천명당_동일업태점포수",
        "읍면동업태_분기개업수","읍면동업태_분기폐업수","읍면동업태_점포비중","읍면동업태_직전분기말점포수",
        "읍면동업태_개업률","읍면동업태_폐업률","읍면동업태_점포수증감률","읍면동업태_폐업률_lag1",
        "읍면동업태_폐업률_lag4","읍면동업태_폐업률_4Q평균","읍면동업태_폐업률_전분기대비증감"]
P = dict(objective="regression_l1", learning_rate=0.05, num_leaves=15, min_child_samples=70, reg_alpha=0.1,
         reg_lambda=1.0, colsample_bytree=0.9, subsample=0.9, n_estimators=120, verbosity=-1, n_jobs=4)
T, N = "다음분기_폐업률", "읍면동업태_영업점포수"

def metrics(y, p_, n):
    w = np.sqrt(n); big = n >= 30
    return dict(MAE=mean_absolute_error(y, p_), RMSE=np.sqrt(mean_squared_error(y, p_)),
                WMAE=np.average(np.abs(y - p_), weights=w), WRMSE=np.sqrt(np.average((y - p_) ** 2, weights=w)),
                Pearson=pearsonr(y, p_)[0], Spearman=spearmanr(y, p_)[0], R2=r2_score(y, p_),
                가중R2=1 - np.sum(n * (y - p_) ** 2) / np.sum(n * (y - np.average(y, weights=n)) ** 2),
                Spearman_점포30이상=spearmanr(y[big], p_[big])[0] if big.sum() > 5 else np.nan)

rows = []
for ty in [2026, 2025]:
    for nm, c in [("원래 분류", c0), ("수정 분류", c1)]:
        tr, te = c[c.기준연도 < ty], c[c.기준연도 == ty]
        pr = np.mean([np.clip(lgb.LGBMRegressor(**P, random_state=s).fit(tr[BASE], tr[T], sample_weight=np.sqrt(tr[N]))
                              .predict(te[BASE]), 0, 1) for s in range(5)], axis=0)
        for part, msk in [("전체", np.ones(len(te), bool)), ("음식점", te.업태구분명.eq("음식점").values), ("카페", te.업태구분명.eq("카페").values)]:
            rows.append({"테스트": ty, "분류": nm, "범위": part, "칸수": int(msk.sum()),
                         **metrics(te[T].values[msk], pr[msk], te[N].values[msk])})
res = pd.DataFrame(rows).round(4)
res.to_csv("업태재분류_결과.csv", index=False, encoding="utf-8-sig")
print(res.to_string(index=False))
