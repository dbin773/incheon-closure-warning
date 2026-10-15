"""
사업체 패널 → 분기×읍면동×업태 칸 데이터 다시 만들기 (업태 분류만 바꿔서 비교하기 위함)
팀원 칸 데이터(새로해보는2)의 정의를 따라 다시 계산하고, 원래 분류로 만들었을 때 팀원 파일과 얼마나 같은지 확인함.
 - 읍면동 이름의 'N가'는 합침 (관동1가·2가·3가 → 관동)
 - 읍면동_(동 전체) 변수·법정동 인구 변수는 업태와 무관하므로 팀원 파일 값을 그대로 붙임
 - 분기폐업수(t) = 직전 분기 칸의 다음분기 폐업수, 직전분기말점포수(t) = 직전 분기 점포수, 개업수 = 점포수 − 직전 + 폐업수
   (2023Q1은 그 전 분기가 없어 비워 둠)
사용: from 칸데이터_재구성 import build ; build(panel_df, label_col, area_df)
"""
import re, numpy as np, pandas as pd

K = ["군구명_개편후", "읍면동_개편후", "업태구분명"]

CAFE_TYPES = {"까페", "전통찻집", "다방", "커피숍"}          # 팀원 패널의 카페 정의 (원본 업태구분명 기준)

def raw_label(raw):
    """원본 업태구분명 → 팀원 규칙의 음식점/카페 (휴게음식점은 커피숍·다방·전통찻집·패스트푸드만 패널에 들어감)"""
    lab = pd.Series(None, index=raw.index, dtype=object)
    lab[raw.업태구분명.isin(CAFE_TYPES)] = "카페"
    lab[(raw.시설유형 == "일반음식점") & ~raw.업태구분명.isin(CAFE_TYPES)] = "음식점"
    lab[(raw.시설유형 == "휴게음식점") & (raw.업태구분명 == "패스트푸드")] = "음식점"
    return lab

def same_quarter(raw, panel_ids, label):
    """같은 분기에 열고 닫아 패널에 없는 가게 수 (팀원 칸 데이터의 분기개업수·분기폐업수에는 들어 있음)"""
    r = raw[~raw.관리번호.isin(panel_ids)].copy()
    r["업태구분명"] = label.loc[r.index]
    r = r[r.업태구분명.notna()]
    oq = pd.to_datetime(r.인허가일자, errors="coerce").dt.to_period("Q")
    cq = pd.to_datetime(r.폐업일자, errors="coerce").dt.to_period("Q")
    r = r[(oq == cq) & (cq >= pd.Period("2023Q1")) & (cq <= pd.Period("2026Q2"))]
    a = r.지번주소.fillna("").str.extract(r"인천광역시 (\S+) (\S+?[동면읍가])(?:\s|$)")
    r["군구명_개편후"] = a[0]; r["읍면동_개편후"] = a[1].str.replace(r"\d+가$", "", regex=True)
    r["기준분기"] = cq.loc[r.index].astype(str)
    return r.groupby(K + ["기준분기"]).size().rename("동분기개폐업").reset_index()

def build(p, label_col, area, sq=None):
    p = p.copy()
    p["읍면동_개편후"] = p["읍면동_개편후"].str.replace(r"\d+가$", "", regex=True)
    p["업태구분명"] = p[label_col]
    p["시설총규모"] = p["시설총규모"].replace(0, np.nan)
    p["소규모"] = p["시설총규모"] < p.groupby("업태구분명")["시설총규모"].transform(lambda s: s.quantile(0.25))  # 업태별 하위 25% (팀원 정의를 정확히 재현하진 못함)
    p["다중"] = p["다중이용업소여부"].eq("Y")
    g = p.groupby(K + ["기준분기", "시간순번"])
    c = g.agg(읍면동업태_영업점포수=("관리번호", "size"), 평균업력개월=("업력개월", "mean"),
              업력1년미만_점포비중=("업력개월", lambda s: (s < 12).mean()), 평균시설규모=("시설총규모", "mean"),
              중앙시설규모=("시설총규모", "median"), 소규모점포비중=("소규모", "mean"), 다중이용업소비중=("다중", "mean"),
              다음분기_폐업수=("다음분기_폐업여부", "sum")).reset_index()
    # 빠진 분기도 칸을 만들어 lag가 정확하도록
    full = c[K].drop_duplicates().merge(c[["기준분기", "시간순번"]].drop_duplicates(), how="cross")
    c = full.merge(c, on=K + ["기준분기", "시간순번"], how="left").sort_values(K + ["시간순번"]).reset_index(drop=True)
    c[["읍면동업태_영업점포수", "다음분기_폐업수"]] = c[["읍면동업태_영업점포수", "다음분기_폐업수"]].fillna(0)
    gg = c.groupby(K)
    n = c.읍면동업태_영업점포수
    c["읍면동업태_직전분기말점포수"] = gg.읍면동업태_영업점포수.shift(1)
    c["읍면동업태_분기폐업수"] = gg.다음분기_폐업수.shift(1)
    if sq is not None:
        c = c.merge(sq, on=K + ["기준분기"], how="left"); c["동분기개폐업"] = c["동분기개폐업"].fillna(0)
        c["읍면동업태_분기폐업수"] = c.읍면동업태_분기폐업수 + c.동분기개폐업
        c = c.drop(columns="동분기개폐업")
    n = c.읍면동업태_영업점포수
    c["읍면동업태_분기개업수"] = n - c.읍면동업태_직전분기말점포수 + c.읍면동업태_분기폐업수
    sp = c.읍면동업태_직전분기말점포수.replace(0, np.nan)
    c["읍면동업태_개업률"] = c.읍면동업태_분기개업수 / sp
    c["읍면동업태_폐업률"] = c.읍면동업태_분기폐업수 / sp
    c["읍면동업태_점포수증감률"] = (n - c.읍면동업태_직전분기말점포수) / sp
    fr = c.groupby(K)["읍면동업태_폐업률"]
    c["읍면동업태_폐업률_lag1"] = fr.shift(1); c["읍면동업태_폐업률_lag4"] = fr.shift(4)
    c["읍면동업태_폐업률_4Q평균"] = fr.transform(lambda s: s.rolling(4, min_periods=1).mean())
    c["읍면동업태_폐업률_전분기대비증감"] = c.읍면동업태_폐업률 - fr.shift(1)
    c = c[n > 0].copy()
    c["다음분기_폐업률"] = c.다음분기_폐업수 / c.읍면동업태_영업점포수
    c["업태구분_2진수"] = (c.업태구분명 == "카페").astype(int)
    # 동 전체 변수 (업태 무관) + 인구
    dong_cols = ["기준분기", "군구명_개편후", "읍면동_개편후", "예측대상분기", "기준연도", "분기번호", "법정동_실제총인구",
                 "법정동_인구증감률_QoQ", "법정동_인구증감률_YoY", "법정동_인구천명당_전체점포수"] + \
                [x for x in area.columns if x.startswith("읍면동_") and x != "읍면동_개편후"]
    d = area[dong_cols].drop_duplicates(["기준분기", "군구명_개편후", "읍면동_개편후"])
    c = c.merge(d, on=["기준분기", "군구명_개편후", "읍면동_개편후"], how="inner")
    c["법정동_인구천명당_동일업태점포수"] = c.읍면동업태_영업점포수 / c.법정동_실제총인구 * 1000
    c["읍면동업태_점포비중"] = c.읍면동업태_영업점포수 / c.읍면동_영업점포수
    return c[[x for x in area.columns if x in c.columns]]

if __name__ == "__main__":
    import sys
    p = pd.read_pickle(sys.argv[1]); area = pd.read_csv(sys.argv[2], encoding="utf-8-sig")
    raw = pd.read_csv(sys.argv[3], encoding="utf-8-sig", low_memory=False,
                      usecols=["시설유형", "관리번호", "업태구분명", "사업장명", "인허가일자", "폐업일자", "지번주소"])
    sq = same_quarter(raw, set(p.관리번호), raw_label(raw))
    r = build(p, "업태구분명", area, sq)
    KK = K + ["기준분기"]
    m = area.merge(r, on=KK, suffixes=("", "_재"))
    print("팀원 행", len(area), "재구성 행", len(r), "겹침", len(m))
    for col in [x for x in area.columns if x + "_재" in m.columns and pd.api.types.is_numeric_dtype(area[x])]:
        a_, b_ = m[col], m[col + "_재"]
        ok = (np.isclose(a_.astype(float), b_.astype(float), rtol=1e-4, equal_nan=True) | b_.isna()).mean()
        print(f"  {col}: 일치 {ok:.3f}")
