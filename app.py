import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date,timedelta
from itertools import product

st.set_page_config(page_title="家銘策略實驗室 V2",page_icon="🧪",layout="wide")
st.title("🧪 家銘台股趨勢雷達 V2｜策略實驗室")
st.caption("用樣本外驗證找穩定正期望，不追求漂亮但脆弱的歷史勝率。")
API="https://api.finmindtrade.com/api/v4/data"

@st.cache_data(ttl=3600)
def price(sid,start,end,token=""):
    p={"dataset":"TaiwanStockPrice","data_id":sid,"start_date":str(start),"end_date":str(end)}
    h={"Authorization":f"Bearer {token}"} if token else {}
    r=requests.get(API,params=p,headers=h,timeout=30); r.raise_for_status()
    d=pd.DataFrame(r.json().get("data",[]))
    if d.empty:return d
    d["date"]=pd.to_datetime(d["date"]); d=d.sort_values("date")
    return d.rename(columns={"close":"Close","Trading_Volume":"Volume"}).reset_index(drop=True)

def prep(d):
    x=d.copy()
    for n in [5,10,20,60,120]: x[f"MA{n}"]=x["Close"].rolling(n).mean()
    x["VOL20"]=x["Volume"].rolling(20).mean()
    x["VR"]=x["Volume"]/x["VOL20"]
    x["RET20"]=x["Close"].pct_change(20)
    x["HH20"]=x["Close"].rolling(20).max()
    return x

def sig(x,vm,gap,mom,ema,tl):
    if tl==2: trend=(x.Close>x.MA5)&(x.MA5>x.MA10)&(x.MA10>x.MA20)
    else: trend=(x.Close>x.MA5)&(x.MA5>x.MA10)&(x.MA10>x.MA20)&(x.MA20>x.MA60)
    slope=(x.MA5>x.MA5.shift(1))&(x.MA10>x.MA10.shift(1))
    entry=trend&slope&(x.VR>=vm)&((x.Close/x.MA5-1)<=gap)&(x.RET20>=mom)&(x.Close>=x.HH20.shift(1))
    return entry,x.Close<x[f"MA{ema}"]

def bt(x,p,cost=.004425):
    en,ex=sig(x,*p); trades=[]; hold=False
    for i,r in x.iterrows():
        if not hold and bool(en.iloc[i]): hold=True; ep=r.Close
        elif hold and bool(ex.iloc[i]):
            trades.append(r.Close/ep-1-cost); hold=False
    if not trades:return None
    s=pd.Series(trades); w=s[s>0]; l=s[s<=0]
    eq=(1+s).cumprod()
    return dict(n=len(s),wr=(s>0).mean(),pf=w.sum()/abs(l.sum()) if len(l) and l.sum()!=0 else np.inf,
                exp=s.mean(),total=(1+s).prod()-1,mdd=(eq/eq.cummax()-1).min())

with st.sidebar:
    sid=st.text_input("股票代號","2409")
    years=st.slider("資料年數",3,10,5)
    token=st.text_input("FinMind Token",type="password")

if st.button("開始策略壓力測試",type="primary",use_container_width=True):
    end=date.today(); start=end-timedelta(days=365*years+180)
    raw=price(sid,start,end,token)
    if raw.empty: st.error("查無資料"); st.stop()
    x=prep(raw).dropna().reset_index(drop=True)
    cut=int(len(x)*.7); ins=x.iloc[:cut].reset_index(drop=True); oos=x.iloc[cut:].reset_index(drop=True)
    combos=list(product([1.,1.2,1.5],[.03,.05,.08],[0.,.03,.06],[10,20],[2,3]))
    rows=[]; bar=st.progress(0)
    for i,p in enumerate(combos):
        a,b=bt(ins,p),bt(oos,p)
        if a and b:
            rows.append({"量倍":p[0],"最大乖離":p[1],"20日動能":p[2],"出場MA":p[3],"趨勢級":p[4],
                         "樣本內交易":a["n"],"樣本內勝率":a["wr"],"樣本內PF":a["pf"],
                         "樣本外交易":b["n"],"樣本外勝率":b["wr"],"樣本外PF":b["pf"],
                         "樣本外期望":b["exp"],"樣本外報酬":b["total"],"樣本外MDD":b["mdd"]})
        bar.progress((i+1)/len(combos))
    bar.empty()
    if not rows: st.warning("訊號太少，請增加資料年數或換股票。"); st.stop()
    r=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    r["穩健分數"]=r["樣本外PF"].fillna(3).clip(0,3)*25+r["樣本外勝率"]*30+(r["樣本外期望"].clip(-.05,.10)+.05)*200-abs(r["樣本內勝率"]-r["樣本外勝率"])*25
    r=r.sort_values("穩健分數",ascending=False)
    top=r.iloc[0]
    st.subheader("🏆 樣本外最佳穩健組合")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("勝率",f'{top["樣本外勝率"]:.1%}')
    c2.metric("Profit Factor",f'{top["樣本外PF"]:.2f}')
    c3.metric("每筆期望",f'{top["樣本外期望"]:.2%}')
    c4.metric("最大回撤",f'{top["樣本外MDD"]:.2%}')
    st.write("最佳組合：量倍 %.1f｜5MA最大乖離 %.0f%%｜20日動能至少 %.0f%%｜趨勢級 %d｜跌破 MA%d 出場" %
             (top["量倍"],top["最大乖離"]*100,top["20日動能"]*100,int(top["趨勢級"]),int(top["出場MA"])))
    show=r.head(20).copy()
    for c in ["最大乖離","20日動能","樣本內勝率","樣本外勝率","樣本外期望","樣本外報酬","樣本外MDD"]:
        show[c]=show[c].map(lambda v:f"{v:.2%}")
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.info("先看樣本外 PF、期望值、最大回撤與交易數。只有樣本內勝率很高的參數，不採用。")

st.caption("研究用途，不構成投資建議。回測仍有資料品質、滑價、流動性、除權息與過度擬合風險。")
