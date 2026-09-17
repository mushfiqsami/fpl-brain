import numpy as np, sys, os, statistics
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from fplbrain.api import ArchiveClient
from fplbrain import optimise
from backtest import f
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
tr=G<=21
lin=LinearRegression().fit(X[tr],Y[tr])
gbm=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.04,max_leaf_nodes=24,min_samples_leaf=60,l2_regularization=1.0,random_state=0).fit(X[tr],Y[tr])
ac=ArchiveClient("2025-26"); teams={t["name"]:int(t["id"]) for t in ac.teams()}
res={k:[] for k in ("model EP","learned linear","learned trees")}
for gw in range(22,39):
    rows={}
    for r in ac.gw(gw):
        rows.setdefault(int(r["element"]),r)
    m=G==gw; idx=np.where(m)[0]
    scores={"model EP":X[idx,F.index("ep")],"learned linear":lin.predict(X[idx]),"learned trees":gbm.predict(X[idx])}
    pool=[];pos_i=[F.index(p) for p in ("pos_gk","pos_def","pos_mid","pos_fwd")]
    for k,i in enumerate(idx):
        r=rows.get(int(E[i]))
        if not r: continue
        pos=int(np.argmax(X[i,pos_i]))+1
        pool.append(dict(id=int(E[i]),k=k,club_id=teams.get(r["team"]),pos=pos,price=f(r,"value")/10.0,name=r["name"]))
    by={p["id"]:p for p in pool}
    act={int(E[i]):Y[i] for i in idx}
    for lab,s in scores.items():
        ep={p["id"]:{gw:float(max(0.0,s[p["k"]]))} for p in pool}
        b=optimise.build_squad(pool,ep,100.0,[gw],decay=1.0)
        if b["status"]!="Optimal": continue
        ids=[p["id"] for p in b["squad"]]
        xi,cap,_v,_b,_d=optimise.rank_xi(ids,ep,gw,by)
        res[lab].append(sum(act[i] for i in xi)+act[cap])
    print("GW%d  "%gw+"  ".join("%s %3.0f"%(k.split()[-1],v[-1]) for k,v in res.items()),flush=True)
print()
base=res["model EP"]
for k,v in res.items():
    dd=[a-b for a,b in zip(v,base)]
    se=statistics.pstdev(dd)/len(dd)**0.5 if len(dd)>1 else 0
    print("%-16s %.1f a GW  (%+.1f vs model, se %.1f)  -> %+.0f over a season"%(k,statistics.fmean(v),statistics.fmean(dd),se,statistics.fmean(dd)*38))
