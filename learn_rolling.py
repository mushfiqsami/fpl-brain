import numpy as np, sys, os, statistics
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LinearRegression, Ridge
from fplbrain.api import ArchiveClient
from fplbrain import optimise
from backtest import f
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
ac=ArchiveClient("2025-26"); teams={t["name"]:int(t["id"]) for t in ac.teams()}
pos_i=[F.index(p) for p in ("pos_gk","pos_def","pos_mid","pos_fwd")]
res={"model EP":[],"learned (expanding window)":[]}
caps={"model EP":[],"learned (expanding window)":[]}
for gw in range(12,39):
    tr=G<gw
    mdl=Ridge(alpha=1.0).fit(X[tr],Y[tr])
    rows={}
    for r in ac.gw(gw): rows.setdefault(int(r["element"]),r)
    idx=np.where(G==gw)[0]
    scores={"model EP":X[idx,F.index("ep")],"learned (expanding window)":mdl.predict(X[idx])}
    pool=[]
    for k,i in enumerate(idx):
        r=rows.get(int(E[i]))
        if not r: continue
        pool.append(dict(id=int(E[i]),k=k,club_id=teams.get(r["team"]),pos=int(np.argmax(X[i,pos_i]))+1,price=f(r,"value")/10.0,name=r["name"]))
    by={p["id"]:p for p in pool}; act={int(E[i]):Y[i] for i in idx}
    for lab,s in scores.items():
        ep={p["id"]:{gw:float(max(0.0,s[p["k"]]))} for p in pool}
        b=optimise.build_squad(pool,ep,100.0,[gw],decay=1.0)
        ids=[p["id"] for p in b["squad"]]
        xi,cap,_v,_b,_d=optimise.rank_xi(ids,ep,gw,by)
        res[lab].append(sum(act[i] for i in xi)+act[cap]); caps[lab].append(act[cap])
    print("GW%d  EP %3.0f  learned %3.0f"%(gw,res["model EP"][-1],res["learned (expanding window)"][-1]),flush=True)
dd=[a-b for a,b in zip(res["learned (expanding window)"],res["model EP"])]
se=statistics.pstdev(dd)/len(dd)**0.5
m=statistics.fmean(dd)
print("\nn=%d gameweeks"%len(dd))
print("model EP : %.1f a GW   learned: %.1f a GW"%(statistics.fmean(res["model EP"]),statistics.fmean(res["learned (expanding window)"])))
print("learned minus model: %+.2f a GW (se %.2f, %.2f sd) -> %+.0f a season  %s"%(m,se,m/se,m*38,"SIGNIFICANT" if abs(m)>1.96*se else "not significant"))
print("weeks learned won / lost / tied: %d / %d / %d"%(sum(x>0 for x in dd),sum(x<0 for x in dd),sum(x==0 for x in dd)))
cd=[a-b for a,b in zip(caps["learned (expanding window)"],caps["model EP"])]
print("captain points alone: %+.2f a GW"%statistics.fmean(cd))
