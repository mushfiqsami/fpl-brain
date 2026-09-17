import numpy as np, sys, os, statistics
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import Ridge
from fplbrain.api import ArchiveClient
from fplbrain import optimise
from backtest import f
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
CORR=["market","own","price","home","fdr","min_l1","min_l3","pts_l3","start_l5","pos_gk","pos_def","pos_mid","pos_fwd"]
KEEP=["ep"]+CORR
ci=[F.index(k) for k in CORR]; ki=[F.index(k) for k in KEEP]; epi=F.index("ep")
ac=ArchiveClient("2025-26"); teams={t["name"]:int(t["id"]) for t in ac.teams()}
pos_i=[F.index(p) for p in ("pos_gk","pos_def","pos_mid","pos_fwd")]
labs=["model EP","A: learned, free","B: EP backbone + learned correction"]
res={k:[] for k in labs}; own_share={k:[] for k in labs}
gwrows={g:ac.gw(g) for g in range(12,39)}
for gw in range(12,39):
    tr=G<gw
    A=Ridge(alpha=1.0).fit(X[tr][:,ki],Y[tr])
    # B: EP stays at weight 1 (rescaled by one fitted slope); a correction is learned on the rest.
    slope=max(0.3,np.polyfit(X[tr][:,epi],Y[tr],1)[0])
    B=Ridge(alpha=1.0).fit(X[tr][:,ci],Y[tr]-slope*X[tr][:,epi])
    rows={}
    for r in gwrows[gw]: rows.setdefault(int(r["element"]),r)
    idx=np.where(G==gw)[0]
    sc={labs[0]:X[idx,epi],labs[1]:A.predict(X[idx][:,ki]),labs[2]:slope*X[idx,epi]+B.predict(X[idx][:,ci])}
    pool=[]
    for k,i in enumerate(idx):
        r=rows.get(int(E[i]))
        if not r: continue
        pool.append(dict(id=int(E[i]),k=k,club_id=teams.get(r["team"]),pos=int(np.argmax(X[i,pos_i]))+1,price=f(r,"value")/10.0,own=X[i,F.index("own")]))
    by={p["id"]:p for p in pool}; act={int(E[i]):Y[i] for i in idx}
    for lab,s in sc.items():
        ep={p["id"]:{gw:float(max(0.0,s[p["k"]]))} for p in pool}
        b=optimise.build_squad(pool,ep,100.0,[gw],decay=1.0)
        ids=[p["id"] for p in b["squad"]]
        xi,cap,_v,_b,_d=optimise.rank_xi(ids,ep,gw,by)
        res[lab].append(sum(act[i] for i in xi)+act[cap])
        own_share[lab].append(statistics.fmean(by[i]["own"] for i in xi))
    print("GW%d "%gw+"  ".join("%3.0f"%res[l][-1] for l in labs),flush=True)
print()
for l in labs:
    dd=[a-b for a,b in zip(res[l],res[labs[0]])]
    se=statistics.pstdev(dd)/len(dd)**0.5 if any(dd) else 0
    print("%-38s %.1f/GW  %+.2f (se %.2f)  %+.0f/season  avg XI ownership %.0f%%"%(
      l,statistics.fmean(res[l]),statistics.fmean(dd),se,statistics.fmean(dd)*38,100*statistics.fmean(own_share[l])))
