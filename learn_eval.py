#!/usr/bin/env python3
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15          # players with a real chance of playing
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
tr=G<=21; te=G>=22
print("train rows %d (GW6-21)  test rows %d (GW22-38)\n"%(tr.sum(),te.sum()))
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
ep=X[:,F.index("ep")]
lin_ep=LinearRegression().fit(ep[tr,None],Y[tr]); p_ep=lin_ep.predict(ep[te,None])
lin=LinearRegression().fit(X[tr],Y[tr]); p_lin=lin.predict(X[te])
gbm=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.04,max_leaf_nodes=24,min_samples_leaf=60,l2_regularization=1.0,random_state=0).fit(X[tr],Y[tr])
p_gbm=gbm.predict(X[te])
print("OUT-OF-SAMPLE R2 on unseen gameweeks")
print("  model EP (rescaled)   %.4f"%r2(Y[te],p_ep))
print("  linear, all features  %.4f"%r2(Y[te],p_lin))
print("  boosted trees         %.4f"%r2(Y[te],p_gbm))
scores={"model EP":ep[te],"learned (linear)":p_lin,"learned (trees)":p_gbm}
g=G[te]; y=Y[te]; e=E[te]
print("\nDECISIONS, per unseen gameweek, scored on actual points")
print("%-18s %12s %14s %14s"%("","top-15 picks","captain (pool","within-player"))
print("%-18s %12s %14s %14s"%("","avg pts","of top-8 EP)","timing corr"))
for name,s in scores.items():
    top=[];cap=[]
    for w in np.unique(g):
        m=g==w
        idx=np.where(m)[0]
        order=idx[np.argsort(-s[idx])]
        top.append(y[order[:15]].mean())
        pool=idx[np.argsort(-ep[te][idx])[:8]]           # the credible captaincy pool, fixed
        cap.append(y[pool[np.argmax(s[pool])]])
    # within-player timing: deviation from own mean
    num=den1=den2=0.0
    for pid in np.unique(e):
        m=e==pid
        if m.sum()<5: continue
        a=y[m]-y[m].mean(); b=s[m]-s[m].mean()
        num+=(a*b).sum(); den1+=(a*a).sum(); den2+=(b*b).sum()
    print("%-18s %12.2f %14.2f %14.3f"%(name,np.mean(top),np.mean(cap),num/np.sqrt(den1*den2)))
perfect=[];
for w in np.unique(g):
    idx=np.where(g==w)[0]; pool=idx[np.argsort(-ep[te][idx])[:8]]; perfect.append(y[pool].max())
print("%-18s %12s %14.2f"%("perfect captain","",np.mean(perfect)))
# paired captain significance: trees vs EP
cd=[]
for w in np.unique(g):
    idx=np.where(g==w)[0]; pool=idx[np.argsort(-ep[te][idx])[:8]]
    cd.append(y[pool[np.argmax(p_gbm[pool])]]-y[pool[np.argmax(ep[te][pool])]])
cd=np.array(cd)
print("\ncaptain: trees minus EP  %+.2f a gameweek (se %.2f, n=%d) = %+.0f over 38 doubled-weeks"%(cd.mean(),cd.std()/np.sqrt(len(cd)),len(cd),cd.mean()*38))
td=[]
for w in np.unique(g):
    idx=np.where(g==w)[0]
    td.append(y[idx[np.argsort(-p_gbm[idx])[:15]]].mean()-y[idx[np.argsort(-ep[te][idx])[:15]]].mean())
td=np.array(td)
print("top-15: trees minus EP  %+.2f a player a gameweek (se %.2f) -> across an XI %+.1f a gameweek"%(td.mean(),td.std()/np.sqrt(len(td)),td.mean()*11))
from sklearn.inspection import permutation_importance
pi=permutation_importance(gbm,X[te],Y[te],n_repeats=3,random_state=0)
print("\nwhat the trees lean on (drop in R2 when shuffled):")
for i in np.argsort(-pi.importances_mean)[:10]:
    print("  %-10s %.4f"%(F[i],pi.importances_mean[i]))
