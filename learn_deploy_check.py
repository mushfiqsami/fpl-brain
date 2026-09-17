import numpy as np, statistics
from sklearn.linear_model import Ridge
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
KEEP=["ep","market","own","price","home","fdr","min_l1","min_l3","pts_l3","start_l5","pos_gk","pos_def","pos_mid","pos_fwd"]
ki=[F.index(k) for k in KEEP]
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
tr=G<=21; te=G>=22
full=Ridge(alpha=1.0).fit(X[tr],Y[tr]); red=Ridge(alpha=1.0).fit(X[tr][:,ki],Y[tr])
print("R2 on unseen GWs: full %.4f  deployable (no p_appear/exp_min) %.4f  model EP %.4f"%(
  r2(Y[te],full.predict(X[te])),r2(Y[te],red.predict(X[te][:,ki])),
  r2(Y[te],np.polyval(np.polyfit(X[tr][:,0],Y[tr],1),X[te][:,0]))))
# how fast does the gain fade for gameweeks further ahead? target = same player's points k weeks later
key={(int(e),int(g)):i for i,(e,g) in enumerate(zip(E,G))}
print("\nhorizon (features at GW, points at GW+k), test on unseen GWs:")
for k in range(0,5):
    rows=[(i,key[(int(E[i]),int(G[i])+k)]) for i in range(len(Y)) if (int(E[i]),int(G[i])+k) in key]
    a=np.array([r[0] for r in rows]); b=np.array([r[1] for r in rows])
    trk=G[a]<=21-k; tek=G[a]>=22
    Xk=X[a][:,ki]; Yk=Y[b]
    m=Ridge(alpha=1.0).fit(Xk[trk],Yk[trk])
    base=np.polyfit(Xk[trk][:,0],Yk[trk],1)
    rl=r2(Yk[tek],m.predict(Xk[tek])); rb=r2(Yk[tek],np.polyval(base,Xk[tek][:,0]))
    print("  k=%d  model EP %.4f  learned %.4f  gain %+.4f"%(k,rb,rl,rl-rb))
print("\ndeployable coefficients, fitted on ALL of 2025/26:")
allm=Ridge(alpha=1.0).fit(X[:,ki],Y)
for k_,c in zip(KEEP,allm.coef_): print("  %-10s %+.4f"%(k_,c))
print("  intercept  %+.4f"%allm.intercept_)
np.save("learn_coef.npy",np.concatenate([[allm.intercept_],allm.coef_]))
