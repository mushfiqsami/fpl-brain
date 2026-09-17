import numpy as np, sys, os
from sklearn.linear_model import LinearRegression
d=np.load("learn_data.npz",allow_pickle=True)
X,Y,G,E=d["X"],d["Y"],d["G"],d["E"]; F=list(d["features"])
rel=X[:,F.index("p_appear")]>0.15
X,Y,G,E=X[rel],Y[rel],G[rel],E[rel]
tr=G<=21; te=G>=22
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
ep=X[:,F.index("ep")]
full=LinearRegression().fit(X[tr],Y[tr]); pf=full.predict(X[te])
base=LinearRegression().fit(ep[tr,None],Y[tr]); pb=base.predict(ep[te,None])
nomin=[i for i,f in enumerate(F) if f not in ("min_l1","min_l3")]
nm=LinearRegression().fit(X[tr][:,nomin],Y[tr]); pn=nm.predict(X[te][:,nomin])
print("R2  EP %.4f | all features %.4f | all EXCEPT recent minutes %.4f"%(r2(Y[te],pb),r2(Y[te],pf),r2(Y[te],pn)))
ml1=X[te][:,F.index("min_l1")]
for lab,m in (("played 0 min last week",ml1==0),("played 1-59",(ml1>0)&(ml1<60)),("played 60+",ml1>=60)):
    sse_b=((Y[te][m]-pb[m])**2).sum(); sse_f=((Y[te][m]-pf[m])**2).sum()
    print("  %-24s n=%5d  error removed by learning: %5.1f%%   model EP %.2f learned %.2f actual %.2f"%(
        lab,m.sum(),100*(sse_b-sse_f)/sse_b,ep[te][m].mean(),pf[m].mean(),Y[te][m].mean()))
print("\ncoefficients (linear, all features):")
for f,c in zip(F,full.coef_): print("  %-10s %+.3f"%(f,c))
print("  intercept  %+.3f"%full.intercept_)
