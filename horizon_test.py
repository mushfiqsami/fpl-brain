"""Does the model beat naive form over a 5-GW horizon, where fixtures matter?"""
import sys,os,statistics
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fplbrain.api import ArchiveClient
from fplbrain.model import TeamStrength,FixtureModel,PlayerModel
from fplbrain import calibrate
from backtest import build_state,POS_ID,f
ac=ArchiveClient("2025-26"); teams=ac.teams(); tbn={t["name"]:int(t["id"]) for t in teams}
cache={g:ac.gw(g) for g in range(1,39)}
fx_all=[]
for r in ac.fixtures():
    try: fx_all.append(dict(id=int(r["id"]),event=int(r["event"]) if r["event"] else None,
        team_h=int(r["team_h"]),team_a=int(r["team_a"]),finished=str(r["finished"]).lower()=="true",
        team_h_score=float(r["team_h_score"]) if r["team_h_score"] else None,
        team_a_score=float(r["team_a_score"]) if r["team_a_score"] else None))
    except: pass
bt=[dict(id=int(t["id"]),name=t["name"],short_name=t["short_name"],
    strength_attack_home=int(t["strength_attack_home"]),strength_attack_away=int(t["strength_attack_away"]),
    strength_defence_home=int(t["strength_defence_home"]),strength_defence_away=int(t["strength_defence_away"])) for t in teams]

print("5-GAMEWEEK HORIZON TEST  (predict total points over GW n..n+4)")
print("="*80)
print(f"{'from GW':>8}{'n':>6}{'model rho':>12}{'naive rho':>12}{'model top10':>14}{'naive top10':>14}{'field':>9}")
print("-"*80)
res=[]
for gw in [6,10,14,18,22,26,30]:
    H=[g for g in range(gw,gw+5) if g<=38]
    state=build_state(ac,gw-1,tbn,cache)
    fxk=[]
    for x in fx_all:
        y=dict(x)
        if y["event"] is not None and y["event"]>=gw: y["finished"]=False;y["team_h_score"]=y["team_a_score"]=None
        fxk.append(y)
    boot=dict(teams=bt,elements=list(state.values()),events=[])
    ts=TeamStrength.build(boot,fxk,prior_weight_games=6.0)
    fm=FixtureModel(ts,1.10,0.90); pm=PlayerModel(ts,fm)
    views={g:fm.team_view(fxk,g) for g in H}
    # actual total over horizon
    tot={};mins={}
    for g in H:
        for r in cache[g]:
            e=int(r["element"]); tot[e]=tot.get(e,0)+f(r,"total_points"); mins[e]=mins.get(e,0)+f(r,"minutes")
    P,A,N=[],[],[]
    for eid,e in state.items():
        if eid not in tot or mins.get(eid,0)<180: continue
        if e["now_cost"]/10.0 < 4.5: continue
        ep=sum(pm.project(e,views[g].get(e["team"],[]))["ep"] for g in H)
        gp=max(1,ts.games.get(e["team"],1))
        P.append(ep); A.append(tot[eid]); N.append(e["total_points"]/gp*len(H))
    if len(P)<80: continue
    rho=calibrate._spearman(P,A); nrho=calibrate._spearman(N,A)
    t10=statistics.fmean(A[i] for i in sorted(range(len(P)),key=lambda i:-P[i])[:10])
    n10=statistics.fmean(A[i] for i in sorted(range(len(N)),key=lambda i:-N[i])[:10])
    fld=statistics.fmean(A)
    res.append((rho,nrho,t10,n10,fld))
    print(f"{gw:>8}{len(P):>6}{rho:>12.3f}{nrho:>12.3f}{t10:>14.1f}{n10:>14.1f}{fld:>9.1f}")
print("-"*80)
m=lambda i: statistics.fmean(r[i] for r in res)
print(f"{'MEAN':>8}{'':>6}{m(0):>12.3f}{m(1):>12.3f}{m(2):>14.1f}{m(3):>14.1f}{m(4):>9.1f}")
print()
print(f"  Model rank correlation over 5 GWs : {m(0):+.3f}")
print(f"  Naive form baseline               : {m(1):+.3f}")
print(f"  Model top-10 scored {m(2):.1f} pts vs naive {m(3):.1f} vs field {m(4):.1f}")
print(f"  Model edge over naive top-10      : {m(2)-m(3):+.1f} pts per player over 5 GWs")
