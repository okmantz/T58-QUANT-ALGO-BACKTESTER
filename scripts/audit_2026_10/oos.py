import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import sys; 
import scan2, fast
cands=[("NQ","drive",{"or":30,"win":0,"minr":0.3,"maxr":9.9,"sf":0.5,"rr":2.5,"bias":0.25}),
       ("NQ","fail",{"or":15,"win":60,"minr":0.3,"maxr":9.9,"sf":1.0,"rr":2.5}),
       ("NQ","orb",{"or":30,"win":120,"minr":0.3,"maxr":9.9,"sf":0.5,"rr":2.5}),
       ("ES","fail",{"or":15,"win":60,"minr":0.3,"maxr":9.9,"sf":1.0,"rr":2.5}),
       ("GC","drive",{"or":15,"win":0,"minr":0.0,"maxr":9.9,"sf":1.0,"rr":2.5,"bias":0.25})]
for sym,fam,p in cands:
    for lab,(s,e) in {"DEV":("2020-01-01","2025-01-01"),"OOS":("2025-01-01","2026-10-01")}.items():
        A=scan2.prep(sym,"5m",s,e); trs=scan2.gen(A,sym,fam,p); d=fast.dollars(sym,trs); sm=fast.summarize(A,sym,d)
        print(sym,fam,lab,{k:sm[k] for k in("n","net","PF","wr")} if sm else None,fast.pass_sim(A,d) if sm else None)
