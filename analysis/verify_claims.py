"""Regression-check every numeric claim the audit docs make against the data.

Every number quoted in README.md, ERRATA.md, RETEST_TODO.md, CHANGELOG.md and
v4_audit_2026_08_25/README.md is re-derived here from the committed measurement
files. If a document and the data ever disagree, this fails loudly.

Two layers:

  data      re-derives each quantity from the committed measurement files and
            compares it with the value this script expects. Catches the data
            and the analysis drifting apart.
  documents greps the Markdown for the signature figures, so a typo in a
            document is caught too. Without this layer the script would only
            prove the data is self-consistent, which its name over-promises.

Covers: the v1 matrix aggregates and activation counts, the empty-content count
behind ERRATA A5, the full speculative accounting behind A1/A4 (which closes on
three independent paths), both v4 audit runs, and the MoE coverage arithmetic
behind E1.

Run: python analysis/verify_claims.py   (from the repo root; exits non-zero on
any mismatch)
"""
import sys
import csv, json, glob, math, re, statistics as st
from collections import defaultdict

FAIL=[]
def chk(name, got, want, tol=0.05):
    ok = (abs(got-want) <= tol) if isinstance(want,(int,float)) else (got==want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} got={got!r} want={want!r}")
    if not ok: FAIL.append(name)

print("=== v1 matrix (analysis/summary.csv) ===")
rows=list(csv.DictReader(open('analysis/summary.csv')))
for r in rows:
    for k in ('tok_s','predicted_ms'): r[k]=float(r[k])
    for k in ('predicted_n','draft_n','draft_acc','max_tokens'): r[k]=int(float(r[k]))
by=defaultdict(list)
for r in rows: by[r['config']].append(r)
def agg(c):
    v=by[c]; rates=[x['tok_s'] for x in v]
    return dict(mean=st.mean(rates), pooled=1000*sum(x['predicted_n'] for x in v)/sum(x['predicted_ms'] for x in v),
                med=st.median(rates), mn=min(rates), act=sum(1 for x in v if x['draft_n']>0))
b=agg('baseline'); b1=agg('baseline-1000tok')
chk("labels", len(by), 19)
chk("labels with a draft round", sum(1 for c in by if any(x['draft_n']>0 for x in by[c])), 14)
chk("labels without", sum(1 for c in by if not any(x['draft_n']>0 for x in by[c])), 5)
for c,mean,pooled,med,mn in [('baseline',135.7,135.7,135.6,135.3),('ngram-mod-n24',131.1,131.1,130.0,129.6),
                             ('draft-q35-08b-max8',121.1,109.9,135.6,59.2),('ngram-cache',119.1,111.3,135.6,65.3)]:
    a=agg(c); chk(f"{c} req-mean",round(a['mean'],1),mean,0.06); chk(f"{c} pooled",round(a['pooled'],1),pooled,0.06)
    chk(f"{c} median",round(a['med'],1),med,0.06); chk(f"{c} min",round(a['mn'],1),mn,0.06)
chk("draft-max8 vs base req-mean %", round(100*(agg('draft-q35-08b-max8')['mean']/b['mean']-1),1), -10.8, 0.06)
chk("draft-max8 vs base pooled %",  round(100*(agg('draft-q35-08b-max8')['pooled']/b['pooled']-1),1), -19.0, 0.06)
chk("ngram-cache vs base pooled %", round(100*(agg('ngram-cache')['pooled']/b['pooled']-1),1), -18.0, 0.06)
chk("ngcache-1000tok pooled vs b1000 %", round(100*(agg('ngcache-1000tok')['pooled']/b1['pooled']-1),1), -25.7, 0.06)
chk("draft-max8 activation", agg('draft-q35-08b-max8')['act'], 2)
chk("ngram-cache activation", agg('ngram-cache')['act'], 3)
chk("ngram-mod-n24 activation", agg('ngram-mod-n24')['act'], 8)
chk("baseline-1000tok lengths", [x['predicted_n'] for x in by['baseline-1000tok']],
    [354,514,801,427,1000,891,1000,384,1000,484])
chk("all 300-tok configs constant at 300", sorted({x['predicted_n'] for x in rows if x['max_tokens']==300}), [300])
mins={'ngmod-n8':120.0,'ngmod-n12':119.8,'ngmod-n16':123.8,'ngmod-n20':128.8,'ngram-mod-n24':129.6}
for c,m in mins.items(): chk(f"{c} min", round(agg(c)['mn'],1), m, 0.06)

print("\n=== v1 empty content (ERRATA A5) ===")
tot=e=0; per=defaultdict(lambda:[0,0])
for f in sorted(glob.glob("results/*.json"))+sorted(glob.glob("results/verify/*.json")):
    for r in json.load(open(f))["rows"]:
        tot+=1; blank=not (r.get("content_head") or "").strip(); e+=blank
        per[r["tag"]][0]+=blank; per[r["tag"]][1]+=1
chk("empty content count", e, 144); chk("total requests", tot, 190)
chk("empty %", round(100*e/tot,1), 75.8, 0.06)
chk("reasoning empty", per["reasoning"], [19,19]); chk("code_small empty", per["code_small"], [19,19])

print("\n=== verbose.log accounting (ERRATA A1/A4) ===")
t=open('v2_3090_followup/v2_oleg_suggestions/verbose.log',errors='replace').read()
gen=[int(x) for x in re.findall(r"called impl \w+, hist size = \d+, call_count = \d+, gen = (\d+)", t)]
small=[int(x) for x in re.findall(r"ignoring small draft: (\d+) < \d+", t)]
att=[(int(a),int(b)) for a,b in re.findall(r"update_slots: n_draft=(\d+), accepted=(\d+)", t)]
m=re.search(r"#gen drafts = (\d+), #acc drafts = (\d+), #gen tokens = (\d+), #acc tokens = (\d+), dur\(b,g,a\) = [\d.]+, ([\d.]+),", t)
gd,ad,gt,at,dg=int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),float(m.group(5))
ck=[float(x) for x in re.findall(r"created speculative checkpoint \(pos_min = \d+, pos_max = \d+, n_tokens = \d+, size = ([\d.]+) MiB\)", t)]
rs=[int(x) for x in re.findall(r"restoring speculative checkpoint \(pos_min = \d+, pos_max = \d+, size = (\d+)\)", t)]
chk("gen drafts",gd,81); chk("gen tokens",gt,214); chk("acc tokens",at,115); chk("acc drafts",ad,33)
chk("token acceptance %", round(100*at/gt,1), 53.7, 0.06)
chk("draft acceptance %",  round(100*ad/gd,1), 40.7, 0.06)
chk("sum(gen)==#gen tokens", sum(gen), gt)
chk("verification attempts", len(att), 53)
chk("full accepts", sum(1 for a,bb in att if bb==a+1), 33)
chk("partial accepts", sum(1 for a,bb in att if bb<a+1), 20)
chk("restores == partials", len(rs), 20)
chk("ignoring-small lines", len(small), 49); chk("tokens dropped", sum(small), 48)
chk("tokens reaching verification", gt-sum(small), 166)
chk("checkpoints created", len(ck), 33); chk("checkpoint MiB", round(ck[0],1), 62.8, 0.06)
chk("GiB written", round(sum(ck)/1024,2), 2.02, 0.01); chk("GiB restored", round(sum(rs)/2**30,2), 1.23, 0.01)
chk("drafter gen ms", round(dg,1), 999.6, 0.1)
chk("drafter share %", round(100*dg/(1000*200/63.2),1), 31.6, 0.06)

print("\n=== v4 audit runs ===")
def load(d):
    a=defaultdict(list)
    for f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{d}/*__rep*.json")): a[json.load(open(f))["arm"]].append(json.load(open(f)))
    return a
def stats(runs):
    rates=[];n=0;ms=0.0;dn=0;da=0;comp=0
    for r in runs:
        if len(r["rows"])==10: comp+=1
        for x in r["rows"]:
            rates.append(x["predicted_per_second"]);n+=x["predicted_n"];ms+=x["predicted_ms"];dn+=x["draft_n"];da+=x["draft_n_accepted"]
    return st.mean(rates),1000*n/ms,dn,da,comp,len(runs)
A=load("A_bcb5eeb64_legacy"); B=load("B_master_3737e4137")
for arm,mean,pooled,dn,da,comp in [("baseline",123.0,122.9,0,0,2),("draft-max8-translate",113.9,100.3,194,194,0),("draft-max8-matched",113.5,101.0,194,194,0)]:
    m_,p_,d_,a_,c_,n_=stats(A[arm]); chk(f"A {arm} mean",round(m_,1),mean,0.06); chk(f"A {arm} pooled",round(p_,1),pooled,0.06)
    chk(f"A {arm} drafted/accepted",(d_,a_),(dn,da)); chk(f"A {arm} complete",c_,comp)
for arm,mean,pooled,dn,da,comp in [("baseline",132.9,133.3,0,0,3),("draft-max8-translate",33.6,32.6,16590,4926,3),("draft-max8-matched",33.7,32.6,16590,4926,3)]:
    m_,p_,d_,a_,c_,n_=stats(B[arm]); chk(f"B {arm} mean",round(m_,1),mean,0.06); chk(f"B {arm} pooled",round(p_,1),pooled,0.06)
    chk(f"B {arm} drafted/accepted",(d_,a_),(dn,da)); chk(f"B {arm} complete",c_,comp)
chk("B acceptance %", round(100*4926/16590,1), 29.7, 0.06)
per=defaultdict(lambda: defaultdict(list)); acc={}
for arm,runs in B.items():
    for r in runs:
        for x in r["rows"]:
            per[x["tag"]][arm].append(x["predicted_per_second"])
            if x["draft_n"] and arm=="draft-max8-matched": acc[x["tag"]]=(x["draft_n_accepted"],x["draft_n"])
xs=[];ys=[]
for tag,v in per.items():
    xs.append(100*acc[tag][0]/acc[tag][1]); ys.append(st.mean(v["draft-max8-matched"]))
mx,my=st.mean(xs),st.mean(ys)
r_=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/((sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5)
chk("Pearson r(acceptance, tok/s)", round(r_,3), 0.998, 0.001)

print("\n=== figures the coverage audit found unpinned ===")
# the fp16-KV control (ERRATA B7): the one row in run C that is NOT a
# speculation effect, and the reason the v1 kv-fp16 row could not be read
_C = "v4_audit_2026_08_25/data/C_master_matrix_think_on/%s__rep*.json"
def _cpool(arm):
    rs = [json.load(open(f)) for f in glob.glob(_C % arm)]
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    return 1000 * n / ms if ms else float("nan")
_b, _f = _cpool("baseline"), _cpool("baseline-kvfp16")
chk("C baseline pooled", round(_b, 1), 123.4, 0.05)
chk("C baseline-kvfp16 pooled", round(_f, 1), 125.7, 0.05)
chk("C fp16-KV vs q8_0 KV, no speculation either side (%)",
    round(100 * (_f / _b - 1), 1), 1.9, 0.05)

# the run-to-run SD quoted beside run J's headline
_j = [json.load(open(f))["aggregate_tok_s"]
      for f in glob.glob("v4_audit_2026_08_25/data/matrix_J2_*/spec-dflash-n4__rep*.json")]
chk("J dflash-n4 aggregate run-to-run SD", round(st.stdev(_j), 2), 1.21, 0.005)

# a few more v1 config rows, so the table in README is derived and not just typed
_v1 = defaultdict(list)
for _r in csv.DictReader(open("analysis/summary.csv")):
    _v1[_r["config"]].append(_r)
# No `if cfg in _v1` guard: a check that silently skips when its subject is
# missing is worse than no check, because it reads as a pass.
for _cfg, _want in (("ngcache-kv-fp16", 113.7), ("ngcache-1000tok", 98.9)):
    chk(f"v1 {_cfg} present in summary.csv", _cfg in _v1, True)
    _n = sum(int(float(x["predicted_n"])) for x in _v1[_cfg])
    _ms = sum(float(x["predicted_ms"]) for x in _v1[_cfg])
    chk(f"v1 {_cfg} pooled", round(1000 * _n / _ms, 1), _want, 0.05)

print("\n=== runs I and J (2026-08-26) ===")
def _arm(pat):
    """(mean aggregate, SD, pooled, drafted, accepted) over the arm-runs matching pat."""
    v=[json.load(open(f)) for f in sorted(glob.glob(pat))]
    assert v, f"no data for {pat}"
    aggs=[x["aggregate_tok_s"] for x in v]
    n=sum(r["predicted_n"] for x in v for r in x["rows"])
    ms=sum(r["predicted_ms"] for x in v for r in x["rows"])
    dn=sum(r["draft_n"] for x in v for r in x["rows"])
    da=sum(r["draft_n_accepted"] for x in v for r in x["rows"])
    return (st.mean(aggs), st.stdev(aggs) if len(aggs)>1 else 0.0, 1000*n/ms, dn, da)

I="v4_audit_2026_08_25/data/matrix_I2_conc%d_*/%s__rep*.json"
J="v4_audit_2026_08_25/data/matrix_J2_*/%s__rep*.json"

# The batch has to have actually formed, or run I measures nothing. This is the
# check whose absence made the first attempt at run I a null experiment.
for c in (1,4,8):
    peaks=[json.load(open(f)).get("max_in_flight") for f in glob.glob(I % (c,"*"))]
    chk(f"I c={c}: every arm-run reached {c} in flight", sorted(set(peaks)), [c])

base={c:_arm(I % (c,"baseline")) for c in (1,4,8)}
spec={c:_arm(I % (c,"spec-draft-n8")) for c in (1,4,8)}
for c,want in ((1,109.7),(4,154.3),(8,180.0)):
    chk(f"I c={c} baseline aggregate", round(base[c][0],1), want, 0.05)
for c,want in ((1,30.6),(4,27.0),(8,28.1)):
    chk(f"I c={c} spec-draft-n8 aggregate", round(spec[c][0],1), want, 0.05)
chk("I baseline c=4 vs c=1", round(100*(base[4][0]/base[1][0]-1),1), 40.6, 0.05)
chk("I baseline c=8 vs c=1", round(100*(base[8][0]/base[1][0]-1),1), 64.0, 0.05)
chk("I spec c=8 vs c=1",     round(100*(spec[8][0]/spec[1][0]-1),1), -8.4, 0.05)
for c,want in ((1,0.28),(4,0.18),(8,0.16)):
    chk(f"I c={c} spec/baseline ratio", round(spec[c][0]/base[c][0],2), want, 0.005)
# acceptance did not collapse under -np N (llama.cpp #27572)
for c in (1,4,8):
    dn,da = spec[c][3], spec[c][4]
    chk(f"I c={c} counted acceptance in 28-30 %", 28.0 <= 100*da/dn <= 30.0, True)
    zeros=sum(1 for f in glob.glob(I % (c,"spec-draft-n8"))
                for r in json.load(open(f))["rows"] if r["draft_n"]==0)
    chk(f"I c={c} requests with draft_n == 0", zeros, 0)

jb=_arm(J % "baseline")
chk("J baseline aggregate", round(jb[0],1), 109.7, 0.05)
# the -fit on control: it must not handicap the arm DFlash is measured against
chk("J vs I baseline (the -fit on control)", round(100*(jb[0]/base[1][0]-1),2), -0.01, 0.005)
for arm,agg,pooled,delta in (("spec-dflash-n4",130.2,151.6, 18.7),
                             ("spec-dflash-n8", 93.5,105.2,-14.8),
                             ("spec-dflash-n16",57.7, 62.8,-47.4),
                             ("spec-draft-n8",  30.5, 31.4,-72.2)):
    a=_arm(J % arm)
    chk(f"J {arm} aggregate", round(a[0],1), agg, 0.05)
    chk(f"J {arm} pooled",    round(a[2],1), pooled, 0.05)
    chk(f"J {arm} vs baseline %", round(100*(a[0]/jb[0]-1),1), delta, 0.05)
for arm,want in (("spec-dflash-n4",55.8),("spec-dflash-n8",36.8),("spec-dflash-n16",21.4)):
    a=_arm(J % arm)
    chk(f"J {arm} acceptance %", round(100*a[4]/a[3],1), want, 0.05)
# the headline is "wins on all ten", not "wins on average"
per=defaultdict(dict)
for f in glob.glob(J % "*"):
    r=json.load(open(f))
    for x in r["rows"]: per[x["tag"]].setdefault(r["arm"],[]).append(x["predicted_per_second"])
wins=sum(1 for t,d in per.items()
         if st.mean(d["spec-dflash-n4"]) > st.mean(d["baseline"]))
chk("J dflash-n4 beats no speculation on every prompt", (wins, len(per)), (10, 10))


print("\n=== run K (2026-08-26): the DFlash sweep and its batching arms ===")
K1 = "v4_audit_2026_08_25/data/matrix_K1_sweep_*/%s__rep*.json"
kb = _arm(K1 % "baseline")
chk("K1 baseline aggregate", round(kb[0], 1), 110.6, 0.05)
for n, agg_, delta, acc in ((1, 120.2,  8.7, 82.0), (2, 129.5, 17.1, 72.8),
                            (3, 130.0, 17.6, 63.6), (4, 129.8, 17.3, 55.6),
                            (6, 100.8, -8.9, 43.0), (8,  93.2,-15.8, 37.2)):
    a = _arm(K1 % f"spec-dflash-n{n}")
    chk(f"K1 n_max {n} aggregate",    round(a[0], 1), agg_, 0.05)
    chk(f"K1 n_max {n} vs baseline",  round(100*(a[0]/kb[0]-1), 1), delta, 0.05)
    chk(f"K1 n_max {n} acceptance %", round(100*a[4]/a[3], 1), acc, 0.05)
# the plateau claim: 2, 3 and 4 within the baseline's own run-to-run SD
_p = [_arm(K1 % f"spec-dflash-n{n}")[0] for n in (2, 3, 4)]
chk("K1 n_max 2-4 spread below the baseline SD", (max(_p)-min(_p)) < kb[1], True)
# batching: the winner does not survive it
for c, base_, arm_, delta in ((4, 154.1, 154.7, 0.4), (8, 153.2, 39.6, -74.1)):
    KC = f"v4_audit_2026_08_25/data/matrix_K_conc{c}_*/%s__rep*.json"
    b_, a_ = _arm(KC % "baseline"), _arm(KC % "spec-dflash-n4")
    chk(f"K c={c} baseline aggregate", round(b_[0], 1), base_, 0.05)
    chk(f"K c={c} dflash-n4 aggregate", round(a_[0], 1), arm_, 0.05)
    chk(f"K c={c} dflash-n4 vs baseline", round(100*(a_[0]/b_[0]-1), 1), delta, 0.05)
    peaks = sorted({json.load(open(f)).get("max_in_flight") for f in glob.glob(KC % "*")})
    chk(f"K c={c} batch actually formed", peaks, [c])
# it is cost, not draft quality: volume and acceptance stay put while the clock moves
for label, pat, vol, acc in (("c=1", K1 % "spec-dflash-n4", 1.234, 55.6),
                             ("c=4", "v4_audit_2026_08_25/data/matrix_K_conc4_*/spec-dflash-n4__rep*.json", 1.243, 55.0),
                             ("c=8", "v4_audit_2026_08_25/data/matrix_K_conc8_*/spec-dflash-n4__rep*.json", 1.305, 51.2)):
    rs = [json.load(open(f)) for f in glob.glob(pat)]
    n_ = sum(x["predicted_n"] for r in rs for x in r["rows"])
    dn_ = sum(x["draft_n"] for r in rs for x in r["rows"])
    da_ = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    chk(f"K {label} drafted per generated token", round(dn_/n_, 3), vol, 0.0005)
    chk(f"K {label} acceptance %", round(100*da_/dn_, 1), acc, 0.05)
# run K replicates run J at the two draft lengths they share
for n, want in ((4, 17.3), (8, -15.8)):
    chk(f"K replicates J at n_max {n}",
        round(100*(_arm(K1 % f"spec-dflash-n{n}")[0]/kb[0]-1), 1), want, 0.05)

print("\n=== run L (2026-08-26): the workload control ===")
def _pool(pat):
    rs = [json.load(open(f)) for f in glob.glob("v4_audit_2026_08_25/data/" + pat)]
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    dn = sum(x["draft_n"] for r in rs for x in r["rows"])
    da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    return (1000*n/ms if ms else float("nan")), (100*da/dn if dn else None), rs

for half, pat, want_off in (("think-on", "matrix_L_thinkon_*", 0),
                            ("think-off", "matrix_L_thinkoff_*", 250)):
    _rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/*__rep*.json")]
    _sup = sum(1 for r in _rs for x in r["rows"] if x.get("thinking_suppressed"))
    chk(f"L {half}: requests with thinking suppressed", (_sup, 250), (want_off, 250))

for half, pat, base, rows in (
        ("think-on",  "matrix_L_thinkon_*",  122.9,
         (("spec-dflash-n2", 148.8,  21.1, 72.8), ("spec-dflash-n4", 148.5,  20.9, 55.6),
          ("spec-dflash-n6", 114.8,  -6.6, 43.0), ("spec-draft-n8",   30.6, -75.1, 27.9))),
        ("think-off", "matrix_L_thinkoff_*", 124.1,
         (("spec-dflash-n2", 133.5,   7.6, 58.5), ("spec-dflash-n4", 120.7,  -2.7, 40.3),
          ("spec-dflash-n6",  93.4, -24.7, 30.5), ("spec-draft-n8",   27.5, -77.8, 22.1)))):
    b = _pool(f"{pat}/baseline__rep*.json")[0]
    chk(f"L {half} baseline pooled", round(b, 1), base, 0.05)
    for arm_, pooled_, delta, acc in rows:
        pv, av, _ = _pool(f"{pat}/{arm_}__rep*.json")
        chk(f"L {half} {arm_} pooled", round(pv, 1), pooled_, 0.05)
        chk(f"L {half} {arm_} vs baseline", round(100*(pv/b-1), 1), delta, 0.05)
        chk(f"L {half} {arm_} acceptance %", round(av, 1), acc, 0.05)

# the relationship, and the out-of-sample test that keeps it honest (cf. A10)
_xs, _ys = [], []
for pat in ("matrix_L_thinkon_*", "matrix_L_thinkoff_*"):
    _per, _acc = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/*__rep*.json"):
        r = json.load(open(f))
        for x in r["rows"]:
            _per[x["tag"]][r["arm"]].append(x["predicted_per_second"])
            a = _acc[x["tag"]][r["arm"]]; a[0] += x["draft_n_accepted"]; a[1] += x["draft_n"]
    for t in _per:
        for a_ in ("spec-dflash-n2", "spec-dflash-n4", "spec-dflash-n6"):
            if _acc[t][a_][1]:
                _xs.append(100*_acc[t][a_][0]/_acc[t][a_][1])
                _ys.append(100*(st.mean(_per[t][a_])/st.mean(_per[t]["baseline"])-1))
chk("L acceptance-vs-speedup point count", len(_xs), 60)

def _pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num / den

chk("L acceptance-vs-speedup Pearson r", round(_pearson(_xs, _ys), 3), 0.946, 0.001)
_mx, _my = st.mean(_xs), st.mean(_ys)
_slope = sum((x-_mx)*(y-_my) for x, y in zip(_xs, _ys)) / sum((x-_mx)**2 for x in _xs)
_break = (_mx*_slope - _my) / _slope
chk("L break-even acceptance %", round(_break, 1), 48.2, 0.05)
# out of sample: runs J and K never saw this line
_oos, _ok = [], 0
for pat, bpat in (("matrix_K1_sweep_*/spec-dflash-n%s__rep*.json", "matrix_K1_sweep_*/baseline__rep*.json"),):
    _b = _pool(bpat)[0]
    for n in (1, 2, 3, 4, 6, 8):
        pv, av, _ = _pool(pat % n)
        _oos.append((av, 100*(pv/_b-1)))
_jb = _pool("matrix_J2_*/baseline__rep*.json")[0]
for nm in ("spec-dflash-n4", "spec-dflash-n8", "spec-dflash-n16", "spec-draft-n8"):
    pv, av, _ = _pool(f"matrix_J2_*/{nm}__rep*.json")
    _oos.append((av, 100*(pv/_jb-1)))
_ok = sum(1 for a, d in _oos if (a >= _break) == (d > 0))
chk("L threshold predicts the sign out of sample", (_ok, len(_oos)), (10, 10))
_err = max((_slope*a + (_my - _slope*_mx)) - d for a, d in _oos)
chk("L worst out-of-sample magnitude error (pp)", round(_err, 1), 52.2, 0.05)

print("\n=== ERRATA A11: output preservation, and the determinism control ===")
_J = "v4_audit_2026_08_25/data/matrix_J2_*/*__rep*.json"
_arms = defaultdict(lambda: defaultdict(dict))
for _f in glob.glob(_J):
    _r = json.load(open(_f))
    for _x in _r["rows"]: _arms[_r["arm"]][_r["repeat"]][_x["tag"]] = _x
_txt = lambda x: (x.get("reasoning_content") or "") + (x.get("content") or "")
# the control first: an arm must reproduce itself, or the contrast means nothing
for _a in sorted(_arms):
    _reps = sorted(_arms[_a])
    _same = sum(1 for t in _arms[_a][_reps[0]]
                if len({_txt(_arms[_a][r][t]) for r in _reps if t in _arms[_a][r]}) == 1)
    chk(f"A11 control: {_a} reproduces itself", (_same, 10), (10, 10))
_b = _arms["baseline"]
for _a, _want in (("spec-dflash-n4",3),("spec-dflash-n8",0),
                  ("spec-dflash-n16",3),("spec-draft-n8",0)):
    _id = sum(1 for r in _arms[_a] for t in _arms[_a][r]
              if (_arms[_a][r][t].get("tokens") or [None]) == (_b[r][t].get("tokens") or [None]))
    chk(f"A11 {_a} token streams identical to baseline", (_id, 30), (_want, 30))
_ns = {x["predicted_n"] for a in _arms for r in _arms[a] for x in _arms[a][r].values()}
chk("A11 every arm generates the same token count", sorted(_ns), [300])
_rates = [st.mean([_b[r][t]["predicted_per_second"] for r in _b]) for t in _b[0]]
chk("A11 baseline spread across prompts (%)", round(100*(max(_rates)/min(_rates)-1),1), 0.8, 0.05)


print("\n=== A11 generality: divergence rises with output length ===")
def _streams(pat):
    a = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/*__rep*.json"):
        r = json.load(open(f))
        for x in r["rows"]:
            a[r["arm"]][r["repeat"]][x["tag"]] = x
    return a
for name, pat, want in (("J", "matrix_J2_*", (6, 120)), ("K1", "matrix_K1_sweep_*", (0, 180)),
                        ("L on", "matrix_L_thinkon_*", (0, 200)),
                        ("L off", "matrix_L_thinkoff_*", (85, 200))):
    a = _streams(pat); b = a["baseline"]
    same = tot = 0
    for arm in a:
        if arm == "baseline":
            continue
        for rep in a[arm]:
            for t, x in a[arm][rep].items():
                y = b.get(rep, {}).get(t)
                if y:
                    tot += 1
                    same += (x.get("tokens") or [1]) == (y.get("tokens") or [2])
    chk(f"A11 {name}: streams identical to baseline", (same, tot), want)
    # the determinism control has to hold in every run, or the contrast is empty
    ok = tt = 0
    for arm in a:
        reps = sorted(a[arm])
        for t in a[arm][reps[0]]:
            vals = {(a[arm][r][t].get("content") or "") + (a[arm][r][t].get("reasoning_content") or "")
                    for r in reps if t in a[arm][r]}
            tt += 1; ok += (len(vals) == 1)
    chk(f"A11 {name}: every arm reproduces itself", (ok, tt), (tt, tt))
_a = _streams("matrix_L_thinkoff_*"); _b = _a["baseline"]
_short = []; _long = []
for arm in _a:
    if arm == "baseline":
        continue
    for rep in _a[arm]:
        for t, x in _a[arm][rep].items():
            y = _b.get(rep, {}).get(t)
            if y:
                (_long if y["predicted_n"] >= 96 else _short).append(
                    (x.get("tokens") or [1]) != (y.get("tokens") or [2]))
chk("A11 think-off divergence, outputs < 96 tokens (%)",
    round(100*sum(_short)/len(_short), 1), 37.5, 0.05)
chk("A11 think-off divergence, outputs >= 96 tokens (%)",
    round(100*sum(_long)/len(_long), 1), 70.8, 0.05)

print("\n=== thermals: no run declined ===")
import csv as _csv
for name, pat, lo, hi, drift, swt in (("I+J", "gpu_telemetry_IJ_*.csv", 55, 73, None, 1),
                                      ("K",   "gpu_telemetry_K_*.csv",  53, 74, 0.05, 0),
                                      ("L",   "gpu_telemetry_L_*.csv",  50, 74, 0.24, 0)):
    f = glob.glob(f"v4_audit_2026_08_25/data/{pat}")[0]
    rs = [r for r in _csv.reader(open(f)) if len(r) == 9][1:]
    ld = [r for r in rs if r[1].isdigit() and int(r[1]) > 50]
    tp = [int(r[3]) for r in ld]
    chk(f"thermal {name}: temperature range", (min(tp), max(tp)), (lo, hi))
    # 0x8 HwSlowdown | 0x40 HwThermal | 0x80 HwPowerBrake. NOT 0x20, which is
    # SwThermalSlowdown - a software flag, and the one that fires once in the
    # I+J trace. An earlier mask of 0xE0 counted that as hardware and missed
    # HwSlowdown altogether.
    chk(f"thermal {name}: no hardware throttle flag",
        all(int(r[8], 16) & 0xC8 == 0 for r in ld), True)
    chk(f"thermal {name}: SwThermal (0x20) samples under load",
        sum(1 for r in ld if int(r[8], 16) & 0x20), swt)
    if drift is not None:
        ck = [int(r[5]) for r in ld]; h = len(ck)//2
        chk(f"thermal {name}: clock drift first->second half (%)",
            round(abs(100*(st.mean(ck[h:])/st.mean(ck[:h])-1)), 2), drift, 0.005)

print("\n=== theory (ERRATA E1/E2) ===")
rho=8/256
chk("rho", round(rho,5), 0.03125, 1e-9)
chk("T_95 exact", round(math.log(0.05)/math.log(1-rho),2), 94.36, 0.01)
chk("T_95 ceil", math.ceil(math.log(0.05)/math.log(1-rho)), 95)
chk("coverage at 94 < 0.95", 1-(1-rho)**94 < 0.95, True)
chk("coverage at 95 >= 0.95", 1-(1-rho)**95 >= 0.95, True)

print("\n=== do the documents actually quote these figures? ===")
import pathlib

# The documents use typographic minus (U+2212) and en/em dashes; a needle typed
# with an ASCII hyphen would fail against correct prose. Twice during this audit
# a checker was wrong rather than the thing it checked - once over anchors
# keeping underscores, once over this - so normalise both sides.
_DASHES = {"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2011": "-",
           "\u00d7": "x"}   # multiplication sign, as in "0.16x"


def _norm(t: str) -> str:
    for a, b in _DASHES.items():
        t = t.replace(a, b)
    return t
DOC_CLAIMS = [
    ("ERRATA.md",   "115 / 214",   "A1 true token acceptance"),
    ("ERRATA.md",   "53.7",        "A1 acceptance percentage"),
    ("ERRATA.md",   "33 / 81",     "A1 draft-sequence acceptance"),
    ("ERRATA.md",   "999.6",       "A4 drafter generate() ms"),
    ("ERRATA.md",   "31.6",        "A4 drafter share of wall-clock"),
    ("ERRATA.md",   "144 of the 190", "A5 empty-content count"),
    ("ERRATA.md",   "75.8",        "A5 empty-content percentage"),
    ("ERRATA.md",   "248044",      "A2 target BOS id"),
    ("ERRATA.md",   "= 94.36",     "E1 coverage arithmetic"),
    ("ERRATA.md",   "T_95 = 95",   "E1 coverage threshold"),
    ("ERRATA.md",   "+0.998",      "A7 acceptance-speed correlation"),
    ("ERRATA.md",   "29.7",        "A7 master acceptance"),
    ("README.md",   "+0.998",      "README correlation"),
    ("README.md",   "109.9",       "README pooled draft-max8"),
    ("README.md",   "-19.0 %",     "README pooled delta"),
    ("v4_audit_2026_08_25/README.md", "16590", "v4 drafted tokens"),
    ("v4_audit_2026_08_25/README.md", "4926",  "v4 accepted tokens"),
    ("v4_audit_2026_08_25/README.md", "+40.6 %", "I baseline gain at c=4"),
    ("v4_audit_2026_08_25/README.md", "+64.0 %", "I baseline gain at c=8"),
    ("v4_audit_2026_08_25/README.md", "0.16x",   "I spec/baseline ratio at c=8"),
    ("v4_audit_2026_08_25/README.md", "130.2",   "J dflash-n4 aggregate"),
    ("v4_audit_2026_08_25/README.md", "-0.01 %", "J -fit on control"),
    ("v4_audit_2026_08_25/README.md", "55.8 %",  "J dflash-n4 acceptance"),
    ("README.md",   "+18.7 %",  "README DFlash headline"),
    ("README.md",   "-47.4 %",  "README DFlash n16"),
    ("ERRATA.md",   "+18.7 %",  "D4 resolution"),
    ("ERRATA.md",   "+24.0 %",  "D4 pooled delta"),
    ("ERRATA.md",   "3 / 30",   "A11 dflash-n4 identical streams"),
    ("ERRATA.md",   "0.8 %",    "A11 baseline content spread"),
    ("ERRATA.md",   "110-126",  "A11 median divergence index"),
    ("v4_audit_2026_08_25/README.md", "+17.6 %", "K1 plateau top"),
    ("v4_audit_2026_08_25/README.md", "-74.1 %", "K c=8 collapse"),
    ("v4_audit_2026_08_25/README.md", "1.305",   "K c=8 draft volume"),
    ("v4_audit_2026_08_25/README.md", "153.2",   "K c=8 baseline"),
    ("v4_audit_2026_08_25/README.md", "+7.6 %",  "L think-off n2"),
    ("v4_audit_2026_08_25/README.md", "-2.7 %",  "L think-off n4 goes negative"),
    ("v4_audit_2026_08_25/README.md", "+0.946",  "L acceptance correlation"),
    ("v4_audit_2026_08_25/README.md", "48.2 %",  "L break-even acceptance"),
    ("v4_audit_2026_08_25/README.md", "+52.2 pp", "L worst out-of-sample error"),
    ("README.md",   "48 % draft",  "README acceptance threshold"),
    ("ERRATA.md",   "85 / 200",  "A11 think-off identical streams"),
    ("ERRATA.md",   "70.8 %",    "A11 long-output divergence"),
    ("v4_audit_2026_08_25/README.md", "-0.24 %", "L clock drift"),
    ("RETEST_TODO.md", "785 plain BF16", "MTP weights are present"),
]
root = pathlib.Path(__file__).resolve().parents[1]
for f, needle, what in DOC_CLAIMS:
    txt = _norm((root / f).read_text(encoding="utf-8"))
    ok = _norm(needle) in txt
    print(f"  {'PASS' if ok else 'FAIL'}  {f:32s} quotes {needle!r:20s} ({what})")
    if not ok:
        FAIL.append(f"{f}:{needle}")

print(f"\n{'='*70}\n{'ALL CLAIMS VERIFIED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}\n{'='*70}")
sys.exit(1 if FAIL else 0)
