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
import csv, hashlib, json, glob, math, os, pathlib, re, statistics as st
from collections import Counter, defaultdict

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
_b_drafted = sum(x["draft_n"] for r in B["draft-max8-matched"] for x in r["rows"])
_b_acc = sum(x["draft_n_accepted"] for r in B["draft-max8-matched"] for x in r["rows"])
chk("B drafted tokens", _b_drafted, 16590)
chk("B accepted tokens", _b_acc, 4926)
chk("B acceptance %", round(100*_b_acc/_b_drafted, 1), 29.7, 0.05)
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
    # assert the source is non-empty first: all() over nothing is True, and a
    # check that certifies an empty file is worse than no check
    chk(f"thermal {name}: loaded samples to check", len(ld) > 100, True)
    chk(f"thermal {name}: no hardware throttle flag",
        all(int(r[8], 16) & 0xC8 == 0 for r in ld), True)
    chk(f"thermal {name}: SwThermal (0x20) samples under load",
        sum(1 for r in ld if int(r[8], 16) & 0x20), swt)
    if drift is not None:
        ck = [int(r[5]) for r in ld]; h = len(ck)//2
        chk(f"thermal {name}: clock drift first->second half (%)",
            round(abs(100*(st.mean(ck[h:])/st.mean(ck[:h])-1)), 2), drift, 0.005)

print("\n=== runs M, N and O (2026-08-26) ===")
def _agg(pat, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/{arm}__rep*.json")]
    rs = [r for r in rs if r.get("rows")]
    if not rs: return None
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    dn = sum(x["draft_n"] for r in rs for x in r["rows"])
    da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    a = [r["aggregate_tok_s"] for r in rs if r.get("wall_s")]
    return {"pooled": 1000*n/ms, "agg": st.mean(a), "acc": (100*da/dn) if dn else None,
            "drafted": dn, "reps": len(rs)}

# M1: both self-speculative families under one policy
_b = _agg("matrix_M1_*", "baseline")
chk("M1 baseline aggregate", round(_b["agg"], 1), 103.3, 0.05)
for arm, agg_, delta, acc in (("spec-mtp-n1", 118.4,  14.6, 89.0),
                              ("spec-mtp-n2", 122.5,  18.6, 78.4),
                              ("spec-mtp-n4", 111.5,   8.0, 61.4),
                              ("spec-mtp-n8",  71.4, -30.9, 41.4),
                              ("spec-dflash-n2", 127.3, 23.2, 72.3),
                              ("spec-dflash-n4", 119.9, 16.1, 55.2)):
    v = _agg("matrix_M1_*", arm)
    chk(f"M1 {arm} aggregate", round(v["agg"], 1), agg_, 0.05)
    chk(f"M1 {arm} vs baseline", round(100*(v["agg"]/_b["agg"]-1), 1), delta, 0.05)
    chk(f"M1 {arm} acceptance %", round(v["acc"], 1), acc, 0.05)
chk("M1 DFlash beats MTP at the same draft length",
    _agg("matrix_M1_*", "spec-dflash-n2")["agg"] > _agg("matrix_M1_*", "spec-mtp-n2")["agg"], True)

# M4: the drafter-precision objection
_b4 = _agg("matrix_M4_q4km_*", "baseline")
for arm, delta, acc in (("spec-mtp-n2", 22.3, 79.4), ("spec-mtp-n4", 1.2, 60.6)):
    v = _agg("matrix_M4_q4km_*", arm)
    chk(f"M4 Q4_K_M {arm} vs baseline", round(100*(v["agg"]/_b4["agg"]-1), 1), delta, 0.05)
    chk(f"M4 Q4_K_M {arm} acceptance %", round(v["acc"], 1), acc, 0.05)
chk("M4 the more quantised drafter is FASTER at n_max 2",
    100*(_agg("matrix_M4_q4km_*","spec-mtp-n2")["agg"]/_b4["agg"]-1) >
    100*(_agg("matrix_M1_*","spec-mtp-n2")["agg"]/_b["agg"]-1), True)

# M3: thinking off, pooled (outputs differ in length by arm - A11)
_b3 = _agg("matrix_M3_thinkoff_*", "baseline")
chk("M3 repeats", _b3["reps"], 5)
for arm, delta, acc in (("spec-mtp-n2", 11.4, 67.5), ("spec-mtp-n4", -8.2, 49.5),
                        ("spec-dflash-n2", 8.5, 58.4)):
    v = _agg("matrix_M3_thinkoff_*", arm)
    chk(f"M3 think-off {arm} pooled vs baseline", round(100*(v["pooled"]/_b3["pooled"]-1), 1), delta, 0.05)
    chk(f"M3 think-off {arm} acceptance %", round(v["acc"], 1), acc, 0.05)
chk("M3 the DFlash/MTP ranking flips with thinking off",
    _agg("matrix_M3_thinkoff_*","spec-mtp-n2")["pooled"] >
    _agg("matrix_M3_thinkoff_*","spec-dflash-n2")["pooled"], True)

# M2: MTP degrades gracefully where DFlash collapses
for c, delta in ((4, 3.4), (8, -7.6)):
    bb = _agg(f"matrix_M2_conc{c}_*", "baseline"); vv = _agg(f"matrix_M2_conc{c}_*", "spec-mtp-n2")
    chk(f"M2 c={c} mtp-n2 vs baseline", round(100*(vv["agg"]/bb["agg"]-1), 1), delta, 0.05)
    peaks = sorted({json.load(open(f)).get("max_in_flight")
                    for f in glob.glob(f"v4_audit_2026_08_25/data/matrix_M2_conc{c}_*/*__rep*.json")})
    chk(f"M2 c={c} batch actually formed", peaks, [c])
_mtp_c8 = 100*(_agg("matrix_M2_conc8_*", "spec-mtp-n2")["agg"] /
               _agg("matrix_M2_conc8_*", "baseline")["agg"] - 1)
_dfl_c8 = 100*(_agg("matrix_K_conc8_*", "spec-dflash-n4")["agg"] /
               _agg("matrix_K_conc8_*", "baseline")["agg"] - 1)
chk("M2 MTP at c=8 loses far less than DFlash did", _mtp_c8 > _dfl_c8 + 50, True)

# N: the ngram-map arms never engage
_bn = _agg("matrix_N_ngrammap_*", "baseline")
for arm, drafted in (("ngram-map-k", 144), ("ngram-map-k-m8", 24), ("ngram-map-k-m4", 12),
                     ("ngram-map-k4v", 144), ("ngram-map-k4v-m8", 24), ("ngram-map-k4v-m4", 12)):
    v = _agg("matrix_N_ngrammap_*", arm)
    chk(f"N {arm} draft tokens over 30 requests", v["drafted"], drafted)
    chk(f"N {arm} acceptance %", round(v["acc"], 1), 0.0, 0.05)
    chk(f"N {arm} within 2 % of baseline", abs(100*(v["agg"]/_bn["agg"]-1)) < 2.0, True)

# O: the head-to-head, and the threshold falsified inside one matrix
_bo = _agg("matrix_O_headtohead_*", "baseline")
chk("O baseline pooled", round(_bo["pooled"], 1), 117.0, 0.05)
# both metrics, each checked against its own baseline: the first version of the
# README table put pooled values beside aggregate deltas
for arm, pooled_, dp, agg_, da_, acc in (
        ("spec-dflash-n2",  145.8,  24.6, 126.6,  21.1, 72.3),
        ("spec-mtp-n2",     142.5,  21.8, 122.8,  17.5, 78.4),
        ("spec-dflash-n4",  138.1,  18.0, 119.6,  14.5, 55.2),
        ("ngram-map-k4v-m8",116.1,  -0.8, 103.8,  -0.7, 50.0),
        ("ngram-mod-n24",   103.4, -11.7,  93.5, -10.5,  5.0),
        ("ngram-cache",      93.9, -19.7,  85.9, -17.8,  5.2),
        ("spec-draft-n8",    30.8, -73.7,  29.8, -71.5, 29.5),
        ("spec-draft-n1",    29.1, -75.1,  28.2, -73.0, 69.7)):
    v = _agg("matrix_O_headtohead_*", arm)
    chk(f"O {arm} pooled", round(v["pooled"], 1), pooled_, 0.05)
    chk(f"O {arm} delta pooled", round(100*(v["pooled"]/_bo["pooled"]-1), 1), dp, 0.05)
    chk(f"O {arm} aggregate", round(v["agg"], 1), agg_, 0.05)
    chk(f"O {arm} delta aggregate", round(100*(v["agg"]/_bo["agg"]-1), 1), da_, 0.05)
    chk(f"O {arm} acceptance %", round(v["acc"], 1), acc, 0.05)
_o1 = _agg("matrix_O_headtohead_*", "spec-draft-n1")
_od = _agg("matrix_O_headtohead_*", "spec-dflash-n2")
chk("O: spec-draft-n1 has HIGHER acceptance than a winning arm", _o1["acc"] > 48.2, True)
chk("O: and is still slower than no speculation", _o1["agg"] < _bo["agg"], True)
chk("O: the threshold is falsified inside one matrix",
    (_o1["acc"] > 48.2) and (_o1["agg"] < _bo["agg"]), True)
chk("O: v1's three methods are the three worst rows",
    sorted([("spec-draft-n1", _o1["pooled"]), ("spec-draft-n8", _agg("matrix_O_headtohead_*","spec-draft-n8")["pooled"]),
            ("ngram-cache", _agg("matrix_O_headtohead_*","ngram-cache")["pooled"])],
           key=lambda t: t[1])[0][0], "spec-draft-n1")

print("\n=== ERRATA A12: checkpoint activity, corrected ===")
_sa = {(r["arm"], r["run"]): r for r in
       json.load(open("v4_audit_2026_08_25/data/spec_accounting_20260826.json"))}
_by_arm = defaultdict(list)
for (a, _run), r in _sa.items():
    _by_arm[a].append(r)
_ext = [r for rs in _by_arm.values() for r in rs if r.get("spec_type") == "draft-simple"]
chk("A12 external-drafter arm-runs extracted", len(_ext) >= 1, True)
_e = _ext[0]
chk("A12 external drafter: checkpoints created", _e["checkpoints_created"], 772)
chk("A12 external drafter: checkpoints restored", _e["checkpoints_restored"], 709)
# size() is data_tgt + data_dft + data_spec, so the logged size is the TOTAL and
# the logged draft figure is a component of it. Adding them double-counts.
chk("A12 logged checkpoint total (MiB)", round(_e["checkpoint_total_mib"], 3), 82.079, 0.0005)
chk("A12 the draft component is part of that total (MiB)",
    round(_e["checkpoint_draft_component_mib"], 3), 19.266, 0.0005)
chk("A12 the draft component is smaller than the total",
    _e["checkpoint_draft_component_mib"] < _e["checkpoint_total_mib"], True)
chk("A12 nominal volume written (GiB)", _e["nominal_state_written_gib"], 61.88, 0.005)
chk("A12 nominal volume read back (GiB)", _e["nominal_state_read_back_gib"], 56.83, 0.005)
chk("A12 nominal volume combined (GiB)", _e["nominal_state_total_gib"], 118.71, 0.005)
chk("A12 writes equal count x LOGGED TOTAL, with no draft added again (GiB)",
    round(_e["checkpoints_created"] * _e["checkpoint_total_mib"] / 1024, 2), 61.88, 0.005)
chk("A12 the invalid wall-clock share is gone from the data",
    "checkpoint_share_pct" in _e, False)
chk("A12 external drafter: generate() seconds", _e["drafter_generate_s"], 17.24, 0.005)
_self = [r for rs in _by_arm.values() for r in rs
         if r.get("spec_type") in ("draft-dflash", "draft-mtp")]
chk("A12 self-speculative arm-runs extracted", len(_self) >= 8, True)
chk("A12 self-speculative: checkpoint events logged, all arms",
    sorted({r["checkpoints_created"] for r in _self} | {r["checkpoints_restored"] for r in _self}), [0])
# per family, not the union of the two - the union hid that MTP covers only 1/2/8
_fam = defaultdict(set)
for r in _self:
    _fam[r["spec_type"]].add(int(r["arm"].rsplit("n", 1)[1]))
chk("A12 DFlash draft lengths with zero checkpoint events",
    sorted(_fam["draft-dflash"]), [1, 2, 4, 6, 8, 16])
chk("A12 MTP draft lengths with zero checkpoint events",
    sorted(_fam["draft-mtp"]), [1, 2, 8])

print("\n=== the acceptance threshold, tested per drafter family ===")
def _pa(pat, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/{arm}__rep*.json")]
    if not rs: return None
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    dn = sum(x["draft_n"] for r in rs for x in r["rows"])
    da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    return (1000*n/ms, (100*da/dn) if dn else None)
BREAK = 48.2
_fam = defaultdict(lambda: [0, 0])
def _score(fam, pat, arms):
    b = _pa(pat, "baseline")
    for a in arms:
        v = _pa(pat, a)
        if not v or v[1] is None: continue
        d = 100*(v[0]/b[0]-1)
        _fam[fam][0] += ((v[1] >= BREAK) == (d > 0)); _fam[fam][1] += 1
_score("external", "C_master_matrix_think_on",
       [f"spec-draft-n{n}" for n in (1, 2, 4, 8, 16, 32)])
_score("external", "matrix_J2_*", ["spec-draft-n8"])
_score("external", "matrix_L_thinkon_*", ["spec-draft-n8"])
_score("dflash", "matrix_K1_sweep_*", [f"spec-dflash-n{n}" for n in (1, 2, 3, 4, 6, 8)])
_score("dflash", "matrix_J2_*", ["spec-dflash-n4", "spec-dflash-n8", "spec-dflash-n16"])
_score("dflash", "matrix_L_thinkon_*", ["spec-dflash-n2", "spec-dflash-n4", "spec-dflash-n6"])
_score("ngram", "C_master_matrix_think_on", ["ngram-cache", "ngram-mod-n24", "ngram-simple"])
chk("threshold: sign correct within DFlash", tuple(_fam["dflash"]), (12, 12))
chk("threshold: sign correct on drafter-free ngram", tuple(_fam["ngram"]), (3, 3))
chk("threshold: sign correct on the external drafter", tuple(_fam["external"]), (6, 8))
chk("threshold: sign correct overall",
    (sum(v[0] for v in _fam.values()), sum(v[1] for v in _fam.values())), (21, 23))
# the two failures, named
_b = _pa("C_master_matrix_think_on", "baseline")
for n, acc, delta in ((1, 68.7, -74.8), (2, 60.3, -72.2)):
    v = _pa("C_master_matrix_think_on", f"spec-draft-n{n}")
    chk(f"threshold failure: spec-draft-n{n} acceptance (%)", round(v[1], 1), acc, 0.05)
    chk(f"threshold failure: spec-draft-n{n} vs baseline (%)",
        round(100*(v[0]/_b[0]-1), 1), delta, 0.05)

print("\n=== ERRATA A13: the two acceptance counters ===")
_cc = json.load(open("v4_audit_2026_08_25/data/acceptance_counter_comparison.json"))
_seq = [r for r in _cc if "conc4" not in r["run"] and "conc8" not in r["run"]]
chk("A13 single-request arm-runs with both counters", len(_seq), 517)
_z = [r for r in _seq if r["checkpoints_created"] == 0]
_n = [r for r in _seq if r["checkpoints_created"] > 0]
chk("A13 arm-runs that take no checkpoint", len(_z), 248)
chk("A13 arm-runs that do", len(_n), 269)
chk("A13 no-checkpoint: largest gap between the counters (pp)",
    round(max(abs(r["server_pct"] - r["drafter_pct"]) for r in _z), 2), 0.80, 0.005)
chk("A13 checkpointing: smallest gap between the counters (pp)",
    round(min(abs(r["server_pct"] - r["drafter_pct"]) for r in _n), 2), 1.00, 0.005)
chk("A13 every repeat is present, not just rep0",
    sorted({r.get("repeat") for r in _cc if r["run"].startswith("matrix_O2")}),
    [0, 1, 2, 3, 4, 5, 6, 7, 8])
chk("A13 the two groups do not overlap",
    max(abs(r["server_pct"] - r["drafter_pct"]) for r in _z) <
    min(abs(r["server_pct"] - r["drafter_pct"]) for r in _n), True)
_pick = lambda a: next(r for r in _seq if r["arm"] == a)
for arm, srv, drf, ck in (("spec-dflash-n2", 72.8, 73.0, 0),
                          ("spec-mtp-n2",    78.4, 78.6, 0),
                          ("spec-draft-n8",  29.7, 41.3, 772),
                          ("ngram-cache",     1.8, 19.1, 236),
                          ("ngram-map-k4v-m8", 0.0, 53.3, 2),
                          ("spec-draft-n1",  68.7, 100.0, 1639)):
    r = _pick(arm)
    chk(f"A13 {arm} server counter (%)", r["server_pct"], srv, 0.05)
    chk(f"A13 {arm} drafter counter (%)", r["drafter_pct"], drf, 0.05)
    chk(f"A13 {arm} checkpoints", r["checkpoints_created"], ck)
# the row that discredits BOTH counters
chk("A13 spec-draft-n1 drafter counter is exactly 1.0",
    _pick("spec-draft-n1")["drafter_drafted"] == _pick("spec-draft-n1")["drafter_accepted"], True)
# ngram-map barely fires at all, on the one quantity no counter can distort
for arm in ("ngram-map-k", "ngram-map-k4v"):
    r = _pick(arm)
    chk(f"A13 {arm} generate() calls", r["drafter_calls_generate"], 3271)
    chk(f"A13 {arm} drafts actually produced", r["drafter_drafts"], 2)

print("\n=== runs P, Q and R, and A14 ===")
def _pl(pat, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/{arm}__rep*.json")]
    rs = [r for r in rs if r.get("rows")]
    if not rs: return None
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    pm = sum(x["timings"].get("prompt_ms", 0) for r in rs for x in r["rows"])
    pn = sum(x["timings"].get("prompt_n", 0) for r in rs for x in r["rows"])
    ag = [r["aggregate_tok_s"] for r in rs if r.get("wall_s")]
    return {"pooled": 1000*n/ms, "agg": st.mean(ag), "prompt_ms": pm, "prompt_n": pn,
            "dec_ms": ms, "reps": len(rs), "nprompt": len(rs[0]["rows"])}

# the extended set exists and is genuinely different
_pb = _pl("matrix_P_extended_*", "baseline")
chk("P prompt count", _pb["nprompt"], 20)
chk("P prompt tokens", _pb["prompt_n"], 4323)
_ob = _pl("matrix_O_headtohead_*", "baseline")
chk("O prompt tokens", _ob["prompt_n"], 1035)
for lbl, b, want in (("v1 ten", _ob, 6.7), ("extended twenty", _pb, 12.7)):
    chk(f"prompt processing share, {lbl} (%)",
        round(100*b["prompt_ms"]/(b["prompt_ms"]+b["dec_ms"]), 1), want, 0.05)

# the generalisation test, on the metric that is comparable across prompt sets
for arm, d1, d2 in (("spec-dflash-n2", 24.6, 20.3), ("spec-dflash-n4", 18.0, 21.3),
                    ("spec-mtp-n2", 21.8, 21.0), ("spec-draft-n8", -73.7, -73.1)):
    chk(f"P {arm} pooled delta, v1 ten", round(100*(_pl("matrix_O_headtohead_*", arm)["pooled"]/_ob["pooled"]-1), 1), d1, 0.05)
    chk(f"P {arm} pooled delta, extended", round(100*(_pl("matrix_P_extended_*", arm)["pooled"]/_pb["pooled"]-1), 1), d2, 0.05)
    chk(f"P {arm} shift across prompt sets under 5 pp", abs(d2-d1) < 5.0, True)

# R: the workload control repeats on the new set
_rb = _pl("matrix_R_ext_thinkoff_*", "baseline")
_m3 = _pl("matrix_M3_thinkoff_*", "baseline")
for arm, v1_, ex_ in (("spec-dflash-n2", 8.5, 8.6), ("spec-mtp-n2", 11.4, 13.2)):
    chk(f"R {arm} think-off, v1 ten", round(100*(_pl("matrix_M3_thinkoff_*", arm)["pooled"]/_m3["pooled"]-1), 1), v1_, 0.05)
    chk(f"R {arm} think-off, extended", round(100*(_pl("matrix_R_ext_thinkoff_*", arm)["pooled"]/_rb["pooled"]-1), 1), ex_, 0.05)

# Q: the anomaly was one measurement that did not replicate
for pat, arm, want, reps in (("matrix_M1_*", "spec-mtp-n2", 22.1, 3),
                             ("matrix_Q_q8_*", "spec-mtp-n2", 21.6, 5),
                             ("matrix_M4_q4km_*", "spec-mtp-n2", 26.6, 3),
                             ("matrix_Q_q4km_*", "spec-mtp-n2", 27.0, 5),
                             ("matrix_M1_*", "spec-mtp-n4", 10.5, 3),
                             ("matrix_Q_q8_*", "spec-mtp-n4", 2.0, 5),
                             ("matrix_M4_q4km_*", "spec-mtp-n4", 3.6, 3),
                             ("matrix_Q_q4km_*", "spec-mtp-n4", 3.6, 5),
                             ("matrix_P_extended_*", "spec-mtp-n4", 2.7, 3)):
    b = _pl(pat, "baseline"); v = _pl(pat, arm)
    chk(f"Q {pat.rstrip('_*')} {arm} pooled delta", round(100*(v["pooled"]/b["pooled"]-1), 1), want, 0.05)
    chk(f"Q {pat.rstrip('_*')} {arm} repeats", v["reps"], reps)
chk("Q the Q4_K_M head beats Q8_0 at BOTH draft lengths at five repeats",
    (round(100*(_pl("matrix_Q_q4km_*","spec-mtp-n2")["pooled"]/_pl("matrix_Q_q4km_*","baseline")["pooled"]-1),1) >
     round(100*(_pl("matrix_Q_q8_*","spec-mtp-n2")["pooled"]/_pl("matrix_Q_q8_*","baseline")["pooled"]-1),1)) and
    (round(100*(_pl("matrix_Q_q4km_*","spec-mtp-n4")["pooled"]/_pl("matrix_Q_q4km_*","baseline")["pooled"]-1),1) >
     round(100*(_pl("matrix_Q_q8_*","spec-mtp-n4")["pooled"]/_pl("matrix_Q_q8_*","baseline")["pooled"]-1),1)), True)
def _d(pat, arm="spec-mtp-n4"):
    b = _pl(pat, "baseline"); v = _pl(pat, arm)
    return round(100*(v["pooled"]/b["pooled"] - 1), 1)
_m1, _q8, _pe = _d("matrix_M1_*"), _d("matrix_Q_q8_*"), _d("matrix_P_extended_*")
chk("A14 the non-replicating pair, gap (pp)", round(_m1 - _q8, 1), 8.5, 0.05)
chk("A14 spec-mtp-n4 Q8_0 outlier is M1, not run Q",
    max([(_m1, "M1"), (_q8, "Q"), (_pe, "P")])[1], "M1")
chk("A14 the other two cluster within 1 pp", abs(_q8 - _pe) < 1.0, True)

print("\n=== A14: the thermal figures the 'unexplained' claim rests on ===")
import csv as _csv2
import subprocess as _sp2
import sys as _sys2, re as _re2
def _tel(pat, lo, hi):
    out = []
    for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}"):
        for r in _csv2.reader(open(f)):
            if len(r) != 9 or not r[1].isdigit():
                continue
            m = _re2.search(r"(\d{2}):(\d{2}):(\d{2})", r[0])
            if m and lo <= m.group(0) <= hi and int(r[1]) > 50:
                out.append((int(r[3]), int(r[5])))
    return out
for lbl, pat, lo, hi, t_, c_ in (("M1", "gpu_telemetry_M_*.csv", "08:00:00", "08:15:00", 66.1, 1938),
                                 ("Q",  "gpu_telemetry_chain2_*.csv", "11:33:00", "11:52:00", 65.8, 1934)):
    v = _tel(pat, lo, hi)
    chk(f"A14 {lbl} mean temperature under load (C)", round(st.mean([x[0] for x in v]), 1), t_, 0.05)
    chk(f"A14 {lbl} mean SM clock under load (MHz)", round(st.mean([x[1] for x in v])), c_, 0.5)
_w1 = _tel("gpu_telemetry_M_*.csv", "08:00:00", "08:15:00")
_w2 = _tel("gpu_telemetry_chain2_*.csv", "11:33:00", "11:52:00")
_t1, _t2 = st.mean([x[0] for x in _w1]), st.mean([x[0] for x in _w2])
_c1, _c2 = st.mean([x[1] for x in _w1]), st.mean([x[1] for x in _w2])
chk("A14 the two windows differ by under 1 C", abs(_t1 - _t2) < 1.0, True)
chk("A14 the two windows differ by under 0.5 % in clock", abs(_c1 - _c2)/_c1 < 0.005, True)
# every request in both prompt sets hits the cap, so the sets differ only in prompt length
for pat in ("matrix_O_headtohead_*", "matrix_P_extended_*"):
    fr = {x.get("finish_reason") for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/baseline__rep*.json")
          for x in json.load(open(f))["rows"]}
    chk(f"{pat.rstrip('_*')} every request reaches the token cap", sorted(fr), ["length"])
# the new set does not inflate acceptance
for arm, v1_, ex_ in (("spec-dflash-n2", 72.3, 72.8), ("spec-mtp-n2", 78.4, 77.3)):
    def _acc(pat):
        rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/{arm}__rep*.json")]
        dn = sum(x["draft_n"] for r in rs for x in r["rows"]); da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
        return 100*da/dn
    chk(f"{arm} acceptance, v1 ten", round(_acc("matrix_O_headtohead_*"), 1), v1_, 0.05)
    chk(f"{arm} acceptance, extended", round(_acc("matrix_P_extended_*"), 1), ex_, 0.05)
    chk(f"{arm} the new prompt set does not inflate acceptance", abs(ex_ - v1_) < 1.5, True)

print("\n=== the acceptance threshold, scored over everything, both counters ===")
# Aggregated across repeats, not keyed by (run, arm). Keyed, the dict kept
# whichever repeat the file listed last - which was harmless while the source
# held one row per arm and became an arbitrary choice the moment it held nine.
# The throughput below is pooled over the same repeats, so the counters are too.
_cc2: dict = {}
for _r in json.load(open("v4_audit_2026_08_25/data/acceptance_counter_comparison.json")):
    _k = (_r["run"], _r["arm"])
    _acc2 = _cc2.setdefault(_k, {"spec_type": _r["spec_type"], "server_drafted": 0,
                                 "server_accepted": 0, "drafter_drafted": 0,
                                 "drafter_accepted": 0, "checkpoints_created": 0,
                                 "repeats": 0})
    for _f2 in ("server_drafted", "server_accepted", "drafter_drafted",
                "drafter_accepted", "checkpoints_created"):
        _acc2[_f2] += _r[_f2]
    _acc2["repeats"] += 1
for _k, _v2 in _cc2.items():
    _v2["server_pct"] = (round(100 * _v2["server_accepted"] / _v2["server_drafted"], 1)
                         if _v2["server_drafted"] else None)
    _v2["drafter_pct"] = (round(100 * _v2["drafter_accepted"] / _v2["drafter_drafted"], 1)
                          if _v2["drafter_drafted"] else None)
def _pl2(run, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{run}/{arm}__rep*.json")]
    rs = [r for r in rs if r.get("rows")]
    if not rs: return None
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    return 1000*n/ms
_BRK = 48.2
_all, _kept = [], []
for (run, arm), r in _cc2.items():
    if "conc4" in run or "conc8" in run: continue
    b, v = _pl2(run, "baseline"), _pl2(run, arm)
    if not b or not v: continue
    fam = ("external" if r["spec_type"] == "draft-simple"
           else "self" if r["spec_type"] in ("draft-dflash", "draft-mtp") else "ngram")
    rec = (fam, run, arm, r["server_pct"], r["drafter_pct"], 100*(v/b-1), r["drafter_drafted"])
    _all.append(rec)
    if r["drafter_drafted"] >= 100: _kept.append(rec)
chk("threshold: (run, arm) pairs with an acceptance figure and a baseline", len(_all), 90)
chk("threshold: excluded for drafting under 100 tokens", len(_all) - len(_kept), 4)
chk("threshold: the excluded ones drafted at most this many tokens",
    max(x[6] for x in _all if x[6] < 100), 45)
chk("threshold: the kept ones drafted at least this many",
    min(x[6] for x in _kept), 132)
def _sc(rows, idx):
    from collections import defaultdict as _dd
    f = _dd(lambda: [0, 0])
    for x in rows:
        f[x[0]][0] += (x[idx] >= _BRK) == (x[5] > 0); f[x[0]][1] += 1
    return f
for idx, lbl in ((3, "server"), (4, "drafter")):
    f = _sc(_kept, idx)
    chk(f"threshold ({lbl}): self-speculative", tuple(f["self"]),
        (57, 59) if lbl == "server" else (58, 59))
    chk(f"threshold ({lbl}): drafter-free n-gram", tuple(f["ngram"]), (8, 11))
    chk(f"threshold ({lbl}): external drafter", tuple(f["external"]), (13, 16))
    chk(f"threshold ({lbl}): overall",
        (sum(v[0] for v in f.values()), sum(v[1] for v in f.values())),
        (78, 86) if lbl == "server" else (79, 86))
# without the exclusion the two counters disagree, which is why it exists
chk("threshold: without the exclusion the counters disagree",
    (sum(v[0] for v in _sc(_all, 3).values()), sum(v[0] for v in _sc(_all, 4).values())),
    (82, 79))
# and the two misses are the ones named in the text
_miss = sorted(f"{x[2]}" for x in _kept if (x[3] >= _BRK) != (x[5] > 0))
chk("threshold: which arms it gets wrong", sorted(set(_miss)),
    ["ngram-map-k4v-m8", "spec-dflash-n4", "spec-draft-n1", "spec-mtp-n4"])
chk("threshold: how many misses", len(_miss), 8)
# five of the eight sit within 2 pp of the boundary, where a threshold cannot
# be informative; the other three are spec-draft-n1 in three different runs
chk("threshold: misses that are spec-draft-n1",
    sum(1 for m in _miss if m == "spec-draft-n1"), 3)
chk("threshold: the rest are all within 2 pp of the boundary",
    sorted({abs(x[3] - _BRK) <= 2.0 for x in _kept
            if (x[3] >= _BRK) != (x[5] > 0) and x[2] != "spec-draft-n1"}), [True])
_m = [x for x in _kept if x[2] == "spec-mtp-n4" and (x[3] >= _BRK) != (x[5] > 0)][0]
chk("threshold: the near-boundary miss sits just above it (pp)",
    round(_m[3] - _BRK, 1), 1.3, 0.05)

print("\n=== coverage debt: emphasised figures the checker had never touched ===")
def _armstat(pat, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"v4_audit_2026_08_25/data/{pat}/{arm}__rep*.json")]
    rs = [r for r in rs if r.get("rows")]
    if not rs: return None
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    dn = sum(x["draft_n"] for r in rs for x in r["rows"])
    da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    return {"pooled": 1000*n/ms, "acc": (100*da/dn) if dn else None,
            "gen": n, "drafted": dn, "vol": dn/n}

# --- ERRATA A10 / run H: the p_min sweep the falsification rests on ---------
_h = _armstat("H_pmin_sweep", "spec-draft-n8")
_hb = _armstat("H_pmin_sweep", "baseline")
for arm, acc_, pooled_, vol_ in (("spec-draft-n8-pmin75", 80.2, 42.8, 0.61),
                                 ("spec-draft-n8-pmin90", 88.2, 42.5, 0.46)):
    v = _armstat("H_pmin_sweep", arm)
    chk(f"H {arm} acceptance %", round(v["acc"], 1), acc_, 0.05)
    chk(f"H {arm} pooled", round(v["pooled"], 1), pooled_, 0.05)
    chk(f"H {arm} draft tokens per generated token", round(v["vol"], 2), vol_, 0.005)
    chk(f"H {arm} vs baseline (%)", round(100*(v["pooled"]/_hb["pooled"]-1), 1),
        -65.5 if "75" in arm else -65.6, 0.05)
_n1 = _armstat("H_pmin_sweep", "spec-draft-n1")
if _n1:
    chk("A10 matched-volume pair: n_max 1 p_min 0, draft per token",
        round(_n1["vol"], 2), 0.50, 0.005)

# --- run E, past the MoESD coverage threshold -------------------------------
_eb = _armstat("E_past_threshold", "baseline")
for arm, want in (("spec-draft-n64", None), ("spec-draft-n96", None), ("spec-draft-n128", None)):
    v = _armstat("E_past_threshold", arm)
    if not v or not _eb: continue
    d = 100*(v["pooled"]/_eb["pooled"]-1)
    chk(f"E {arm} is far below baseline", d < -85.0, True)
    chk(f"E {arm} drafts more than it generates", v["vol"] > 5.0, True)
chk("E: throughput keeps falling past the 95-token coverage threshold",
    _armstat("E_past_threshold", "spec-draft-n128")["pooled"] <
    _armstat("E_past_threshold", "spec-draft-n64")["pooled"], True)

# --- ERRATA B7: the fp16-KV n-gram control ----------------------------------
_c1 = _armstat("C_master_matrix_think_on", "ngram-cache")
_c2 = _armstat("C_master_matrix_think_on", "ngram-cache-kvfp16")
chk("B7 ngram-cache-kvfp16 vs ngram-cache (%)",
    round(100*(_c2["pooled"]/_c1["pooled"]-1), 1), -4.2, 0.05)

# --- ERRATA A13: the worst drafter-counter reading --------------------------
_ccj = {(r["run"], r["arm"]): r for r in
        json.load(open("v4_audit_2026_08_25/data/acceptance_counter_comparison.json"))}
_worst = max(_ccj.values(), key=lambda r: r["drafter_pct"] if r["server_pct"] == 0.0 else -1)
chk("A13 highest drafter reading among arms the server calls 0.0 %",
    round(_worst["drafter_pct"], 1), 70.0, 0.05)

# --- ERRATA A1 round accounting from the archived verbose log ---------------
_va = json.load(open("analysis/verbose_accounting.json"))[0]
chk("A1 verification attempts", _va["verification_attempts"], 53)
chk("A1 partially accepted and redone", _va["attempts_partially_accepted"], 20)
chk("A1 fraction of rounds thrown away (%)",
    round(100*_va["attempts_partially_accepted"]/_va["verification_attempts"], 1), 37.7, 0.05)

# --- ERRATA C4b: the single sw_thermal sample -------------------------------
import csv as _csv3
_c4 = list(_csv3.DictReader(open("v4_audit_2026_08_25/data/gpu_telemetry_20260825.csv")))
def _a(r, k): return (r.get(k) or "").strip() == "Active"
def _n(r, k):
    v = (r.get(k) or "").replace("MHz", "").replace("W", "").replace("%", "").replace("MiB", "").strip()
    try: return float(v)
    except Exception: return None
_c4l = [r for r in _c4 if not _a(r, "thr_gpu_idle") and (_n(r, "util_pct") or 0) >= 50]
chk("C4b rows in the committed trace", len(_c4), 1317)
chk("C4b rows under load", len(_c4l), 1272)
_ct = [_n(r, "temp_c") for r in _c4l]; _cg = [_n(r, "gfx_mhz") for r in _c4l]
chk("C4b temperature range", (int(min(_ct)), int(max(_ct))), (58, 75))
chk("C4b mean temperature", round(st.mean(_ct), 1), 64.7, 0.05)
chk("C4b clock range", (int(min(_cg)), int(max(_cg))), (1800, 1965))
chk("C4b mean clock", round(st.mean(_cg)), 1937, 0.5)
chk("C4b sw_power_cap samples", sum(1 for r in _c4l if _a(r, "thr_sw_power_cap")), 636)
chk("C4b sw_thermal samples", sum(1 for r in _c4l if _a(r, "thr_sw_thermal")), 2)
chk("C4b hw_thermal samples", sum(1 for r in _c4l if _a(r, "thr_hw_thermal")), 1)
chk("C4b hw_power_brake samples", sum(1 for r in _c4l if _a(r, "thr_hw_power_brake")), 0)
chk("C4b the thermal flags carried no meaningful downclock",
    sorted(int(_n(r, "gfx_mhz")) for r in _c4l if _a(r, "thr_sw_thermal") or _a(r, "thr_hw_thermal")),
    [1935, 1950, 1950])
chk("C4b power limit is stock and constant",
    sorted({int(_n(r, "power_limit_w")) for r in _c4l}), [350])

print("\n=== A12: the checkpoint cost, timed in the source ===")
# Keyed by arm this was a dict, so the four arm-runs of one arm collapsed into
# whichever the JSON listed last. It went unnoticed because the extractor that
# wrote the file stripped only `__rep0.log`, leaving repeats 1-3 under distinct
# keys; fixing the extractor exposed the defect here. Grouped, not keyed.
_tmr = json.load(open("v4_audit_2026_08_25/data/checkpoint_timers_20260826.json"))
_tm: dict = {}
for _r in _tmr:
    _tm.setdefault(_r["arm"], []).append(_r)
chk("A12 timer records are one per arm-run, not one per arm",
    len(_tmr), sum(len(v) for v in _tm.values()))
chk("A12 every arm is covered at the same depth",
    sorted({len(v) for v in _tm.values()}), [4])
chk("A12 each arm carries every repeat index",
    sorted({tuple(sorted(r["repeat"] for r in v)) for v in _tm.values()}),
    [(0, 1, 2, 3)])
_ext = _tm["spec-draft-n8"]
chk("A12 timed arm-runs of the external drafter", len(_ext), 4)
chk("A12 checkpoint creates per arm-run", sorted({r["creates"] for r in _ext}), [785])
chk("A12 restores per arm-run", sorted({r["restores"] for r in _ext}), [728])
chk("A12 update_dft on the speculative checkpoint never fires",
    sorted({r["update_dft_s"] for r in _ext}), [0.0])
chk("A12 update_tgt seconds", round(st.mean([r["update_tgt_s"] for r in _ext]), 2), 17.34, 0.005)
chk("A12 load_tgt seconds", round(st.mean([r["load_tgt_s"] for r in _ext]), 2), 16.33, 0.005)
chk("A12 load_dft seconds", round(st.mean([r["load_dft_s"] for r in _ext]), 2), 5.41, 0.005)
_ck = st.mean([r["checkpoint_total_s"] for r in _ext])
chk("A12 checkpoint total seconds", round(_ck, 2), 39.07, 0.005)
chk("A12 the total is reproducible across arm-runs",
    round(max(r["checkpoint_total_s"] for r in _ext) -
          min(r["checkpoint_total_s"] for r in _ext), 2) <= 0.05, True)
chk("A12 DFlash performs no checkpoint operations",
    sorted({r["checkpoint_total_s"] for r in _tm["spec-dflash-n2"]}), [0.0])
chk("A12 the baseline performs none either",
    sorted({r["checkpoint_total_s"] for r in _tm["baseline"]}), [0.0])
# Zero seconds is not zero events: a create that took no measurable time would
# satisfy the two checks above. The claim is that the controls take NO
# checkpoints, so the counts are what has to be zero, in every repeat of both.
for _ctrl in ("baseline", "spec-dflash-n2"):
    chk(f"A12 {_ctrl} creates, every repeat",
        sorted({r["creates"] for r in _tm[_ctrl]}), [0])
    chk(f"A12 {_ctrl} restores, every repeat",
        sorted({r["restores"] for r in _tm[_ctrl]}), [0])
    chk(f"A12 {_ctrl} update_tgt seconds, every repeat",
        sorted({r["update_tgt_s"] for r in _tm[_ctrl]}), [0.0])
    chk(f"A12 {_ctrl} load_tgt seconds, every repeat",
        sorted({r["load_tgt_s"] for r in _tm[_ctrl]}), [0.0])
    chk(f"A12 {_ctrl} load_dft seconds, every repeat",
        sorted({r["load_dft_s"] for r in _tm[_ctrl]}), [0.0])

# the accounting, from run T's own decode times
_T = glob.glob("v4_audit_2026_08_25/data/matrix_T_timers_*")[0]
def _decT(arm):
    v = [sum(x["predicted_ms"] for x in json.load(open(f))["rows"]) / 1000
         for f in glob.glob(f"{_T}/{arm}__rep*.json") if json.load(open(f)).get("rows")]
    return st.mean(v)
_b, _s, _d = _decT("baseline"), _decT("spec-draft-n8"), _decT("spec-dflash-n2")
_exc = _s - _b
chk("A12 excess decode seconds", round(_exc, 1), 71.4, 0.05)
chk("A12 checkpoint share of the excess (%)", round(100 * _ck / _exc, 1), 54.7, 0.05)
chk("A12 checkpointing is the largest single term", _ck > 0.5 * _exc, True)
chk("A12 DFlash is faster than no speculation on the same build (s)",
    round(_d - _b, 1), -5.3, 0.05)

# the withdrawn estimate measured the restore direction and missed the create
_restore = st.mean([r["load_tgt_s"] + r["load_dft_s"] for r in _ext])
chk("A12 restore-only total, which the log-gap rule could see (s)",
    round(_restore, 1), 21.7, 0.05)
chk("A12 the withdrawn 24.2 s is close to the restore side, not the whole",
    abs(24.2 - _restore) < abs(24.2 - _ck), True)

# the instrumented build had to reproduce the stock one
_O2 = glob.glob("v4_audit_2026_08_25/data/matrix_O2_latin_*")[0]
def _pool(d, arm):
    rs = [json.load(open(f)) for f in glob.glob(f"{d}/{arm}__rep*.json")]
    rs = [r for r in rs if r.get("rows")]
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    return 1000 * n / ms
for arm, tol in (("baseline", 1.0), ("spec-draft-n8", 1.0), ("spec-dflash-n2", 1.0)):
    _delta = 100 * (_pool(_T, arm) / _pool(_O2, arm) - 1)
    chk(f"A12 control: instrumented reproduces stock for {arm} (within {tol} %)",
        abs(_delta) < tol, True)

print("\n=== theory (ERRATA E1/E2) ===")
rho=8/256
chk("rho", round(rho,5), 0.03125, 1e-9)
chk("T_95 exact", round(math.log(0.05)/math.log(1-rho),2), 94.36, 0.01)
chk("T_95 ceil", math.ceil(math.log(0.05)/math.log(1-rho)), 95)
chk("coverage at 94 < 0.95", 1-(1-rho)**94 < 0.95, True)
chk("coverage at 95 >= 0.95", 1-(1-rho)**95 >= 0.95, True)

print("\n=== do the documents actually quote these figures? ===")

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
    ("README.md",   "48.2 % acceptance", "README acceptance threshold, as fitted"),
    ("README.md",   "45-48 %", "README reports it as a range, not a point"),
    ("ERRATA.md",   "85 / 200",  "A11 think-off identical streams"),
    ("ERRATA.md",   "70.8 %",    "A11 long-output divergence"),
    ("v4_audit_2026_08_25/README.md", "-0.24 %", "L clock drift"),
    ("RETEST_TODO.md", "785 plain BF16", "MTP weights are present"),
    ("ERRATA.md",   "772",       "A12 checkpoints created"),
    ("ERRATA.md",   "118.71",    "A12 corrected nominal volume"),
    ("ERRATA.md",   "61.88",     "A12 corrected write side"),
    ("README.md",   "118.7 GiB", "README corrected nominal volume"),
    ("README.md",   "21.1 %",    "README unattributed share, measured"),
    ("ERRATA.md",   "68.7 %",    "A12 threshold failure point"),
    ("ERRATA.md",   "0.5 pp",    "A13 no-checkpoint agreement"),
    ("ERRATA.md",   "636 of 1272", "C4b sw_power_cap, corrected"),
    ("ERRATA.md",   "2 of 1272", "C4b sw_thermal, corrected"),
    ("ERRATA.md",   "mean 64.7", "C4b mean temperature, corrected"),
    ("v4_audit_2026_08_25/README.md", "1935 MHz against a run maximum of 1965", "v4 thermal line, corrected"),
    ("ERRATA.md",   "53.3 pp",   "A13 worst divergence"),
    ("ERRATA.md",   "1639 of 1639", "A13 the drafter counter is also a tautology"),
    ("v4_audit_2026_08_25/README.md", "3271", "N generate() calls"),
    ("v4_audit_2026_08_25/README.md", "4323", "P extended prompt tokens"),
    ("v4_audit_2026_08_25/README.md", "12.7 %", "P prompt-processing share"),
    ("v4_audit_2026_08_25/README.md", "+20.3 %", "P dflash-n2 on the new set"),
    ("ERRATA.md",   "0.56 pp",   "A14 median between-run spread"),
    ("v4_audit_2026_08_25/README.md", "78 / 86", "threshold scorecard"),
    ("v4_audit_2026_08_25/README.md", "57 / 59", "threshold, self-speculative"),
    ("README.md",   "57 / 59",   "README threshold scorecard"),
    ("ERRATA.md",   "8.5 pp",    "A14 the pair that did not replicate"),
    ("README.md",   "+26.7 %",   "README discloses the same-config replicate"),
    ("v4_audit_2026_08_25/README.md", "292.1 s", "P pooled includes the draft cost"),
    ("v4_audit_2026_08_25/README.md", "72.8 %",  "P acceptance not inflated"),
    ("ERRATA.md",   "39.08",     "A12 measured checkpoint total"),
    ("ERRATA.md",   "54.7 %",    "A12 checkpoint share"),
    ("ERRATA.md",   "21.9 ms",   "A12 median create"),
    ("README.md",   "39.07 s",   "README checkpoint total"),
    ("README.md",   "69.7 %",    "README the falsifying acceptance"),
    ("README.md",   "146.2",     "README O2 winner"),
    ("README.md",   "+26.3 %",   "README O2 headline"),
    ("README.md",   "[+25.5 %, +27.1 %]", "README O2 interval"),
    ("README.md",   "-74.8 %",   "README O2 worst row"),
    ("ERRATA.md",   "not token-stream", "A11 corrected framing"),
    ("ERRATA.md",   "not a validity test", "A13 corrected inference"),
    ("ERRATA.md",   "b6a5c490bb932ffa", "A15 the launcher hash"),
    ("ERRATA.md",   "ce94855f4f2d82ba", "A15 the instrumented library"),
    ("v4_audit_2026_08_25/README.md", "Latin", "the balanced design is described"),
    ("v4_audit_2026_08_25/README.md", "+23.2 %", "M1 DFlash n2"),
    ("v4_audit_2026_08_25/README.md", "+18.6 %", "M1 MTP n2"),
    ("v4_audit_2026_08_25/README.md", "+22.3 %", "M4 Q4_K_M is faster"),
    ("v4_audit_2026_08_25/README.md", "+11.4 %", "M3 MTP survives think-off"),
    ("v4_audit_2026_08_25/README.md", "-7.6 %",  "M2 MTP at c=8"),
    ("v4_audit_2026_08_25/README.md", "0.0 %",   "N ngram-map never engages"),
    ("v4_audit_2026_08_25/README.md", "closed by runs T and T3",
     "the checkpoint wall-clock gap is no longer listed as open"),
    ("v4_audit_2026_08_25/README.md", "A16",
     "the unexplained between-run shift is listed as open"),
    ("v4_audit_2026_08_25/README.md", "A17",
     "the length confound is listed as open"),
]
root = pathlib.Path(__file__).resolve().parents[1]
for f, needle, what in DOC_CLAIMS:
    txt = _norm((root / f).read_text(encoding="utf-8"))
    ok = _norm(needle) in txt
    print(f"  {'PASS' if ok else 'FAIL'}  {f:32s} quotes {needle!r:20s} ({what})")
    if not ok:
        FAIL.append(f"{f}:{needle}")


_T = "v4_audit_2026_08_25/data/matrix_T_timers_20260826_182639"
_T3 = "v4_audit_2026_08_25/data/matrix_T3_timers_20260826_203251"
print("\n=== the acceptance threshold, refitted without the length confound ===")
# Half of run L's 60 fitted points are its thinking-off half, where the arms
# generated different numbers of tokens (A17). Both coordinates of those points
# - acceptance and the delta - move when that is controlled, so the threshold
# quoted from them carries a sensitivity that was not reported.


def _thr_series(pattern, only=None, length_matched=False):
    per = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    lens = defaultdict(lambda: defaultdict(set))
    for f in glob.glob(f"v4_audit_2026_08_25/data/{pattern}"):
        r = json.load(open(f))
        for x in r["rows"]:
            per[x["tag"]][r["arm"]].append(x["predicted_per_second"])
            a = acc[x["tag"]][r["arm"]]
            a[0] += x["draft_n_accepted"]
            a[1] += x["draft_n"]
            lens[x["tag"]][r["arm"]].add(x["predicted_n"])
    out = []
    for t in per:
        if "baseline" not in per[t]:
            continue
        if length_matched and len({n for a in lens[t] for n in lens[t][a]}) > 1:
            continue
        for arm in per[t]:
            if arm == "baseline" or not acc[t][arm][1]:
                continue
            if only and only not in arm:
                continue
            out.append((100 * acc[t][arm][0] / acc[t][arm][1],
                        100 * (st.mean(per[t][arm]) / st.mean(per[t]["baseline"]) - 1)))
    return out


def _thr_fit(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sl = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    return sl, -(my - sl * mx) / sl, len(pts)


_on = _thr_series("matrix_L_thinkon_*/*__rep*.json", only="dflash")
_off = _thr_series("matrix_L_thinkoff_*/*__rep*.json", only="dflash")
_offm = _thr_series("matrix_L_thinkoff_*/*__rep*.json", only="dflash", length_matched=True)
_sl_pub, _br_pub, _n_pub = _thr_fit(_on + _off)
_sl_lm, _br_lm, _n_lm = _thr_fit(_on + _offm)
_sl_on, _br_on, _n_on = _thr_fit(_on)
chk("threshold: fitted points as published", _n_pub, 60)
chk("threshold: break-even as published (%)", round(_br_pub, 1), 48.2, 0.05)
chk("threshold: break-even on the length-matched fit (%)", round(_br_lm, 1), 46.5, 0.05)
chk("threshold: break-even on the thinking-on half (%)", round(_br_on, 1), 45.4, 0.05)
chk("threshold: it moves less than the slope does",
    abs(_br_lm - _br_pub) / _br_pub < abs(_sl_lm - _sl_pub) / _sl_pub, True)
_rm = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                     .joinpath("README.md").read_text(encoding="utf-8")).split())
chk("README reports the threshold as a range", "45-48 %" in _rm, True)
_pd = json.load(open("analysis/plot_data.json"))["acceptance_threshold"]
chk("the chart's fit matches the checker's", round(_pd["break_even"], 1),
    round(_br_pub, 1), 0.05)
chk("the chart records the length-matched fit too",
    round(_pd["break_even_length_matched"], 1), round(_br_lm, 1), 0.05)
chk("the chart records how many points each fit used",
    (_pd["n_fitted"], _pd["n_fitted_length_matched"]), (60, 45))
chk("README no longer says the checkpoint wall clock is not established",
    "How much wall clock that costs is **not** established here" in _rm, False)


print("\n=== the memory-policy table names every run ===")
# The table in BENCHMARK_ENV.md exists to say which runs are comparable, and it
# stopped at run L while most of the repository - the headline included - ran
# under a policy it did not mention.
_env = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                      .joinpath("BENCHMARK_ENV.md").read_text(encoding="utf-8")).split())
_policy: dict = {}
for _mp in sorted(glob.glob("v4_audit_2026_08_25/data/*/manifest.json")):
    _m = json.load(open(_mp))
    _n = os.path.basename(os.path.dirname(_mp))
    _tag = _n.split("_")[1] if _n.startswith(("matrix_", "smoke_")) else _n.split("_")[0]
    _policy.setdefault(str(_m.get("fit_target") or "-"), set()).add(_tag)
chk("BENCHMARK_ENV names the fit-target values that were used",
    sorted(k for k in _policy if k != "-" and k not in _env), [])
chk("the headline run's policy is in the table",
    "3072" in _env and "O2" in _env, True)
chk("every fit-target 3072 run tag appears in the table",
    sorted(t for t in _policy.get("3072", set()) if t not in _env), [])
# the telemetry script produced one of seventeen traces; the other two schemas
# were inline in driver scripts that were never committed
_tel = pathlib.Path(__file__).resolve().parents[1].joinpath("bench", "gpu_telemetry.sh") \
    .read_text(encoding="utf-8")
chk("the telemetry script carries every schema that was used",
    sorted(x for x in ("full", "compact", "raw") if f"{x})" not in _tel), [])
chk("its compact schema has the nine fields those traces have",
    len([c for c in _tel.split("echo 'ts,util")[1].split("'")[0].split(",")]), 8)
chk("BENCHMARK_ENV names all three", all(x in _env for x in
    ("`full`", "`compact`", "`raw`")), True)


print("\n=== the harness that ran each run is recoverable ===")
# `harness_tree_sha` is what the caller declared and is labelled as such; the
# field that pins the harness exactly is `runner_sha256`, the hash of
# bench/retest_runner.py as it was when the run started. That is only provenance
# if the file with that hash still exists somewhere in this repository, so the
# check is: does some commit's bench/retest_runner.py hash to it?
import subprocess as _sp

_repo = pathlib.Path(__file__).resolve().parents[1]


def _runner_blobs() -> set:
    """Every version of bench/retest_runner.py in this repository's history."""
    try:
        revs = _sp.run(["git", "-C", str(_repo), "rev-list", "--all",
                        "--", "bench/retest_runner.py"],
                       capture_output=True, text=True, timeout=120)
        if revs.returncode != 0:
            return set()
        out = set()
        for rev in revs.stdout.split():
            blob = _sp.run(["git", "-C", str(_repo), "show",
                            f"{rev}:bench/retest_runner.py"],
                           capture_output=True, timeout=60)
            if blob.returncode == 0:
                out.add(hashlib.sha256(blob.stdout).hexdigest())
        return out
    except Exception:  # noqa: BLE001
        return set()


_blobs = _runner_blobs()
_declared = {}
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/manifest.json")):
    _m = json.load(open(_d))
    if _m.get("runner_sha256"):
        _declared[os.path.basename(os.path.dirname(_d))] = _m["runner_sha256"]
chk("runs that record which harness ran them", len(_declared) >= 1, True)

# `harness_tree_sha` is the caller's word for which commit it ran from, and a
# caller can be wrong: run U's driver script declared 842c971b while the runner
# it actually executed is the one committed at 1b8053a. The two fields are
# compared, and every disagreement has to be listed here on purpose rather than
# passing silently. `runner_sha256` is the authoritative one.
# `harness_tree_sha` is the caller's word for which commit it ran from. A caller
# can be wrong, and the commit can also stop existing. Both happened here: run
# U's driver was edited while it waited, so it declared the commit it was
# written against rather than the one holding the runner it executed; and the
# 2026-08-27 identity rewrite gave 50 commits new SHAs, so eight manifests name
# a commit no longer in this history. The manifests are recorded data and are
# left as written; data/harness_sha_rewrite_map.json carries the mapping.
_REWRITE = json.load(open("v4_audit_2026_08_25/data/harness_sha_rewrite_map.json"))
_RW_MAP = _REWRITE["cause_1_identity_rewrite"]["map"]
_RW_RUNS = _REWRITE["cause_2_driver_declared_the_wrong_commit"]["runs"]
chk("the rewrite map names the two commits that moved", len(_RW_MAP), 2)
# Everything below needs this repository's history. `tests/data_mutate.py` runs
# this file in a mirror that has no `.git`, and CI runs that harness, so the
# git-backed checks have to skip there rather than fail - which is how they
# broke the "unit and mutation" job while passing everywhere else.
_HAS_GIT = _sp2.run(["git", "-C", str(_repo), "rev-parse", "--git-dir"],
                    capture_output=True).returncode == 0
if not _HAS_GIT:
    print("  ----  no git history here (a mirror, or a shallow clone); "
          "the harness-provenance checks are skipped, not passed")
if _HAS_GIT:
    chk("and both new SHAs are in this history",
        sorted(v for v in _RW_MAP.values()
               if _sp2.run(["git", "-C", str(_repo), "cat-file", "-e", v],
                           capture_output=True).returncode != 0), [])
_resolved, _mismatch, _dangling = 0, [], []
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/manifest.json")):
    _m = json.load(open(_d))
    _rn = os.path.basename(os.path.dirname(_d))
    _decl = (_m.get("harness_tree_sha") or {}).get("sha")
    _actual = _m.get("runner_sha256")
    if not (_decl and _actual):
        continue
    _decl = _RW_MAP.get(_decl, _decl)
    try:
        _blob = _sp2.run(["git", "-C", str(_repo), "show",
                          f"{_decl}:bench/retest_runner.py"],
                         capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        continue
    if _blob.returncode != 0:
        _dangling.append(_rn)
        continue
    _resolved += 1
    if hashlib.sha256(_blob.stdout).hexdigest() != _actual:
        _mismatch.append(_rn)
if _HAS_GIT and (_resolved or _dangling):
    chk("every declared harness commit resolves, after the rewrite map",
        sorted(_dangling), [])
    chk("the runs whose declared commit is not the one that ran",
        sorted(_mismatch), sorted(_RW_RUNS))
    chk("and each one says why", sorted(r for r, w in _RW_RUNS.items() if not w), [])
elif _HAS_GIT:
    print("  ----  declared-vs-actual harness commits: nothing to resolve")
if _HAS_GIT and _blobs:
    chk("every recorded harness hash is in this repository's history",
        sorted(k for k, v in _declared.items() if v not in _blobs), [])
    # deliberately NOT asserting that the working-tree file is in history: it
    # is not, until it is committed, and a check that cannot fail is worse than
    # no check. The assertion above is the one that matters.
    print(f"  ----  {len(_blobs)} harness version(s) in history, "
          f"{len(_declared)} run(s) pin one")
else:
    print("  SKIP  git history unavailable (shallow clone?); "
          "runner_sha256 not resolved")


print("\n=== A17: are the arms compared on the same amount of work? ===")
_lm = json.load(open("analysis/length_matching.json"))["runs"]
_on = [r for r in _lm if r["think"] != "off"]
# run V's hard-cap half is thinking-off AND fully length-matched by design, so
# the confound claims below are about the thinking-off runs that did NOT
# force the cap - which is every archived one
_off = [r for r in _lm if r["think"] == "off" and not r.get("ignore_eos")]
_off_capped = [r for r in _lm if r["think"] == "off" and r.get("ignore_eos")]
# the two prompts A17 names, checked rather than remembered
_R = "v4_audit_2026_08_25/data/matrix_R_ext_thinkoff_20260826_110747"
_rlen: dict = {}
for _f in glob.glob(f"{_R}/*__rep*.json"):
    _r = json.load(open(_f))
    for _x in _r["rows"]:
        _rlen.setdefault(_x["tag"], {}).setdefault(_r["arm"], set()).add(_x["predicted_n"])
chk("A17 run R code_bash baseline length",
    sorted(_rlen["code_bash"]["baseline"]), [300])
chk("A17 run R code_bash speculative lengths",
    sorted(n for a, v in _rlen["code_bash"].items() if a != "baseline" for n in v),
    [187, 188, 188])
chk("A17 run R code_rust baseline length",
    sorted(_rlen["code_rust"]["baseline"]), [203])
chk("A17 run R code_rust speculative lengths",
    sorted(n for a, v in _rlen["code_rust"].items() if a != "baseline" for n in v),
    [300, 300, 300])
chk("A17 thinking-off runs measured without a hard cap", len(_off), 5)
chk("A17 and the one measured with it", len(_off_capped), 1)
chk("A17 the capped one is fully length-matched",
    [r["prompts"] == r["length_matched_prompts"] for r in _off_capped], [True])
chk("A17 thinking-on runs with a computable comparison", len(_on), 31)
chk("A17 every thinking-on run is fully length-matched",
    sorted({r["prompts"] == r["length_matched_prompts"] for r in _on}), [True])
chk("A17 no thinking-on arm moves when the comparison is restricted",
    sorted({v.get("shift_pp", 0.0) for r in _on for v in r["arms"].values()}), [0.0])
chk("A17 every thinking-off run is only partly length-matched",
    sorted({r["prompts"] > r["length_matched_prompts"] for r in _off}), [True])
_shifts = {(r["run"], a): v["shift_pp"] for r in _off for a, v in r["arms"].items()
           if "shift_pp" in v}
chk("A17 arm-vs-baseline comparisons in the uncapped thinking-off runs",
    len(_shifts), 18)
_model = {k: v for k, v in _shifts.items() if not k[1].startswith("ngram-")}
_ngram = {k: v for k, v in _shifts.items() if k[1].startswith("ngram-")}
chk("A17 arms that draft from a model", len(_model), 16)
chk("A17 every one of them shifts the same way",
    sorted({v > 0 for v in _model.values()}), [True])
chk("A17 the largest such shift (pp)", round(max(_model.values()), 2), 16.79, 0.005)
chk("A17 the smallest such shift (pp)", round(min(_model.values()), 2), 2.52, 0.005)
chk("A17 the exceptions are the n-gram arms", len(_ngram), 2)
chk("A17 and both are in run D",
    sorted({k[0] for k in _ngram}), ["D_master_matrix_think_off"])
chk("A17 ngram-cache moves the other way (pp)",
    round(_shifts[("D_master_matrix_think_off", "ngram-cache")], 2), -6.37, 0.005)
_L = [r for r in _off if r["run"].startswith("matrix_L_thinkoff")][0]
chk("A17 run L spec-dflash-n4 as published (%)",
    _L["arms"]["spec-dflash-n4"]["all_prompts_pct"], -2.69, 0.005)
chk("A17 run L spec-dflash-n4 length-matched (%)",
    _L["arms"]["spec-dflash-n4"]["length_matched_pct"], 14.10, 0.005)
chk("A17 that is a sign flip",
    _L["arms"]["spec-dflash-n4"]["all_prompts_pct"] < 0
    < _L["arms"]["spec-dflash-n4"]["length_matched_pct"], True)

# the raw fact underneath it: thinking-on never stops early, thinking-off does
_stops = {"on": Counter(), "off": Counter()}
for _f in glob.glob("v4_audit_2026_08_25/data/*/*__rep*.json"):
    _m = json.load(open(os.path.join(os.path.dirname(_f), "manifest.json")))
    _k = "off" if str(_m.get("think")) == "off" else "on"
    for _x in json.load(open(_f)).get("rows") or []:
        _stops[_k][_x.get("finish_reason")] += 1
chk("A17 no thinking-on request stopped before the cap",
    dict(_stops["on"]), {"length": 5724})
# finish_reason is the server's word for it; the claim is about token counts
_short = 0
for _f in glob.glob("v4_audit_2026_08_25/data/*/*__rep*.json"):
    _m = json.load(open(os.path.join(os.path.dirname(_f), "manifest.json")))
    if str(_m.get("think")) == "off" or not _m.get("max_tokens"):
        continue
    for _x in json.load(open(_f)).get("rows") or []:
        if _x["predicted_n"] != _m["max_tokens"]:
            _short += 1
chk("A17 and every one of them generated exactly max_tokens", _short, 0)
chk("A17 thinking-off requests that stopped early", _stops["off"]["stop"], 881)
# the second reading A17 overturns: acceptance, not only throughput
_D = [r for r in _lm if r["run"].startswith("D_master")][0]
_Con = [r for r in _lm if r["run"] == "C_master_matrix_think_on"][0]
chk("A17 draft model acceptance, thinking on",
    _Con["arms"]["spec-draft-n8"]["acceptance_all_prompts"], 29.67, 0.005)
chk("A17 draft model acceptance, thinking off, all prompts",
    _D["arms"]["spec-draft-n8"]["acceptance_all_prompts"], 23.09, 0.005)
chk("A17 draft model acceptance, thinking off, length-matched",
    _D["arms"]["spec-draft-n8"]["acceptance_length_matched"], 30.26, 0.005)
chk("A17 the acceptance fall is the short outputs, not the workload",
    _D["arms"]["spec-draft-n8"]["acceptance_length_matched"]
    > _Con["arms"]["spec-draft-n8"]["acceptance_all_prompts"], True)
chk("A17 ngram-cache acceptance returns to its thinking-on value",
    _D["arms"]["ngram-cache"]["acceptance_length_matched"], 1.85, 0.02)
chk("A17 ngram-mod really does stop drafting with thinking off",
    _D["arms"]["ngram-mod-n24"]["acceptance_all_prompts"], None)
_readme_txt = " ".join(pathlib.Path(__file__).resolve().parents[1]
                       .joinpath("README.md").read_text(encoding="utf-8").split())
chk("README no longer states the reading A17 overturns",
    "reasoning traces are *easier* for a 0.8 B drafter" in _readme_txt, False)
chk("README shows the length-matched acceptance instead",
    "30.3 %" in _readme_txt, True)
chk("ERRATA quotes the sign flip", "-2.7 %" in _norm(
    " ".join(pathlib.Path(__file__).resolve().parents[1]
             .joinpath("ERRATA.md").read_text(encoding="utf-8").split())), True)


print("\n=== ERRATA's tables, parsed cell by cell ===")
# Same lesson as the README's: the values are computed and asserted, and the
# TABLES were not. Six of eight planted perturbations in A12, A13 and C4b passed
# every check before this section existed.
_ER_LINES = pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md") \
    .read_text(encoding="utf-8").splitlines()


def _md_table(header_startswith):
    i = next(i for i, l in enumerate(_ER_LINES) if l.startswith(header_startswith))
    rows = []
    for l in _ER_LINES[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip().strip("*`").replace("`", "").strip("* ").strip()
                     for c in l.strip("|").split("|")])
    return rows


def _f(x):
    return float(_norm(x).replace("%", "").replace("pp", "").replace(" ", ""))


# --- A12's accounting: seconds and shares of the excess -------------------
_acc = {r[0]: r[1:] for r in _md_table("| | seconds | share of the excess |")}
chk("A12 table rows parsed", len(_acc), 9)
_excess = _f(_acc["excess to account for"][0])
chk("A12 table: the excess equals the two decode rows",
    round(_f(_acc["spec-draft-n8, decode"][0]) - _f(_acc["no speculation, decode"][0]), 1),
    round(_excess, 1), 0.05)
_parts = {"update_tgt \u2014 785 checkpoint creates": "update_tgt_s",
          "load_tgt \u2014 728 restores": "load_tgt_s",
          "load_dft \u2014 728 restores": "load_dft_s"}
for _label, _field in _parts.items():
    chk(f"A12 table: {_field} seconds",
        round(st.mean([r[_field] for r in _ext]), 2), _f(_acc[_label][0]), 0.005)
    chk(f"A12 table: {_field} share of the excess (%)",
        round(100 * st.mean([r[_field] for r in _ext]) / _excess, 1),
        _f(_acc[_label][1]), 0.05)
# tolerance 0.015: the three displayed components are each rounded to two
# places, so they can add to one hundredth away from the total, which is
# rounded once from the raw microseconds. ERRATA says so in the text.
chk("A12 table: the checkpoint total is the sum of its three parts",
    round(sum(_f(_acc[k][0]) for k in _parts), 2),
    _f(_acc["speculative checkpoint, total"][0]), 0.015)
chk("A12 table: and its share",
    round(100 * _f(_acc["speculative checkpoint, total"][0]) / _excess, 1),
    _f(_acc["speculative checkpoint, total"][1]), 0.05)
chk("A12 table: the rows add up to the excess",
    round(sum(_f(_acc[k][0]) for k in
              ("speculative checkpoint, total", "drafter generate()", "unattributed")), 2),
    round(_excess, 2), 0.05)
chk("A12 table: the shares add up to 100 %",
    round(sum(_f(_acc[k][1]) for k in
              ("speculative checkpoint, total", "drafter generate()", "unattributed")), 1),
    100.0, 0.15)

# --- A13's counter comparison --------------------------------------------
_a13 = {r[0]: r[1:] for r in
        _md_table("| arm | server counter | drafter's own counter | gap | checkpoints |")}
chk("A13 table rows parsed", len(_a13), 6)
_ccmap = {}
for _r in _cc:
    _ccmap.setdefault(_r["arm"], []).append(_r)
for _arm, _cells in _a13.items():
    _cands = [r for r in _ccmap.get(_arm, [])
              if round(r["server_pct"], 1) == _f(_cells[0])]
    chk(f"A13 table {_arm}: the row matches a measured arm-run", bool(_cands), True)
    if not _cands:
        continue
    _r0 = _cands[0]
    chk(f"A13 table {_arm}: drafter counter",
        round(_r0["drafter_pct"], 1), _f(_cells[1]), 0.05)
    chk(f"A13 table {_arm}: the gap is the difference",
        round(abs(_r0["server_pct"] - _r0["drafter_pct"]), 1), _f(_cells[2]), 0.05)
    chk(f"A13 table {_arm}: checkpoint count",
        _r0["checkpoints_created"], int(_f(_cells[3])))

# --- C4b's clock line, which the thermal checks did not cover -------------
_c4b = " ".join(_norm(" ".join(_ER_LINES)).split())
_c4b_report = _sp2.run(
    [_sys2.executable, str(pathlib.Path(__file__).resolve().parents[1]
                           / "analysis" / "thermal_report.py"),
     "v4_audit_2026_08_25/data/gpu_telemetry_20260825.csv"],
    capture_output=True, text=True, timeout=120).stdout
chk("C4b quotes the clock range and mean",
    "1800-1965 MHz of a 2100 MHz maximum, mean 1937" in _c4b, True)
chk("and thermal_report computes exactly that",
    "1800-1965 MHz of 2100 MHz, mean 1937" in _norm(_c4b_report), True)
chk("C4b quotes the temperature range and mean",
    "58-75 C, mean 64.7" in _norm(_c4b_report), True)


print("\n=== every quoted throughput belongs to some measured arm-run ===")
# The three headline tables are bound cell by cell above. The rest - the v4
# audit README alone has dozens - are not, and planting wrong numbers in them
# passed every check. This is the general net, and it is deliberately narrow:
# for a table whose header names a throughput column, the value in that column
# of each arm row must equal SOME throughput derivable for that arm. It cannot
# tell which run a figure belongs to, so it catches a typo or a stale number
# and not a figure attributed to the wrong run. A wider version of this, which
# swept every number in the row, produced 150 false positives from percentage
# and acceptance columns; a check that needs that many exemptions is worse than
# none.
_TPUT: dict = {}
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*")):
    if not os.path.isfile(os.path.join(_d, "manifest.json")):
        continue
    _per: dict = {}
    for _f in glob.glob(f"{_d}/*__rep*.json"):
        _r = json.load(open(_f))
        if not _r.get("rows"):
            continue
        _p = _per.setdefault(_r["arm"], {"n": 0, "ms": 0, "rates": [],
                                         "agg": [], "rp": []})
        _p["n"] += sum(x["predicted_n"] for x in _r["rows"])
        _p["ms"] += sum(x["predicted_ms"] for x in _r["rows"])
        _p["rates"] += [x["predicted_per_second"] for x in _r["rows"]]
        if _r.get("aggregate_tok_s"):
            _p["agg"].append(_r["aggregate_tok_s"])
        _p["rp"].append(1000 * sum(x["predicted_n"] for x in _r["rows"])
                        / sum(x["predicted_ms"] for x in _r["rows"]))
    for _arm, _q in _per.items():
        _vals = []
        if _q["ms"]:
            _vals.append(1000 * _q["n"] / _q["ms"])
        if _q["rates"]:
            _vals += [st.mean(_q["rates"]), st.median(_q["rates"]), min(_q["rates"])]
        if _q["agg"]:
            _vals += _q["agg"] + [st.mean(_q["agg"])]
        if _q["rp"]:
            _vals += _q["rp"] + [st.mean(_q["rp"])]
        _TPUT.setdefault(_arm, set()).update(round(v, 1) for v in _vals)
# the v1 tier lives in analysis/summary.csv and uses its own arm names
for _c, _v in by.items():
    _rates = [x["tok_s"] for x in _v]
    _n = sum(x["predicted_n"] for x in _v)
    _ms = sum(x["predicted_ms"] for x in _v)
    _TPUT.setdefault(_c, set()).update(
        round(x, 1) for x in
        ([1000 * _n / _ms] if _ms else []) +
        [st.mean(_rates), st.median(_rates), min(_rates), max(_rates)])

# A throughput cell is a bare decimal, optionally bold and optionally with a
# +/- SD. A percentage cell carries %, a count is an integer. Keying on the cell
# shape rather than on the column header covers tables headed "pooled", "req-mean"
# and "aggregate" alike, and the two-row headers in the run L section.
_ROW = re.compile(r"^\|\s*\**`?([a-z0-9-]+)`?[^|]*\|")
_CELL = re.compile(r"^\**\s*(\d{2,3}\.\d)\s*(?:±\s*\d+\.\d+\s*)?\**$")
# Cells that look like a throughput and are not one. Named exactly, so the
# exemption cannot widen by accident.
_NOT_A_THROUGHPUT = {
    ("ERRATA.md", "spec-draft-n8", 97.2),   # A12's table: seconds of decode
    ("v2_3090_followup/README.md", "baseline", 139.9),   # the v2 tier's own
    ("v2_3090_followup/SUMMARY.md", "baseline", 139.9),  # numbers, from
    ("v2_3090_followup/SUMMARY.md", "baseline", 140.0),  # results_v2.json
    ("v2_3090_followup/SUMMARY.md", "baseline", 139.7),
    ("v2_3090_followup/SUMMARY.md", "baseline", 139.5),
}
_bad_t = []
for _f in sorted(glob.glob("*.md") + glob.glob("*/*.md")):
    for _i, _raw in enumerate(open(_f, encoding="utf-8"), 1):
        _line = _norm(_raw.strip())
        if not _line.startswith("|"):
            continue
        _m = _ROW.match(_line)
        if not _m:
            continue
        _arm = _m.group(1)
        if _arm not in _TPUT:
            continue
        for _cell in [c.strip() for c in _line.strip("|").split("|")][1:]:
            _c = _CELL.match(_norm(_cell))
            if not _c:
                continue
            _v = float(_c.group(1))
            if (_f, _arm, _v) in _NOT_A_THROUGHPUT:
                continue
            # 0.051: the documents quote one decimal, so anything further than
            # half a display unit away is a different number, not a rounding.
            if not any(abs(_v - x) <= 0.051 for x in _TPUT[_arm]):
                _bad_t.append(f"{_f}:{_i} {_arm} = {_v}")
chk("arms with a derivable throughput", len(_TPUT) >= 30, True)
chk("every quoted throughput is one of them", _bad_t, [])


print("\n=== the v2 result files carry corrected metadata and unchanged numbers ===")
# Three v2 JSONs are not byte-identical to the published release: each gained an
# `audit_2026_08_25` block and had `draft_model` and `interpretation` corrected.
# Each says "measurements unchanged", which was true and unverifiable. The
# fingerprint below is every number in the file OUTSIDE that block, sorted -
# taken while master was still reachable and checked against it there.
_V2_NUMBERS = {
    "v2_3090_followup/results_v2.json":
        ("4910e6a311a1d7da7d6e206385b6861d0fd9a70e3f85c972751694f513a6a3bb", 241),
    "v2_3090_followup/n3_results_20260426.json":
        ("09e7e614050e3b15a7724db22ac45293109f4f55dcedbd699b6a34666ac6430e", 149),
    "v2_3090_followup/exp2_codejson_n3/results.json":
        ("5a84cb9ead5f4f5ac5610d09aeb8c80bdc56e3b5e3c8c246674d8e5a0737bbc4", 215),
}


def _measurement_numbers(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "audit_2026_08_25":
                continue
            _measurement_numbers(v, out)
    elif isinstance(o, list):
        for v in o:
            _measurement_numbers(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out.append(float(o))
    return out


for _f, (_want, _n) in _V2_NUMBERS.items():
    _d = json.load(open(pathlib.Path(__file__).resolve().parents[1] / _f))
    _v = sorted(_measurement_numbers(_d, []))
    chk(f"{_f}: measurement values unchanged",
        hashlib.sha256(repr(_v).encode()).hexdigest(), _want)
    chk(f"{_f}: how many there are", len(_v), _n)
    chk(f"{_f}: carries an audit block that says what it did",
        bool((_d.get("audit_2026_08_25") or {}).get("status")), True)
chk("the two whose metadata was corrected say so",
    sorted(f for f in _V2_NUMBERS
           if (json.load(open(pathlib.Path(__file__).resolve().parents[1] / f))
               .get("audit_2026_08_25") or {}).get("status")
           == "corrected metadata; measurements unchanged"),
    ["v2_3090_followup/n3_results_20260426.json",
     "v2_3090_followup/results_v2.json"])
chk("and Exp 2 says its treatment was never verified",
    (json.load(open(pathlib.Path(__file__).resolve().parents[1]
                    / "v2_3090_followup/exp2_codejson_n3/results.json"))
     .get("audit_2026_08_25") or {}).get("status"),
    "EXPLORATORY - the intended treatment was not verified")


print("\n=== the raw-evidence manifest ===")
_man = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "EVIDENCE_MANIFEST.sha256"
_mlines = [l for l in _man.read_text(encoding="utf-8").splitlines()
           if l and not l.startswith("#")]
chk("manifest entries", len(_mlines), 721)
chk("every entry is a sha256 and a path",
    sorted({bool(re.fullmatch(r"[0-9a-f]{64}  \S.*", l)) for l in _mlines}), [True])
chk("logs in the manifest", sum(1 for l in _mlines if l.endswith(".log")), 702)
chk("telemetry traces in the manifest",
    sum(1 for l in _mlines if l.endswith(".csv")), 19)
chk("no duplicate paths", len({l.split("  ", 1)[1] for l in _mlines}), len(_mlines))
_v4r = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "README.md"
chk("the archive hash is recorded in both places",
    all("29c2401f100390268bbd52e43b5c2da9a61440bad3dabe502ca1684478771fd6" in t
        for t in (_man.read_text(encoding="utf-8"), _v4r.read_text(encoding="utf-8"))), True)
chk("and both say it is not published",
    all("not published" in t for t in
        (_man.read_text(encoding="utf-8"), _v4r.read_text(encoding="utf-8"))), True)
# Every committed run's logs should appear in the manifest. Four of the earliest
# were renamed when they were archived, so the committed directory and the one
# the logs live in differ; the mapping is written out rather than exempted,
# because "not in the manifest" and "in the manifest under another name" are
# different facts and the first version of this check could not tell them apart.
_ARCHIVED_AS = {
    "C_master_matrix_think_on": "matrix_C_20260825_204529",
    "D_master_matrix_think_off": "matrix_D_20260825_204529",
    "E_past_threshold": "matrix_E_threshold_20260825_224802",
    "H_pmin_sweep": "matrix_H_pmin_20260826_005716",
}
# Runs A and B predate log retention on the host; their arm-run JSON survives
# and their logs do not.
_NO_LOGS_RETAINED = {"A_bcb5eeb64_legacy", "B_master_3737e4137"}
_runs_in_manifest = {l.split("  ", 1)[1].split("/")[0] for l in _mlines if l.endswith(".log")}
_committed = {d.name for d in (pathlib.Path(__file__).resolve().parents[1]
                               / "v4_audit_2026_08_25" / "data").iterdir()
              if d.is_dir() and list(d.glob("*__rep*.json"))}
chk("every committed run's logs are in the manifest, under its own name or its archived one",
    sorted(r for r in _committed - _NO_LOGS_RETAINED
           if _ARCHIVED_AS.get(r, r) not in _runs_in_manifest), [])
chk("the renamed ones really are in it under the other name",
    sorted(k for k, v in _ARCHIVED_AS.items() if v not in _runs_in_manifest), [])
chk("and the two without logs have arm-runs anyway",
    sorted(r for r in _NO_LOGS_RETAINED
           if not list((pathlib.Path(__file__).resolve().parents[1]
                        / "v4_audit_2026_08_25" / "data" / r).glob("*__rep*.json"))), [])


print("\n=== the historical scripts still document what was run ===")
# Six driver scripts are kept as evidence and carry a header saying the
# measurement flags are UNCHANGED and only paths were parameterised. That is a
# claim about the files, and until now only the header made it. Two of those
# flags are the subject of ERRATA D1 and D2 - `-no-cnv`, which llama-cli
# rejected, and `/no_think`, which did not disable thinking - so a well-meaning
# "fix" to either would delete the evidence for a published retraction.
_HISTORICAL = {
    # occurrences, not lines: `--draft-max 16 --draft-min 4` is one line and
    # two occurrences, and the first version of this table was filled in from
    # `grep -c`, which counts lines
    "run_matrix.sh": (1, 1, 4),
    "run_p0_matrix.sh": (1, 1, 9),
    "run_verify_matrix.sh": (1, 1, 2),
    "v2_3090_followup/bench_3090_oleg.sh": (2, 7, 8),
    "v3_dflash_2026_05_07/bench/bench_dflash.sh": (2, 6, 4),
    "v2_3090_followup/exp2_codejson_n3/run_n3_codejson.sh": (2, 6, 4),
}
for _f, (_ncnv, _nthink, _ndraft) in _HISTORICAL.items():
    _t = pathlib.Path(__file__).resolve().parents[1].joinpath(_f) \
        .read_text(encoding="utf-8")
    chk(f"{_f}: -no-cnv occurrences", _t.count("-no-cnv"), _ncnv)
    chk(f"{_f}: /no_think occurrences", _t.count("/no_think"), _nthink)
    chk(f"{_f}: draft-max/draft-min occurrences",
        _t.count("draft-max") + _t.count("draft-min"), _ndraft)
    chk(f"{_f}: says it is kept as evidence",
        "HISTORICAL SCRIPT" in _t and "UNCHANGED" in _t, True)
    chk(f"{_f}: every host path is overridable",
        [l.strip() for l in _t.splitlines()
         if re.match(r'^[A-Z_]+="?/home/[a-z]', l.strip())], [])


print("\n=== withdrawn figures must not reappear ===")
# Every number this repository retracted, and the one place each is allowed to
# be mentioned: inside the entry that retracts it. A13's 101.3 MiB survived in
# the upstream-issues table for eleven days after A12 withdrew it, because the
# correction was propagated by searching for the numbers it changed rather than
# for the number it removed.
_WITHDRAWN = {
    "101.3 MiB": "the checkpoint size with the draft component added twice (A12)",
    "133.2 GiB": "the nominal volume before that correction (A12)",
    "76.4 GiB": "the written half of it (A12)",
    "19.5 % of the wall": "the log-interval wall-clock share (A12)",
}
_ALLOWED = ("ERRATA.md",)          # the entries that retract them
for _needle, _what in _WITHDRAWN.items():
    _hits = []
    for _f in ("README.md", "CHANGELOG.md", "RETEST_TODO.md", "BENCHMARK_ENV.md",
               "v4_audit_2026_08_25/README.md", "ERRATA.md"):
        _t = _norm(pathlib.Path(__file__).resolve().parents[1]
                   .joinpath(_f).read_text(encoding="utf-8"))
        if _norm(_needle) not in _t:
            continue
        # A mention that says it was withdrawn is fine; a bare restatement is
        # not. Prose wraps, so the retraction wording is looked for in the
        # PARAGRAPH containing the match and within 200 characters of it - a
        # two-line window let a table row next to a retraction paragraph pass,
        # which is how the first version of this check failed to fire on a
        # planted mention.
        for _m in re.finditer(re.escape(_norm(_needle)), _t):
            _a = _t.rfind("\n\n", 0, _m.start()) + 2
            _b = _t.find("\n\n", _m.end())
            _b = len(_t) if _b < 0 else _b
            _para = _t[max(_a, _m.start() - 200):min(_b, _m.end() + 200)].lower()
            if not any(w in _para for w in
                       ("until", "withdraw", "earlier", "was wrong", "retract",
                        "double", "added a second time", "before that correction",
                        "corrected", "both were wrong", "an earlier version")):
                _hits.append(f"{_f}: ...{_t[_m.start() - 40:_m.end() + 20].strip()}...")
    chk(f"withdrawn: {_needle} is only mentioned as withdrawn", _hits, [])


print("\n=== the file inventory is counted, not typed ===")
# The v4 Files table listed two of the run directories and called itself the
# file list. Counts in prose go stale silently; these are derived.
_root = pathlib.Path(__file__).resolve().parents[1]
_dirs = sorted(d for d in (_root / "v4_audit_2026_08_25" / "data").iterdir() if d.is_dir())
_armruns = sorted((_root / "v4_audit_2026_08_25" / "data").glob("*/*__rep*.json"))
_v4 = " ".join(_norm((_root / "v4_audit_2026_08_25" / "README.md")
                     .read_text(encoding="utf-8")).split())
chk("v4 README states the run-directory count",
    f"{len(_dirs)} directories, {len(_armruns)} arm-runs" in _v4, True)
chk("every run directory carries a manifest",
    sorted(d.name for d in _dirs if not (d / "manifest.json").is_file()), [])
_attested = [d for d in _dirs if (d / "RUN_COMPLETE.json").is_file()]
chk("attested runs carry the completeness marker", len(_attested) >= 3, True)
# `paired_blocks.py` writes its output INTO the run directory, so an
# exploratory invocation with different options silently replaces committed
# data. Two of these were committed at --iters=2000 from a run I made while
# looking at something else.
_pbs = {os.path.basename(os.path.dirname(f)):
        json.load(open(f)).get("bootstrap_iters")
        for f in glob.glob("v4_audit_2026_08_25/data/*/paired_blocks.json")}
chk("every committed paired_blocks.json used the default iteration count",
    sorted(k for k, v in _pbs.items() if v != 20000), [])
chk("run directories carrying one", len(_pbs) >= 5, True)
chk("no run directory carries a failure marker",
    sorted(d.name for d in _dirs if (d / "RUN_FAILED.json").is_file()), [])


print("\n=== nine of eleven --spec-type methods ===")
# The eleven are the whole of llama.cpp's table at
# common/speculative.cpp:33-43 on 3737e4137. Anything measured here appears in
# some arm-run's argv; `none` is the baseline, which passes no --spec-type at
# all and takes the default.
_ALL_SPEC_TYPES = [
    "none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
    "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
    "ngram-mod", "ngram-cache",
]
chk("llama.cpp offers this many --spec-type values", len(_ALL_SPEC_TYPES), 11)
_seen_types = {"none"}          # the baseline arm passes no flag
for _f in glob.glob("v4_audit_2026_08_25/data/*/*__rep*.json"):
    _a = json.load(open(_f)).get("argv") or []
    if "--spec-type" in _a:
        _seen_types.add(_a[_a.index("--spec-type") + 1])
chk("every measured value is one llama.cpp defines",
    sorted(_seen_types - set(_ALL_SPEC_TYPES)), [])
chk("methods measured here", len(_seen_types), 9)
chk("the two that were not",
    sorted(set(_ALL_SPEC_TYPES) - _seen_types), ["draft-dspark", "draft-eagle3"])
_docs = " ".join(_norm(" ".join(
    pathlib.Path(__file__).resolve().parents[1].joinpath(f).read_text(encoding="utf-8")
    for f in ("README.md", "ERRATA.md", "CHANGELOG.md",
              "v4_audit_2026_08_25/README.md"))).split())
chk("the documents say nine of eleven", "ine of master's eleven" in _docs
    or "ine of eleven" in _docs, True)
chk("and name both exclusions",
    "draft-eagle3" in _docs and "draft-dspark" in _docs, True)
chk("the exclusions are named where the claim is made",
    all(x in _norm(pathlib.Path(__file__).resolve().parents[1]
                   .joinpath("CHANGELOG.md").read_text(encoding="utf-8"))
        for x in ("draft-eagle3", "draft-dspark")), True)


print("\n=== the headline table, parsed cell by cell ===")
# Greping the document for a computed value proves the string exists somewhere,
# not that the row is right: changing the headline's "+26.3 %" to "+26.9 %"
# passed every check here, because "+26.3 %" still appeared in the replication
# table below it. The table is parsed and compared row by row instead.
_readme_lines = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("README.md").read_text(encoding="utf-8").splitlines()
_hdr = next(i for i, l in enumerate(_readme_lines)
            if l.startswith("| arm | pooled tok/s | change |"))
_table = {}
for _l in _readme_lines[_hdr + 2:]:
    if not _l.startswith("|"):
        break
    _c = [x.strip().strip("*").strip().strip("`").strip("*") for x in _l.strip("|").split("|")]
    if len(_c) < 6:
        continue
    _table[_c[0].replace("**", "").strip("`")] = _c[1:6]
chk("headline table rows parsed", len(_table), 9)


def _cell_pct(x):
    return float(_norm(x).replace("%", "").replace(" ", ""))


def _pooled_arm(d, a):
    n = ms = 0
    for f in glob.glob(f"{d}/{a}__rep*.json"):
        r = json.load(open(f))
        if r.get("crashed"):
            continue
        n += sum(x["predicted_n"] for x in r["rows"])
        ms += sum(x["predicted_ms"] for x in r["rows"])
    return 1000 * n / ms


_pb_o2 = {a["arm"]: a for a in json.load(open(f"{_O2}/paired_blocks.json"))["arms"]}
_base_o2 = _pooled_arm(_O2, "baseline")
for _arm, _cells in _table.items():
    _key = "baseline" if _arm.startswith("no speculation") else _arm
    _dn = _da = _ng = 0
    for _f in glob.glob(f"{_O2}/{_key}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _da += _x["draft_n_accepted"]
            _ng += _x["predicted_n"]
    chk(f"table row {_arm}: pooled", round(_pooled_arm(_O2, _key), 1),
        float(_cells[0]), 0.05)
    chk(f"table row {_arm}: draft/gen", round(_dn / _ng, 2), float(_cells[3]), 0.005)
    if _key == "baseline":
        continue
    chk(f"table row {_arm}: change",
        round(_pb_o2[_key]["point_pct"], 1), _cell_pct(_cells[1]), 0.05)
    _lo, _hi = [_cell_pct(x) for x in
                _norm(_cells[2]).strip("[]").split(",")]
    chk(f"table row {_arm}: interval",
        [round(x, 1) for x in _pb_o2[_key]["ci95_t_pct"]], [_lo, _hi], 0.05)
    chk(f"table row {_arm}: acceptance",
        round(100 * _da / _dn, 1), _cell_pct(_cells[4]), 0.05)


print("\n=== the headline draft/gen column ===")
# Acceptance without draft volume reads as success for an arm that never fires:
# ngram-map-k4v-m8 shows 50.0 % from 216 draft tokens over 27 000 generated.
_dg = {}
for _arm in json.load(open(f"{_O2}/manifest.json"))["arms"]:
    _dn = _da = _n = 0
    for _f in glob.glob(f"{_O2}/{_arm}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _da += _x["draft_n_accepted"]
            _n += _x["predicted_n"]
    _dg[_arm] = (_dn, _da, _n)
_rmd = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                      .joinpath("README.md").read_text(encoding="utf-8")).split())
for _arm, _want in (("spec-dflash-n2", 0.81), ("spec-mtp-n2", 0.77),
                    ("spec-dflash-n4", 1.24), ("ngram-map-k4v-m8", 0.01),
                    ("ngram-mod-n24", 0.19), ("ngram-cache", 0.17),
                    ("spec-draft-n8", 1.86), ("spec-draft-n1", 0.50)):
    _dn, _da, _n = _dg[_arm]
    chk(f"draft/gen for {_arm}", round(_dn / _n, 2), _want, 0.005)
chk("ngram-map-k4v-m8 drafted this many tokens", _dg["ngram-map-k4v-m8"][0], 216)
chk("of which this many were accepted", _dg["ngram-map-k4v-m8"][1], 108)
chk("out of this many generated", _dg["ngram-map-k4v-m8"][2], 27000)
chk("README says how little it drafts", "216 tokens across 27 000" in _rmd, True)
chk("the baseline drafts nothing", _dg["baseline"][0], 0)
_hh = {r["arm"]: r for r in json.load(open("analysis/plot_data.json"))["head_to_head"]["rows"]}
chk("the chart carries the same draft/gen as the table",
    sorted(a for a, v in _hh.items()
           if v.get("draft_per_generated") is not None
           and round(v["draft_per_generated"], 2) != round(_dg[a][0] / _dg[a][2], 2)), [])
chk("the chart plots the run the README documents",
    json.load(open("analysis/plot_data.json"))["head_to_head"]["run"],
    os.path.basename(_O2))


print("\n=== the headline interval column is the t interval, and says so ===")
# The table quoted the Student-t interval while the prose above it described a
# block bootstrap. Both are computed; only one is published, and which one is
# now stated and asserted.
_pb = json.load(open(f"{_O2}/paired_blocks.json"))
_README0 = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                          .joinpath("README.md").read_text(encoding="utf-8")).split())
for _a in _pb["arms"]:
    _lo, _hi = _a["ci95_t_pct"]
    _txt = f"[{_lo:+.1f} %, {_hi:+.1f} %]".replace("+0.0", "+0.0")
    chk(f"README quotes the t interval for {_a['arm']}",
        _norm(_txt) in _README0, True)
    _blo, _bhi = _a["ci95_boot_pct"]
    chk(f"t is the wider interval for {_a['arm']}",
        (_hi - _lo) >= (_bhi - _blo) - 1e-9, True)
chk("README names the interval it publishes",
    "95 % CI (t, over blocks)" in _README0, True)


print("\n=== run O3: the headline matrix, replicated with full provenance ===")
_O3 = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_O3_latin_*"))[0]
_mO3 = json.load(open(f"{_O3}/manifest.json"))
_mO2 = json.load(open(f"{_O2}/manifest.json"))
chk("O3 arm-runs", len(glob.glob(f"{_O3}/*__rep*.json")), 81)
chk("O3 is attested", os.path.isfile(f"{_O3}/RUN_COMPLETE.json"), True)
chk("O3 records the balance it achieved",
    _mO3.get("schedule_is_position_balanced"), True)
chk("O3 asserted the stock library per arm-run",
    (_mO3.get("expect_lib_sha256") or "")[:16], "a0cbe4d04bcda3f8")
for _k in ("target_sha256", "draft_sha256", "dflash_sha256", "mtp_sha256",
           "common_args", "max_tokens", "seed", "think", "concurrency", "ctx",
           "fit_target", "n_prompts", "arms", "repeats"):
    chk(f"O3 {_k} matches O2", _mO2.get(_k) == _mO3.get(_k), True)

_same = _diff = 0
for _arm in _mO2["arms"]:
    for _rep in range(9):
        _fa, _fb = f"{_O2}/{_arm}__rep{_rep}.json", f"{_O3}/{_arm}__rep{_rep}.json"
        if not (os.path.isfile(_fa) and os.path.isfile(_fb)):
            continue
        for _xa, _xb in zip(json.load(open(_fa))["rows"], json.load(open(_fb))["rows"]):
            _ka = json.dumps([_xa.get("tokens"), _xa.get("content"),
                              _xa.get("reasoning_content")], ensure_ascii=False,
                             sort_keys=True)
            _kb = json.dumps([_xb.get("tokens"), _xb.get("content"),
                              _xb.get("reasoning_content")], ensure_ascii=False,
                             sort_keys=True)
            if _ka == _kb:
                _same += 1
            else:
                _diff += 1
chk("O2 and O3 request-pairs compared", _same + _diff, 810)
chk("O2 and O3 outputs are byte-identical", _diff, 0)


def _rel(d, arm):
    def _p(a):
        n = ms = 0
        for f in glob.glob(f"{d}/{a}__rep*.json"):
            r = json.load(open(f))
            if r.get("crashed"):
                continue
            n += sum(x["predicted_n"] for x in r["rows"])
            ms += sum(x["predicted_ms"] for x in r["rows"])
        return 1000 * n / ms if ms else None
    return 100 * (_p(arm) / _p("baseline") - 1)


_shifts = {a: round(_rel(_O3, a) - _rel(_O2, a), 1)
           for a in _mO2["arms"] if a != "baseline"}
chk("O3 shift on spec-dflash-n2 (pp)", _shifts["spec-dflash-n2"], -2.9, 0.05)
chk("O3: every other arm moves by at most 1.0 pp",
    sorted(a for a, v in _shifts.items()
           if a != "spec-dflash-n2" and abs(v) > 1.0), [])
chk("O3: the DFlash arm moves further than any other",
    max(_shifts, key=lambda a: abs(_shifts[a])), "spec-dflash-n2")
_acc_diff = []
for _arm in _mO2["arms"]:
    if _arm == "baseline":
        continue
    _v = []
    for _d in (_O2, _O3):
        _dn = _da = 0
        for _f in glob.glob(f"{_d}/{_arm}__rep*.json"):
            for _x in json.load(open(_f))["rows"]:
                _dn += _x["draft_n"]
                _da += _x["draft_n_accepted"]
        _v.append(100 * _da / _dn if _dn else 0.0)
    _acc_diff.append(abs(_v[0] - _v[1]))
chk("O3 acceptance matches O2 to a tenth of a point on every arm",
    round(max(_acc_diff), 2) <= 0.1, True)
_rmo3 = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                       .joinpath("README.md").read_text(encoding="utf-8")).split())
chk("README states the replication", "810 request-pairs are byte-identical" in _rmo3, True)
# the two intervals A16 says barely overlap
_pb2 = {a["arm"]: a["ci95_t_pct"] for a in
        json.load(open(f"{_O2}/paired_blocks.json"))["arms"]}
_pb3 = {a["arm"]: a["ci95_t_pct"] for a in
        json.load(open(f"{_O3}/paired_blocks.json"))["arms"]}
chk("O2's interval on the DFlash arm",
    [round(x, 1) for x in _pb2["spec-dflash-n2"]], [25.5, 27.1])
chk("O3's interval on the DFlash arm",
    [round(x, 1) for x in _pb3["spec-dflash-n2"]], [21.4, 25.6])
chk("they overlap by less than either is wide",
    round(min(_pb2["spec-dflash-n2"][1], _pb3["spec-dflash-n2"][1])
          - max(_pb2["spec-dflash-n2"][0], _pb3["spec-dflash-n2"][0]), 1), 0.1, 0.05)
chk("ERRATA says they barely overlap",
    "barely overlap" in _norm(pathlib.Path(__file__).resolve().parents[1]
                              .joinpath("ERRATA.md").read_text(encoding="utf-8")), True)


print("\n=== the O2/O3 replication table and the footnote, parsed ===")
# Same weakness as the headline table had: greping for a value proves the string
# exists, not that the row says it. Both are parsed.
_rl = pathlib.Path(__file__).resolve().parents[1].joinpath("README.md") \
    .read_text(encoding="utf-8").splitlines()
_rh = next(i for i, l in enumerate(_rl) if l.startswith("| arm | O2 | O3 | shift |"))
_rep_rows = {}
for _l in _rl[_rh + 2:]:
    if not _l.startswith("|"):
        break
    _c = [x.strip().strip("*").strip().strip("`").strip("*").strip()
          for x in _l.strip("|").split("|")]
    if len(_c) >= 4:
        _rep_rows[_c[0]] = _c[1:4]
chk("replication table rows parsed", len(_rep_rows), 9)


def _num(x):
    return float(_norm(x).replace("%", "").replace("pp", "").replace(" ", ""))


for _arm, _c in _rep_rows.items():
    if _arm.startswith("no speculation"):
        chk("replication table: O2 baseline pooled",
            round(_pooled_arm(_O2, "baseline"), 1), _num(_c[0]), 0.05)
        chk("replication table: O3 baseline pooled",
            round(_pooled_arm(_O3, "baseline"), 1), _num(_c[1]), 0.05)
        chk("replication table: baseline change (%)",
            round(100 * (_pooled_arm(_O3, "baseline") / _pooled_arm(_O2, "baseline") - 1), 1),
            _num(_c[2]), 0.05)
        continue
    chk(f"replication table {_arm}: O2", round(_rel(_O2, _arm), 1), _num(_c[0]), 0.05)
    chk(f"replication table {_arm}: O3", round(_rel(_O3, _arm), 1), _num(_c[1]), 0.05)
    chk(f"replication table {_arm}: shift",
        round(_rel(_O3, _arm) - _rel(_O2, _arm), 1), _num(_c[2]), 0.05)

# the footnote's six measurements, each against the run it names
_TAGDIR = {}
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/matrix_*")):
    _mp = os.path.join(_d, "manifest.json")
    if os.path.isfile(_mp):
        _TAGDIR[os.path.basename(_d).split("_")[1]] = _d
_foot = " ".join(_norm(" ".join(_rl)).split())
_fn = re.findall(r"(O2|O3|M1|T3|U[1-6]|O|T) \*\*([+-][0-9.]+) %\*\* (\d\d:\d\d)", _foot)
chk("footnote measurements parsed", len(_fn), 12)
for _tag, _val, _clock in _fn:
    chk(f"footnote {_tag}: the value it prints",
        round(_rel(_TAGDIR[_tag], "spec-dflash-n2"), 1), float(_val), 0.05)
    chk(f"footnote {_tag}: the clock time it prints",
        json.load(open(f"{_TAGDIR[_tag]}/manifest.json"))["created"][11:16], _clock)


print("\n=== run U: within an invocation against between invocations ===")
# The designed test behind A16. Six independent invocations of one script,
# fifteen minutes apart, each two balanced blocks of {baseline, spec-dflash-n2},
# on the stock binary asserted per arm-run.
_U = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_U*_dflashvar_*"))
chk("U invocations", len(_U), 6)
chk("every U invocation is attested",
    sorted(d for d in _U if not os.path.isfile(f"{d}/RUN_COMPLETE.json")), [])
chk("every U invocation is position-balanced",
    sorted({json.load(open(f"{d}/manifest.json"))["schedule_is_position_balanced"]
            for d in _U}), [True])
chk("every U invocation asserted the stock library",
    sorted({(json.load(open(f"{d}/manifest.json")).get("expect_lib_sha256") or "")[:16]
            for d in _U}), ["a0cbe4d04bcda3f8"])


def _u_blocks(d, arm):
    out = {}
    for f in glob.glob(f"{d}/{arm}__rep*.json"):
        r = json.load(open(f))
        out[r["repeat"]] = (1000 * sum(x["predicted_n"] for x in r["rows"])
                            / sum(x["predicted_ms"] for x in r["rows"]))
    return out


_u_within, _u_means = [], []
for _d in _U:
    _b, _a = _u_blocks(_d, "baseline"), _u_blocks(_d, "spec-dflash-n2")
    _urel = [100 * (_a[k] / _b[k] - 1) for k in sorted(_b) if k in _a]
    _u_within.append(st.stdev(_urel))
    _u_means.append(st.mean(_urel))
chk("U: within-invocation SD of the block ratios (pp)",
    round(st.mean(_u_within), 2), 0.55, 0.005)
chk("U: between-invocation SD of the six means (pp)",
    round(st.stdev(_u_means), 2), 3.15, 0.005)
chk("U: the variance ratio",
    round(st.stdev(_u_means) ** 2 / st.mean(_u_within) ** 2), 33, 0.5)
chk("U: range across the six invocations (pp)",
    round(max(_u_means) - min(_u_means), 1), 8.3, 0.05)
chk("U: the lowest invocation (%)", round(min(_u_means), 1), 17.3, 0.05)
chk("U: the highest invocation (%)", round(max(_u_means), 1), 25.6, 0.05)

_u_same = _u_diff = 0
for _arm in ("baseline", "spec-dflash-n2"):
    for _rep in (0, 1):
        _ref = None
        for _d in _U:
            _h = [hashlib.sha256(json.dumps(
                [x.get("tokens"), x.get("content"), x.get("reasoning_content")],
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                for x in json.load(open(f"{_d}/{_arm}__rep{_rep}.json"))["rows"]]
            if _ref is None:
                _ref = _h
            for _x, _y in zip(_ref, _h):
                if _x == _y:
                    _u_same += 1
                else:
                    _u_diff += 1
chk("U: request-pairs compared across invocations", _u_same + _u_diff, 240)
chk("U: all of them byte-identical", _u_diff, 0)

# the reference is steady while the arm under test is not. The comparable set
# is rebuilt here rather than borrowed, because this section runs before the
# footnote's.
_TGT_SHA = "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"
_comparable_dirs = []
for _mp in sorted(glob.glob("v4_audit_2026_08_25/data/matrix_*/manifest.json")):
    _mm = json.load(open(_mp))
    if (_mm.get("think") == "on" and _mm.get("concurrency") == 1
            and _mm.get("prompt_set", "v1") == "v1" and str(_mm.get("ctx")) == "8192"
            and str(_mm.get("fit_target")) == "3072"
            and _mm.get("target_sha256") == _TGT_SHA
            and "spec-dflash-n2" in (_mm.get("arms") or {})):
        _comparable_dirs.append(os.path.dirname(_mp))
chk("comparable measurements of the DFlash arm", len(_comparable_dirs), 12)
_dfl_base = []
for _d in _comparable_dirs:
    _n = _ms = 0
    for _f in glob.glob(f"{_d}/baseline__rep*.json"):
        _r = json.load(open(_f))
        _n += sum(x["predicted_n"] for x in _r["rows"])
        _ms += sum(x["predicted_ms"] for x in _r["rows"])
    _dfl_base.append(1000 * _n / _ms)
chk("the baseline's CV across the twelve comparable runs (%)",
    round(100 * st.stdev(_dfl_base) / st.mean(_dfl_base), 2), 0.42, 0.005)
chk("the baseline's range across them",
    [round(min(_dfl_base), 2), round(max(_dfl_base), 2)], [115.72, 117.25], 0.005)
chk("README contrasts the steady reference with the moving arm",
    "CV of **0.42 %**" in " ".join(pathlib.Path(__file__).resolve().parents[1]
                                   .joinpath("README.md").read_text(encoding="utf-8").split()),
    True)


print("\n=== A16: how the 43 block measurements are distributed ===")
# Every block of every comparable run, for the one arm that moves.
_lvl, _lvl_meta = [], []
for _d in _comparable_dirs:
    def _blk(arm, d=_d):
        out = {}
        for f in glob.glob(f"{d}/{arm}__rep*.json"):
            r = json.load(open(f))
            out[r["repeat"]] = (1000 * sum(x["predicted_n"] for x in r["rows"])
                                / sum(x["predicted_ms"] for x in r["rows"]))
        return out
    _b, _a = _blk("baseline"), _blk("spec-dflash-n2")
    for _k in sorted(_b):
        if _k in _a:
            _lvl.append(100 * (_a[_k] / _b[_k] - 1))
            _lvl_meta.append((os.path.basename(_d).split("_")[1], _k))
chk("A16 block-level measurements of the DFlash arm", len(_lvl), 43)
_hi = [x for x in _lvl if x >= 23]
_lo = [x for x in _lvl if x < 23]
chk("A16 high level: n", len(_hi), 30)
chk("A16 high level: mean (%)", round(st.mean(_hi), 2), 25.70, 0.005)
chk("A16 high level: SD (pp)", round(st.stdev(_hi), 2), 1.18, 0.005)
chk("A16 low level: n", len(_lo), 13)
chk("A16 low level: mean (%)", round(st.mean(_lo), 2), 20.33, 0.005)
chk("A16 low level: SD (pp)", round(st.stdev(_lo), 2), 1.63, 0.005)
# the words the entry is allowed to use. "Two discrete levels" is not one of
# them: the widest gap in the sorted values isolates one run at the bottom, not
# the split, and two runs span more than 3 pp internally.
_sorted = sorted(_lvl)
_gaps = sorted(((_sorted[i + 1] - _sorted[i], _sorted[i]) for i in range(len(_sorted) - 1)),
               reverse=True)
chk("A16 the block-level range (%)",
    [round(min(_lvl), 1), round(max(_lvl), 1)], [17.0, 27.8], 0.05)
chk("A16 the widest gap in the sorted values (pp)", round(_gaps[0][0], 2), 2.06, 0.005)
chk("A16 and it is not at the +23 % split", round(_gaps[0][1], 1) < 23, True)
chk("A16 the gap at the split is the second widest (pp)",
    round(_gaps[1][0], 2), 1.32, 0.005)
_byrun = {}
for _v, (_t, _k) in zip(_lvl, _lvl_meta):
    _byrun.setdefault(_t, []).append(_v)
chk("A16 runs that cross the split", sorted(
    t for t, xs in _byrun.items() if any(x >= 23 for x in xs) and any(x < 23 for x in xs)),
    ["O3"])
chk("A16 runs that do not", len(_byrun) - 1, 11)
chk("A16 the largest within-run spread among the rest (pp)",
    round(max(max(xs) - min(xs) for t, xs in _byrun.items() if t != "O3"), 2), 3.27, 0.005)
chk("ERRATA withdraws the two-level wording",
    "does not support the word" in _norm(
        pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md")
        .read_text(encoding="utf-8")), True)
chk("and no longer calls it two discrete levels",
    "two levels, not noise" in _norm(
        pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md")
        .read_text(encoding="utf-8")), False)
chk("A16 the gap between them (pp)",
    round(st.mean(_hi) - st.mean(_lo), 1), 5.4, 0.05)

# identical speculative work in every one of them
_work = set()
for _d in _comparable_dirs:
    for _f in glob.glob(f"{_d}/spec-dflash-n2__rep*.json"):
        _r = json.load(open(_f))
        _work.add((sum(x["draft_n"] for x in _r["rows"]),
                   round(100 * sum(x["draft_n_accepted"] for x in _r["rows"])
                         / sum(x["draft_n"] for x in _r["rows"]), 1)))
chk("A16 the speculative work is identical in all 43", sorted(_work), [(2441, 72.3)])

# run O3 transitions mid-run, and only this arm moves with it
_o3 = {}
for _arm in json.load(open(f"{_O3}/manifest.json"))["arms"]:
    _o3[_arm] = {}
    for _f in glob.glob(f"{_O3}/{_arm}__rep*.json"):
        _r = json.load(open(_f))
        _o3[_arm][_r["repeat"]] = (1000 * sum(x["predicted_n"] for x in _r["rows"])
                                   / sum(x["predicted_ms"] for x in _r["rows"]))
_dip = [round(100 * (_o3["spec-dflash-n2"][k] / _o3["spec-dflash-n2"][0] - 1), 2)
        for k in (4, 5, 6, 7)]
chk("A16 O3's dip, blocks 4-7 against its own block 0 (%)",
    _dip, [-4.45, -4.66, -3.33, -2.93], 0.005)
_others = {a: max(abs(100 * (v[k] / v[0] - 1)) for k in range(9))
           for a, v in _o3.items() if a != "spec-dflash-n2"}
chk("A16 no other arm leaves 1.24 % of its own block 0",
    sorted(a for a, m in _others.items() if round(m, 2) > 1.24), [])
chk("A16 the largest excursion among them (%)",
    round(max(_others.values()), 2), 1.24, 0.005)
chk("A16 the same drafter at twice the draft length (%)",
    round(_others["spec-dflash-n4"], 2), 1.01, 0.005)
chk("A16 the dip is at least four times the next largest excursion",
    round(max(abs(x) for x in _dip) / max(_others.values()), 1) >= 3.7, True)
_tl = json.load(open("analysis/plot_data.json"))["two_levels"]
chk("the chart plots the same 43 blocks", _tl["n"], len(_lvl))
chk("the chart's high group matches", (_tl["high_n"], round(_tl["high_mean"], 2)),
    (len(_hi), round(st.mean(_hi), 2)))
chk("the chart's low group matches", (_tl["low_n"], round(_tl["low_mean"], 2)),
    (len(_lo), round(st.mean(_lo), 2)))
chk("ERRATA shows the chart",
    "analysis/plot_two_levels.png" in pathlib.Path(__file__).resolve().parents[1]
    .joinpath("ERRATA.md").read_text(encoding="utf-8"), True)
chk("ERRATA states the correction to the between-invocation framing",
    "This corrects the framing above" in _norm(
        pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md")
        .read_text(encoding="utf-8")), True)


print("\n=== run V: the length confound, measured instead of subsetted ===")
_VF = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_V_freerun_*"))[0]
_VH = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_V_hardcap_*"))[0]
_mVF = json.load(open(f"{_VF}/manifest.json"))
_mVH = json.load(open(f"{_VH}/manifest.json"))
chk("V: the two halves differ only in ignore_eos",
    [(_mVF.get("ignore_eos"), _mVH.get("ignore_eos"))] +
    [_mVF.get(k) == _mVH.get(k) for k in
     ("arms", "repeats", "think", "ctx", "fit_target", "prompt_set",
      "max_tokens", "seed", "target_sha256", "dflash_sha256", "mtp_sha256",
      "draft_sha256", "server_lib_sha256", "common_args")],
    [(False, True)] + [True] * 14)
chk("V: both halves are position-balanced",
    [_mVF["schedule_is_position_balanced"], _mVH["schedule_is_position_balanced"]],
    [True, True])
chk("V: both are attested",
    [os.path.isfile(f"{d}/RUN_COMPLETE.json") for d in (_VF, _VH)], [True, True])


def _v_lengths(d):
    out = set()
    for f in glob.glob(f"{d}/*__rep*.json"):
        for x in json.load(open(f))["rows"]:
            out.add(x["predicted_n"])
    return out


chk("V freerun: distinct output lengths", len(_v_lengths(_VF)), 14)
chk("V freerun: the shortest and longest",
    [min(_v_lengths(_VF)), max(_v_lengths(_VF))], [22, 300])
chk("V hardcap: every request generated exactly the cap",
    sorted(_v_lengths(_VH)), [300])
chk("V hardcap: and every one is a length stop",
    sorted({x.get("finish_reason") for f in glob.glob(f"{_VH}/*__rep*.json")
            for x in json.load(open(f))["rows"]}), ["length"])

# two decimals: two of these land on an exact half, and which way a half rounds
# is not something an assertion should depend on
for _arm, _free, _hard, _shift in (("spec-dflash-n2", 11.35, 20.60, 9.26),
                                   ("spec-mtp-n2", 11.50, 21.18, 9.68),
                                   ("spec-dflash-n4", -1.35, 10.55, 11.90),
                                   ("spec-draft-n8", -76.76, -70.45, 6.31)):
    chk(f"V {_arm}: freerun (%)", round(_rel(_VF, _arm), 2), _free, 0.005)
    chk(f"V {_arm}: hard cap (%)", round(_rel(_VH, _arm), 2), _hard, 0.005)
    chk(f"V {_arm}: the shift (pp)",
        round(_rel(_VH, _arm) - _rel(_VF, _arm), 2), _shift, 0.005)
chk("V: the arm A17 is about changes sign under the hard cap",
    (_rel(_VF, "spec-dflash-n4") < 0, _rel(_VH, "spec-dflash-n4") > 0), (True, True))
chk("V: every arm moves the way the subsetting said it would",
    sorted({round(_rel(_VH, a) - _rel(_VF, a), 1) > 0 for a in _mVF["arms"]
            if a != "baseline"}), [True])
# and the table itself, parsed - the cells were computed and asserted while the
# table was not, which is the third time that gap has appeared in this file
_vt_lines = pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md") \
    .read_text(encoding="utf-8").splitlines()
_vt_i = next(i for i, l in enumerate(_vt_lines)
             if l.startswith("| arm | freerun, as the archive did it | hard cap | shift |"))
_vt = {}
for _l in _vt_lines[_vt_i + 2:]:
    if not _l.startswith("|"):
        break
    _c = [x.strip().strip("*`").replace("`", "").strip("* ").strip()
          for x in _l.strip("|").split("|")]
    if len(_c) >= 4:
        _vt[_c[0]] = _c[1:4]
chk("A17 run V table rows parsed", len(_vt), 4)
for _arm, _cells in _vt.items():
    chk(f"A17 V table {_arm}: freerun",
        round(_rel(_VF, _arm), 2),
        float(_norm(_cells[0]).replace("%", "").replace(" ", "")), 0.005)
    chk(f"A17 V table {_arm}: hard cap",
        round(_rel(_VH, _arm), 2),
        float(_norm(_cells[1]).replace("%", "").replace(" ", "")), 0.005)
    chk(f"A17 V table {_arm}: the shift is the difference",
        round(_rel(_VH, _arm) - _rel(_VF, _arm), 2),
        float(_norm(_cells[2]).split()[0].replace("pp", "").replace(",", "")), 0.005)

_er_v = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                       .joinpath("ERRATA.md").read_text(encoding="utf-8")).split())
chk("ERRATA reports run V", "Run V measures it instead of subsetting" in _er_v, True)
chk("and says the two methods agree in direction, not magnitude",
    "agree in direction and not in magnitude" in _er_v, True)


print("\n=== the balanced design is verified, not declared ===")
# The README says run O2's Latin square was "verified from the execution log,
# not from the design", and nothing here re-derived it. `t_start` is
# `time.perf_counter()` inside the single driver process, so it is monotonic
# across the whole run and recovers the order the arms actually ran in from the
# committed arm-runs alone.


def _observed_positions(d):
    per_block: dict = {}
    for f in glob.glob(f"{d}/*__rep*.json"):
        r = json.load(open(f))
        if not r.get("rows"):
            continue
        per_block.setdefault(r["repeat"], []).append(
            (min(x["t_start"] for x in r["rows"]), r["arm"]))
    pos: dict = {}
    for rep in sorted(per_block):
        for i, (_, arm) in enumerate(sorted(per_block[rep])):
            pos.setdefault(arm, []).append(i + 1)
    return pos, len(per_block)


def _is_balanced(pos):
    n = len(pos)
    return bool(pos) and all(sorted(v) == list(range(1, n + 1)) for v in pos.values())


_o2pos, _o2blocks = _observed_positions(_O2)
chk("O2 blocks recovered from the monotonic clock", _o2blocks, 9)
chk("O2 arms recovered", len(_o2pos), 9)
chk("O2 is a full cyclic Latin square, verified from the arm-runs",
    _is_balanced(_o2pos), True)
chk("O2 every arm visits every position exactly once",
    sorted({tuple(sorted(v)) for v in _o2pos.values()}), [tuple(range(1, 10))])

_t3pos, _t3blocks = _observed_positions(_T3)
chk("T3 is position-balanced, verified from the arm-runs", _is_balanced(_t3pos), True)
chk("T3 blocks", _t3blocks, 3)

_tpos, _tblocks = _observed_positions(_T)
chk("T is NOT position-balanced, which is why T3 exists", _is_balanced(_tpos), False)
chk("T's baseline sits twice in position 1", _tpos["baseline"], [1, 3, 2, 1])
chk("T's manifest nonetheless recorded `latin`",
    json.load(open(f"{_T}/manifest.json")).get("order_mode"), "latin")


# The A12 volume table gives two runs. Run J's row is checked above; run T's is
# the one that matches the timing table beside it, and it was added only after a
# review pointed out that the section quoted one run's counts in one table and
# the other's in the next without saying so.
_SZ = 82.079
_t_creates, _t_restores = 785, 728
chk("A12 run T nominal volume written (GiB)",
    round(_t_creates * _SZ / 1024, 2), 62.92, 0.005)
chk("A12 run T nominal volume read back (GiB)",
    round(_t_restores * _SZ / 1024, 2), 58.35, 0.005)
chk("A12 run T nominal volume combined (GiB)",
    round((_t_creates + _t_restores) * _SZ / 1024, 2), 121.27, 0.005)
chk("A12 run T's counts are the ones the timing table used",
    sorted({(r["creates"], r["restores"]) for r in _tmr if r["arm"] == "spec-draft-n8"}),
    [(_t_creates, _t_restores)])
_er = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                     .joinpath("ERRATA.md").read_text(encoding="utf-8")).split())
chk("ERRATA gives both runs' volumes side by side",
    "121.27" in _er and "118.71" in _er, True)
chk("ERRATA says which run each timing table belongs to",
    "come from the same twelve logs" in _er, True)


print("\n=== A16: run T against run T3 ===")
_er_a16 = " ".join(_norm(pathlib.Path(__file__).resolve().parents[1]
                         .joinpath("ERRATA.md").read_text(encoding="utf-8")).split())


def _pool(d, arm):
    n = ms = 0
    for f in glob.glob(f"{d}/{arm}__rep*.json"):
        r = json.load(open(f))
        n += sum(x["predicted_n"] for x in r["rows"])
        ms += sum(x["predicted_ms"] for x in r["rows"])
    return (1000 * n / ms, ms / 1000, len(glob.glob(f"{d}/{arm}__rep*.json")))


for _arm, _want in (("baseline", 0.79), ("spec-draft-n8", -0.11),
                    ("spec-dflash-n2", -3.40)):
    _a = _pool(_T, _arm)[0]
    _b = _pool(_T3, _arm)[0]
    chk(f"A16 {_arm} T -> T3 (%)", round(100 * (_b / _a - 1), 2), _want, 0.005)

# the design that made T3 worth running at all
_mT = json.load(open(f"{_T}/manifest.json"))
_mT3 = json.load(open(f"{_T3}/manifest.json"))
# A16's thermal comparison, from the two traces now committed beside the runs
_TEL = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "data"


def _trace(name, util_key, thr_key):
    rows = list(_csv2.DictReader(open(_TEL / name, encoding="utf-8")))
    out = []
    for r in rows:
        try:
            u = float((r[util_key] or "").strip().split()[0])
        except Exception:  # noqa: BLE001
            continue
        out.append((u, r))
    return out


def _mean(rows, key):
    v = []
    for _u, r in rows:
        try:
            v.append(float((r[key] or "").strip().split()[0]))
        except Exception:  # noqa: BLE001
            pass
    return st.mean(v) if v else float("nan")


_tT = [(u, r) for u, r in _trace("gpu_telemetry_T_20260826_182639.csv",
                                 "util", "throttle") if u >= 50]
_tT3 = [(u, r) for u, r in _trace("gpu_telemetry_T3_20260826_203251.csv",
                                  " utilization.gpu [%]",
                                  " clocks_event_reasons.active") if u >= 50]
chk("A16 T loaded telemetry samples", len(_tT), 156)
chk("A16 T3 loaded telemetry samples", len(_tT3), 599)
chk("A16 mean temperature, T (C)", round(_mean(_tT, "temp"), 1), 63.5, 0.05)
chk("A16 mean temperature, T3 (C)", round(_mean(_tT3, " temperature.gpu"), 1), 63.6, 0.05)
chk("A16 mean SM clock, T (MHz)", round(_mean(_tT, "clk_sm")), 1946, 0.5)
chk("A16 mean SM clock, T3 (MHz)",
    round(_mean(_tT3, " clocks.current.sm [MHz]")), 1947, 0.5)
chk("A16 mean board power, T (W)", round(_mean(_tT, "pwr"), 1), 240.3, 0.05)
chk("A16 mean board power, T3 (W)", round(_mean(_tT3, " power.draw [W]"), 1), 240.1, 0.05)


def _bits(rows, key, bit):
    n = 0
    for _u, r in rows:
        try:
            if int((r[key] or "0").strip(), 16) & bit:
                n += 1
        except ValueError:
            pass
    return n


# and the tool named in the documents must actually produce those numbers
_thermal = _sp2.run(
    [_sys2.executable, str(pathlib.Path(__file__).resolve().parents[1]
                           / "analysis" / "thermal_report.py"),
     "v4_audit_2026_08_25/data/gpu_telemetry_20260825.csv",
     "v4_audit_2026_08_25/data/gpu_telemetry_T_20260826_182639.csv",
     "v4_audit_2026_08_25/data/gpu_telemetry_T3_20260826_203251.csv"],
    capture_output=True, text=True, timeout=120).stdout
chk("thermal_report reproduces C4b's power-cap count", "636 / 1272" in _thermal, True)
chk("thermal_report reproduces C4b's sw-thermal count", "2 / 1272" in _thermal, True)
chk("thermal_report reproduces C4b's hw-thermal count", "1 / 1272" in _thermal, True)
chk("thermal_report reproduces A16's T power-cap count", "29 / 156" in _thermal, True)
chk("thermal_report reproduces A16's T3 power-cap count", "27 / 599" in _thermal, True)
chk("thermal_report reads all three schemas",
    sorted(x for x in ("schema=full", "schema=compact", "schema=raw")
           if x not in _thermal), [])
chk("thermal_report states the sampling interval it found",
    "sampling 5s" in _thermal and "sampling 1s" in _thermal, True)
chk("A16 sw_power_cap samples, T", _bits(_tT, "throttle", 0x4), 29)
chk("A16 sw_power_cap samples, T3",
    _bits(_tT3, " clocks_event_reasons.active", 0x4), 27)
chk("A16 no thermal-slowdown flag on any loaded sample of either run",
    (_bits(_tT, "throttle", 0x20) + _bits(_tT, "throttle", 0x40)
     + _bits(_tT3, " clocks_event_reasons.active", 0x20)
     + _bits(_tT3, " clocks_event_reasons.active", 0x40)), 0)
chk("ERRATA no longer claims the two throttle rates are the same",
    "the same software power-cap events at the same rate" in _er_a16, False)
chk("ERRATA says the fractions are not comparable",
    "cannot be set beside each other" in _er_a16, True)
chk("A16 T3 is position-balanced", _mT3.get("schedule_is_position_balanced"), True)
chk("A16 T3 asserted the instrumented library per arm-run",
    (_mT3.get("expect_lib_sha256") or "")[:16], "ce94855f4f2d82ba")
chk("A16 T3 asserted the commit", _mT3.get("expect_commit"), "3737e4137")
# `expect_*` is what the caller demanded. What answered is per arm-run, and that
# is the field worth checking: a manifest can record an expectation that was
# never met if the marker were written without validation, which is the exact
# defect A16's run exists after fixing.
_obs_commit, _obs_lib, _obs_log = set(), [], 0
for _f in sorted(glob.glob(f"{_T3}/*__rep*.json")):
    _r = json.load(open(_f))
    _obs_commit.add(_r.get("server_loaded_commit"))
    _obs_lib.append((_r.get("server_lib_sha256") or {}).get("libllama-server-impl.so"))
    if re.fullmatch(r"[0-9a-f]{64}", str(_r.get("server_log_sha256") or "")):
        _obs_log += 1
chk("A16 every T3 arm-run reports the commit it was asked for",
    sorted(_obs_commit), ["3737e4137"])
chk("A16 every T3 arm-run reports the instrumented library",
    (len(_obs_lib), sum(1 for x in _obs_lib if (x or "").startswith("ce94855f"))),
    (9, 9))
chk("A16 every T3 arm-run carries a hash of its own server log", _obs_log, 9)
chk("A16 T3's arm-runs and the run's own count agree",
    len(glob.glob(f"{_T3}/*__rep*.json")),
    json.load(open(f"{_T3}/RUN_COMPLETE.json"))["observed_arm_runs"])
for _k in ("target_sha256", "draft_sha256", "dflash_sha256", "server_lib_sha256",
           "common_args", "max_tokens", "seed", "think", "concurrency", "ctx",
           "fit_target", "n_prompts", "arms"):
    chk(f"A16 {_k} is identical in T and T3", _mT.get(_k) == _mT3.get(_k), True)

# byte-identical output is the whole point of the entry
_sig = {}
for _lbl, _d in (("T", _T), ("T3", _T3)):
    for _arm in _mT["arms"]:
        _r = json.load(open(f"{_d}/{_arm}__rep0.json"))
        _sig[(_lbl, _arm)] = [
            (x["tag"], hashlib.sha256(json.dumps(
                [x.get("tokens"), x.get("content"), x.get("reasoning_content")],
                ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            for x in _r["rows"]]
for _arm in _mT["arms"]:
    chk(f"A16 {_arm} output is byte-identical across T and T3",
        _sig[("T", _arm)] == _sig[("T3", _arm)], True)

# the attribution replicates; the headline arm does not, at this precision
_ck3 = st.mean([r["checkpoint_total_s"] for r in
                json.load(open(f"{_T3}/checkpoint_timers.json"))
                if r["arm"] == "spec-draft-n8"])
# the per-prompt claims A16 makes, each one checked rather than summarised
_pp: dict = {}
for _lbl, _d in (("T", _T), ("T3", _T3)):
    for _f in glob.glob(f"{_d}/*__rep*.json"):
        _r = json.load(open(_f))
        for _x in _r["rows"]:
            _k = (_lbl, _r["arm"], _x["tag"])
            _v = _pp.setdefault(_k, [0, 0, 0, 0.0])
            _v[0] += _x["draft_n"]
            _v[1] += _x["draft_n_accepted"]
            _v[2] += _x["predicted_n"]
            _v[3] += _x["predicted_ms"]
_arms3 = sorted({k[1] for k in _pp})
_acc_gap, _ratio_gap, _rate_gap = [], [], []
for _arm in _arms3:
    for _tag in sorted({k[2] for k in _pp if k[1] == _arm}):
        _a, _b = _pp[("T", _arm, _tag)], _pp[("T3", _arm, _tag)]
        if _a[0] and _b[0]:
            _acc_gap.append(abs(100 * _a[1] / _a[0] - 100 * _b[1] / _b[0]))
            _ratio_gap.append(abs(_a[0] / _a[2] - _b[0] / _b[2]))
        _rate_gap.append(100 * ((_b[2] / _b[3]) / (_a[2] / _a[3]) - 1))
chk("A16 acceptance matches to a tenth of a point on every prompt",
    round(max(_acc_gap), 2) <= 0.1, True)
chk("A16 draft tokens per generated token match to three decimals",
    round(max(_ratio_gap), 4) < 0.0005, True)
_dflash = []
for _tag in sorted({k[2] for k in _pp if k[1] == "spec-dflash-n2"}):
    _a, _b = _pp[("T", "spec-dflash-n2", _tag)], _pp[("T3", "spec-dflash-n2", _tag)]
    _dflash.append(100 * ((_b[2] / _b[3]) / (_a[2] / _a[3]) - 1))
chk("A16 the DFlash shortfall is on every prompt",
    sorted({x < 0 for x in _dflash}), [True])
chk("A16 the smallest per-prompt DFlash shortfall (%)",
    round(max(_dflash), 1), -0.6, 0.05)
chk("A16 the largest per-prompt DFlash shortfall (%)",
    round(min(_dflash), 1), -4.7, 0.05)

chk("A16 T3 checkpoint total (s)", round(_ck3, 2), 39.16, 0.005)
chk("A16 T3 creates and restores match run T",
    sorted({(r["creates"], r["restores"]) for r in
            json.load(open(f"{_T3}/checkpoint_timers.json"))
            if r["arm"] == "spec-draft-n8"}), [(785, 728)])
_ex3 = _pool(_T3, "spec-draft-n8")[1] / 3 - _pool(_T3, "baseline")[1] / 3
chk("A16 T3 checkpoint share of the excess (%)", round(100 * _ck3 / _ex3, 1), 54.6, 0.05)
chk("ERRATA records that the cause is not isolated",
    "The cause is not isolated" in _norm(
        pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md")
        .read_text(encoding="utf-8")), True)


print("\n=== the headline footnote is derived, not typed ===")
# The footnote under the headline table quoted run O's +24.6 % as this table's
# own figure for eleven days, because it was written when run O was the headline
# and nothing tied it to the data when O2 replaced it. Each figure below is
# recomputed and the README is required to contain exactly that string.
_TGT = "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"


def _pooled_of(d, arm):
    tot_n = tot_ms = 0
    for f in glob.glob(f"{d}/{arm}__rep*.json"):
        r = json.load(open(f))
        if r.get("crashed"):
            continue
        tot_n += sum(x["predicted_n"] for x in r["rows"])
        tot_ms += sum(x["predicted_ms"] for x in r["rows"])
    return (1000 * tot_n / tot_ms) if tot_ms else None


_comparable = {}
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/matrix_*")):
    _mp = os.path.join(_d, "manifest.json")
    if not os.path.isfile(_mp):
        continue
    _m = json.load(open(_mp))
    # every knob that changes what is being measured, held fixed
    if not (_m.get("think") == "on" and _m.get("concurrency") == 1
            and _m.get("prompt_set", "v1") == "v1" and str(_m.get("ctx")) == "8192"
            and str(_m.get("fit_target")) == "3072"
            and _m.get("target_sha256") == _TGT and "baseline" in _m["arms"]):
        continue
    _b = _pooled_of(_d, "baseline")
    if not _b:
        continue
    _tag = os.path.basename(_d).split("_")[1]
    _comparable[_tag] = {a: 100 * (_pooled_of(_d, a) / _b - 1)
                         for a in _m["arms"] if a != "baseline" and _pooled_of(_d, a)}

_README = _norm(pathlib.Path(__file__).resolve().parents[1].joinpath("README.md")
                .read_text(encoding="utf-8"))
for _tag in ("O", "M1", "O2", "T", "T3", "O3",
             "U1", "U2", "U3", "U4", "U5", "U6"):
    _v = _comparable.get(_tag, {}).get("spec-dflash-n2")
    chk(f"footnote: run {_tag} spec-dflash-n2 is a real run", _v is not None, True)
    if _v is not None:
        chk(f"README quotes run {_tag}'s {_v:+.1f} %",
            _norm(f"{_v:+.1f} %").replace("+", "+") in _README, True)

_dfl = {t: v["spec-dflash-n2"] for t, v in _comparable.items() if "spec-dflash-n2" in v}
chk("footnote: spec-dflash-n2 was measured in twelve comparable runs", len(_dfl), 12)
_spread = max(_dfl.values()) - min(_dfl.values())
chk("footnote: the between-run spread (pp)", round(_spread, 1), 9.4, 0.05)
chk("README quotes that spread", "Range 9.4 pp" in _README, True)

_all_spreads = sorted(
    (max(v.values()) - min(v.values()) for v in
     ({t: c[a] for t, c in _comparable.items() if a in c}
      for a in {a for c in _comparable.values() for a in c})
     if len(v) > 1), reverse=True)
chk("footnote: spec-dflash-n2's spread is now the largest",
    _all_spreads.index(round(_spread, 10)) if round(_spread, 10) in _all_spreads
    else [round(x, 10) for x in _all_spreads].index(round(_spread, 10)), 0)
chk("footnote: run U3 is the lowest, so O2 is not", min(_dfl, key=_dfl.get), "U3")
chk("README names the lowest", "run U3 is, at +17.3 %" in _README, True)
chk("README says the interval describes the run, not the configuration",
    "describe run O2, not the configuration" in " ".join(_README.split()), True)
chk("README quotes the between-run range",
    "+17.3 % to +26.7 %" in _norm(_README), True)
chk("and the range is what the twelve runs actually span",
    [round(min(_dfl.values()), 1), round(max(_dfl.values()), 1)], [17.3, 26.7], 0.05)
chk("README no longer claims the lower of the two is quoted",
    "The lower of the two is quoted" in _README, False)


print("\n=== the checker audits itself ===")
# A chk() whose computed side contains no name is comparing one literal with
# another and can never fail. Six of these were found on 2026-08-26, four of them
# in the newest ERRATA item, where care had slipped. They read as passes and
# certify nothing, so the checker now refuses to be one of them.
import ast as _ast
import pathlib as _pl2
_src = _pl2.Path(__file__).read_text(encoding="utf-8")
_tree = _ast.parse(_src)
_SAFE = {"round", "abs", "sorted", "len", "max", "min", "sum", "int", "float", "str",
         "tuple", "list", "set", "True", "False", "None"}
_literal_only = []
for _n in _ast.walk(_tree):
    if not (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name)
            and _n.func.id == "chk" and len(_n.args) >= 2):
        continue
    _got = _n.args[1]
    _names = {x.id for x in _ast.walk(_got) if isinstance(x, _ast.Name)} \
           | {x.attr for x in _ast.walk(_got) if isinstance(x, _ast.Attribute)}
    if not (_names - _SAFE):
        _label = _n.args[0]
        _literal_only.append(getattr(_label, "value", None)
                             or _ast.unparse(_label)[:60])
chk("checker: no chk() compares literals with literals",
    (len(_literal_only), _literal_only[:3]), (0, []))
chk("checker: number of assertions", len([1 for _n in _ast.walk(_tree)
     if isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name) and _n.func.id == "chk"]) > 150, True)

print(f"\n{'='*70}\n{'ALL CLAIMS VERIFIED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}\n{'='*70}")
sys.exit(1 if FAIL else 0)
