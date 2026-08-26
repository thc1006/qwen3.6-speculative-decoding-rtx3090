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
import csv, json, glob, math, os, re, statistics as st
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
chk("A13 single-request arm-runs with both counters", len(_seq), 73)
_z = [r for r in _seq if r["checkpoints_created"] == 0]
_n = [r for r in _seq if r["checkpoints_created"] > 0]
chk("A13 arm-runs that take no checkpoint", len(_z), 31)
chk("A13 arm-runs that do", len(_n), 42)
chk("A13 no-checkpoint: largest gap between the counters (pp)",
    round(max(abs(r["server_pct"] - r["drafter_pct"]) for r in _z), 1), 0.5, 0.05)
chk("A13 checkpointing: smallest gap between the counters (pp)",
    round(min(abs(r["server_pct"] - r["drafter_pct"]) for r in _n), 1), 1.0, 0.05)
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
import csv as _csv2, re as _re2
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
_cc2 = {(r["run"], r["arm"]): r for r in
        json.load(open("v4_audit_2026_08_25/data/acceptance_counter_comparison.json"))}
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
chk("threshold: arm-runs with an acceptance figure and a baseline", len(_all), 44)
chk("threshold: excluded for drafting under 100 tokens", len(_all) - len(_kept), 7)
chk("threshold: the excluded ones drafted at most this many tokens",
    max(x[6] for x in _all if x[6] < 100), 55)
chk("threshold: the kept ones drafted at least this many",
    min(x[6] for x in _kept), 586)
def _sc(rows, idx):
    from collections import defaultdict as _dd
    f = _dd(lambda: [0, 0])
    for x in rows:
        f[x[0]][0] += (x[idx] >= _BRK) == (x[5] > 0); f[x[0]][1] += 1
    return f
for idx, lbl in ((3, "server"), (4, "drafter")):
    f = _sc(_kept, idx)
    chk(f"threshold ({lbl}): self-speculative", tuple(f["self"]), (28, 29))
    chk(f"threshold ({lbl}): drafter-free n-gram", tuple(f["ngram"]), (2, 2))
    chk(f"threshold ({lbl}): external drafter", tuple(f["external"]), (5, 6))
    chk(f"threshold ({lbl}): overall",
        (sum(v[0] for v in f.values()), sum(v[1] for v in f.values())), (35, 37))
# without the exclusion the two counters disagree, which is why it exists
chk("threshold: without the exclusion the counters disagree",
    (sum(v[0] for v in _sc(_all, 3).values()), sum(v[0] for v in _sc(_all, 4).values())), (41, 37))
# and the two misses are the ones named in the text
_miss = sorted(f"{x[2]}" for x in _kept if (x[3] >= _BRK) != (x[5] > 0))
chk("threshold: which arms it gets wrong", _miss, ["spec-draft-n1", "spec-mtp-n4"])
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
_tm = {r["arm"]: r for r in
       json.load(open("v4_audit_2026_08_25/data/checkpoint_timers_20260826.json"))}
_ext = [r for k, r in _tm.items() if k.startswith("spec-draft-n8")]
chk("A12 timed arm-runs of the external drafter", len(_ext), 4)
chk("A12 checkpoint creates per arm-run", sorted({r["creates"] for r in _ext}), [785])
chk("A12 restores per arm-run", sorted({r["restores"] for r in _ext}), [728])
chk("A12 update_dft on the speculative checkpoint never fires",
    sorted({r["update_dft_s"] for r in _ext}), [0.0])
chk("A12 update_tgt seconds", round(st.mean([r["update_tgt_s"] for r in _ext]), 2), 17.34, 0.005)
chk("A12 load_tgt seconds", round(st.mean([r["load_tgt_s"] for r in _ext]), 2), 16.33, 0.005)
chk("A12 load_dft seconds", round(st.mean([r["load_dft_s"] for r in _ext]), 2), 5.41, 0.005)
_ck = st.mean([r["checkpoint_total_s"] for r in _ext])
chk("A12 checkpoint total seconds", round(_ck, 2), 39.08, 0.005)
chk("A12 the total is reproducible across arm-runs",
    round(max(r["checkpoint_total_s"] for r in _ext) -
          min(r["checkpoint_total_s"] for r in _ext), 2) <= 0.05, True)
chk("A12 DFlash performs no checkpoint operations",
    _tm["spec-dflash-n2"]["checkpoint_total_s"], 0.0)
chk("A12 the baseline performs none either",
    _tm["baseline"]["checkpoint_total_s"], 0.0)

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
    ("README.md",   "48 % acceptance", "README acceptance threshold"),
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
    ("v4_audit_2026_08_25/README.md", "35 / 37", "threshold scorecard"),
    ("v4_audit_2026_08_25/README.md", "28 / 29", "threshold, self-speculative"),
    ("README.md",   "28 / 29",   "README threshold scorecard"),
    ("ERRATA.md",   "8.5 pp",    "A14 the pair that did not replicate"),
    ("README.md",   "+26.7 %",   "README discloses the same-config replicate"),
    ("v4_audit_2026_08_25/README.md", "292.1 s", "P pooled includes the draft cost"),
    ("v4_audit_2026_08_25/README.md", "72.8 %",  "P acceptance not inflated"),
    ("ERRATA.md",   "39.08",     "A12 measured checkpoint total"),
    ("ERRATA.md",   "54.7 %",    "A12 checkpoint share"),
    ("ERRATA.md",   "21.9 ms",   "A12 median create"),
    ("README.md",   "39.08 s",   "README checkpoint total"),
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
]
root = pathlib.Path(__file__).resolve().parents[1]
for f, needle, what in DOC_CLAIMS:
    txt = _norm((root / f).read_text(encoding="utf-8"))
    ok = _norm(needle) in txt
    print(f"  {'PASS' if ok else 'FAIL'}  {f:32s} quotes {needle!r:20s} ({what})")
    if not ok:
        FAIL.append(f"{f}:{needle}")


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
