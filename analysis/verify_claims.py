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
RAN = []


def chk(name, got, want, tol=0.05):
    ok = (abs(got-want) <= tol) if isinstance(want,(int,float)) else (got==want)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} got={got!r} want={want!r}")
    RAN.append(name)
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
# kept under names nothing later reuses, because the table that publishes this
# reconstruction is read further down, once ERRATA's lines are loaded
_A4 = {"gen_drafts": gd, "gen_tokens": gt, "impl_calls": len(gen),
       "empty": small.count(0), "size1": sum(1 for _x in small if _x == 1),
       "dropped_lines": len(small), "dropped_tokens": sum(small),
       "attempts": len(att), "full": sum(1 for _a, _b in att if _b == _a + 1),
       "partial": sum(1 for _a, _b in att if _b < _a + 1), "restores": len(rs),
       "acc_tokens": at,
       "min_flag": sorted({int(_b) for _, _b in re.findall(
           r"ignoring small draft: (\d+) < (\d+)", t)})[0],
       "counter": sum(_a for _a, _b in att if _b == _a + 1),
       "log": t, "small": list(small),
       "gen_ms": dg, "wall_ms": 1000 * 200 / 63.2,
       "ckpts": len(ck), "ckpt_mib": ck[0],
       "gib_written": sum(ck) / 1024, "gib_read": sum(rs) / 2 ** 30}
_A4["events"] = _A4["impl_calls"] + _A4["empty"]
_A4["fresh"] = _A4["events"] - _A4["dropped_lines"]
_A4["fresh_tokens"] = _A4["gen_tokens"] - _A4["dropped_tokens"]

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
def _cstat(arm):
    """Every column run C's tables publish, from that arm's five repeats.

    `sd` is deliberately NOT the SD of `pooled`. The published column is the
    spread of the five repeats' *request means* - llama.cpp's
    `predicted_per_second` averaged over the ten prompts - which is a different
    estimator from the pooled rate in the column beside it. Checked: pooled per
    repeat reproduces 3 of the 13 published SDs, the request mean reproduces
    13 of 13. The table caption now says so.
    """
    rs = [json.load(open(f)) for f in sorted(glob.glob(_C % arm))]
    if not rs:
        raise SystemExit(f"run C has no arm-runs for {arm!r}")
    n = sum(x["predicted_n"] for r in rs for x in r["rows"])
    ms = sum(x["predicted_ms"] for r in rs for x in r["rows"])
    dn = sum(x["draft_n"] for r in rs for x in r["rows"])
    da = sum(x["draft_n_accepted"] for r in rs for x in r["rows"])
    return {"pooled": 1000 * n / ms,
            "acc": 100 * da / dn if dn else None,
            "draft_per_gen": dn / n,
            "sd": st.stdev(st.mean(x["predicted_per_second"] for x in r["rows"])
                           for r in rs),
            "reps": len(rs)}


def _c_reps(arm, how="mean"):
    """The five per-repeat rates for a run C arm, request mean or pooled."""
    rs = [json.load(open(f)) for f in sorted(glob.glob(_C % arm))]
    if how == "mean":
        return [st.mean(x["predicted_per_second"] for x in r["rows"]) for r in rs]
    return [1000 * sum(x["predicted_n"] for x in r["rows"])
            / sum(x["predicted_ms"] for x in r["rows"]) for r in rs]


_C_ARM = {"baseline": "baseline", "baseline-kvfp16": "baseline-kvfp16",
          "ngram-simple": "ngram-simple", "ngram-mod n=24": "ngram-mod-n24",
          "ngram-cache": "ngram-cache", "ngram-cache-kvfp16": "ngram-cache-kvfp16",
          "draft model n_max 1": "spec-draft-n1", "draft model n_max 2": "spec-draft-n2",
          "draft model n_max 4": "spec-draft-n4", "draft model n_max 8": "spec-draft-n8",
          "draft model n_max 16": "spec-draft-n16",
          "draft model n_max 32": "spec-draft-n32",
          "draft model, v1's config": "spec-draft-v1cfg"}


def _c_lookup(label):
    """Resolve a published row label to a run C arm, or fail loudly.

    Two documents write the same rows differently - `draft model n_max 8` in
    the v4 README, `draft model, n_max 8` in ERRATA - so the comma is dropped
    before the lookup. An unrecognised label is an error, not a skipped row: a
    check that quietly ignores what it cannot resolve reads as a pass.
    """
    key = label.replace("draft model, n_max", "draft model n_max").strip()
    if key in _C_ARM:
        return _C_ARM[key]
    if key in set(_C_ARM.values()):     # ERRATA A18 names the directories
        return key
    raise SystemExit(f"unrecognised run C row label: {label!r}")


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
# `aggregate`, not `agg`: `agg()` is a module-level function the v1 section
# still needs, and a loop target of the same name replaced it with a float for
# the rest of the file. The self-audit below now refuses that shape.
for arm,aggregate,pooled,delta in (("spec-dflash-n4",130.2,151.6, 18.7),
                             ("spec-dflash-n8", 93.5,105.2,-14.8),
                             ("spec-dflash-n16",57.7, 62.8,-47.4),
                             ("spec-draft-n8",  30.5, 31.4,-72.2)):
    a=_arm(J % arm)
    chk(f"J {arm} aggregate", round(a[0],1), aggregate, 0.05)
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

_DASH_EARLY = {"\u2212": "-", "\u2013": "-", "\u2014": "-",
               "\u00a0": " ", "\u2009": " ", "\u202f": " "}


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import table_coverage as _tcovn                                    # noqa: E402


_V1CSV = list(csv.DictReader(
    open(pathlib.Path(__file__).resolve().parent / "summary.csv", encoding="utf-8")))


def _num_rows(lines, header_startswith, label_col=0):
    """{first cell: [every number in the row]} for one markdown table.

    The same extractor the coverage probe perturbs with, so a row that passes
    here has no number left for the probe to change unnoticed. Reading a table
    any other way is how eighty numbers sat in "parsed" tables unread.
    """
    i = next(i for i, l in enumerate(lines)
             if l.strip().lstrip("> ").strip().startswith(header_startswith))
    out = {}
    for l in lines[i + 2:]:
        raw = l.strip().lstrip("> ").strip()
        if not raw.startswith("|"):
            break
        cells = [c.strip() for c in raw.strip("|").split("|")]
        # a footnote marker is not part of a row's identity
        out[cells[label_col].replace("`", "").replace("*", "")
            .replace("\u2020", "").strip()] = [
            x[2] for sp in _tcovn._pipe_spans(raw)
            for x in _tcovn._numbers_in(raw, sp)]
    return out


def _num_rows_seq(lines, header_startswith, nth=0):
    """The same, in document order, for tables whose first cell repeats.

    `nth` selects among tables that share a header. The census marks every
    table matching a header as parsed once that header reaches a reader, so a
    document with two `| Path | Contents |` tables needs both read or the
    coverage it reports is a table it never looked at.
    """
    _hits = [i for i, l in enumerate(lines)
             if l.strip().lstrip("> ").strip().startswith(header_startswith)]
    i = _hits[nth]
    out = []
    for l in lines[i + 2:]:
        raw = l.strip().lstrip("> ").strip()
        if not raw.startswith("|"):
            break
        out.append(([c.strip() for c in raw.strip("|").split("|")],
                    [x[2] for sp in _tcovn._pipe_spans(raw)
                     for x in _tcovn._numbers_in(raw, sp)]))
    return out


def _num_row_check(label, got, want):
    # no tolerance parameter: comparing lists, `chk` tests equality whatever it
    # is given, so a caller passing one would get exact comparison anyway
    chk(f"{label}: every number in the row",
        [float(x) for x in got], [float(x) for x in want])


def _norm_early(t: str) -> str:
    for _a, _b in _DASH_EARLY.items():
        t = t.replace(_a, _b)
    return t


def _cellv4(x):
    """One numeric table cell, before `_pnum2` exists further down the file."""
    t = _norm_early(x).replace("%", "").replace("pp", "").replace("*", "").replace("`", "")
    for _c in (" ", "\u00a0", "\u2009", "\u202f"):
        t = t.replace(_c, "")
    return float(t)


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

# ...and the table that publishes them, cell by cell. Computing a figure from
# the data and comparing it with a literal proves the literal, not the document:
# `tests/data_mutate.py` changed run M's published aggregate from 127.3 to 130.3
# and nothing here noticed, because nothing here read that cell.
_RM_LINES = (pathlib.Path(__file__).resolve().parents[1]
             / "README.md").read_text(encoding="utf-8").splitlines()
_RT_LINES = (pathlib.Path(__file__).resolve().parents[1]
             / "RETEST_TODO.md").read_text(encoding="utf-8").splitlines()


_V4R_TEXT = (pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
             / "README.md").read_text(encoding="utf-8")
_V4R_LINES = (pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
              / "README.md").read_text(encoding="utf-8").splitlines()


def _v4_table(header_startswith):
    i = next(i for i, l in enumerate(_V4R_LINES) if l.startswith(header_startswith))
    rows = []
    for l in _V4R_LINES[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip().strip("*`").replace("`", "").strip("* ").strip()
                     for c in _norm_early(l).strip("|").split("|")])
    return rows


# Answer 2's ten rows: run B, per prompt, matched arm, mean of 3 repeats. The
# Pearson r computed from exactly this data is asserted above, but the ten rows
# it is computed from were not, so a wrong cell in any of them was publishable
# and the correlation would still have checked out. Found by perturbing every
# published table in turn - `analysis/table_coverage.py --probe`.
_A2 = _v4_table("| prompt | baseline | with draft | vs baseline | real acceptance |")
chk("v4 README Answer 2: ten prompts", len(_A2), 10)
_a2b, _a2d, _a2a = defaultdict(list), defaultdict(list), {}
for _r in B["baseline"]:
    for _x in _r["rows"]:
        _a2b[_x["tag"]].append(_x["predicted_per_second"])
for _r in B["draft-max8-matched"]:
    for _x in _r["rows"]:
        _a2d[_x["tag"]].append(_x["predicted_per_second"])
        if _x["draft_n"]:
            _a2a[_x["tag"]] = (_x["draft_n_accepted"], _x["draft_n"])
chk("v4 README Answer 2: every row is a prompt run B measured",
    sorted(r[0] for r in _A2), sorted(_a2b))
for _row in _A2:
    _t = _row[0]
    _bm, _dm = st.mean(_a2b[_t]), st.mean(_a2d[_t])
    chk(f"v4 README Answer 2 {_t} baseline", round(_bm, 1), _cellv4(_row[1]), 0.06)
    chk(f"v4 README Answer 2 {_t} with draft", round(_dm, 1), _cellv4(_row[2]), 0.06)
    chk(f"v4 README Answer 2 {_t} vs baseline (%)",
        round(100 * (_dm / _bm - 1), 1), _cellv4(_row[3]), 0.06)
    chk(f"v4 README Answer 2 {_t} real acceptance (%)",
        round(100 * _a2a[_t][0] / _a2a[_t][1]), _cellv4(_row[4]), 0.5)

# Run C's thirteen-arm table, the largest in the repository at 65 cells. Every
# number in it was typed from a report and nothing read it back, so any one of
# them could have been wrong. `analysis/table_coverage.py --probe` is what
# found that; this is the fix.
_CT = _v4_table("| arm | pooled tok/s | vs baseline | acceptance | "
                "draft tokens per generated token | run-to-run SD |")
chk("v4 README run C table: thirteen arms", len(_CT), 13)
chk("v4 README run C table: every row resolves to a measured arm",
    sorted(_c_lookup(r[0]) for r in _CT),
    sorted({f.split("/")[-1].split("__rep")[0]
            for f in glob.glob(_C % "*")}))
_c_base = _cstat("baseline")["pooled"]
for _row in _CT:
    _a = _c_lookup(_row[0])
    _s = _cstat(_a)
    chk(f"v4 README C {_a}: five repeats", _s["reps"], 5)
    chk(f"v4 README C {_a} pooled", round(_s["pooled"], 1), _cellv4(_row[1]), 0.06)
    if any(c.isdigit() for c in _row[2]):
        chk(f"v4 README C {_a} vs baseline (%)",
            round(100 * (_s["pooled"] / _c_base - 1), 1), _cellv4(_row[2]), 0.06)
    else:
        chk(f"v4 README C {_a} is a no-speculation row, so no vs-baseline cell",
            _s["draft_per_gen"], 0.0)
    if any(c.isdigit() for c in _row[3]):
        chk(f"v4 README C {_a} acceptance (%)",
            round(_s["acc"], 1), _cellv4(_row[3]), 0.06)
    else:
        chk(f"v4 README C {_a} drafts nothing, so no acceptance cell",
            _s["acc"], None)
    chk(f"v4 README C {_a} draft tokens per generated token",
        round(_s["draft_per_gen"], 2), _cellv4(_row[4]), 0.006)
    chk(f"v4 README C {_a} run-to-run SD", round(_s["sd"], 2), _cellv4(_row[5]), 0.006)
chk("v4 README C table: the fastest arm is the fp16-KV control",
    max(_CT, key=lambda r: _cellv4(r[1]))[0], "baseline-kvfp16")
chk("v4 README C table: the slowest is the longest draft",
    min(_CT, key=lambda r: _cellv4(r[1]))[0], "draft model n_max 32")

# Run O's head-to-head, nine arms and six numeric columns. It is the table the
# "purpose-built draft paths win" claim rests on, and until now the only thing
# read from it was the two acceptance figures in its footnote.
_OT = _v4_table("| arm | pooled tok/s | Δ pooled | aggregate tok/s | Δ aggregate |")
_O_DIR = "matrix_O_headtohead_20260826_081806"
chk("v4 README run O table: nine arms", len(_OT), 9)


def _o_arm(label):
    """`spec-dflash-n2 - self-speculative` names an arm; `no speculation` is the baseline."""
    head = _norm_early(label).replace("*", "").split(" - ")[0].strip()
    return "baseline" if head == "no speculation" else head


chk("v4 README run O table: every row is an arm the run measured",
    sorted(_o_arm(r[0]) for r in _OT),
    sorted({f.split("/")[-1].split("__rep")[0]
            for f in glob.glob(f"v4_audit_2026_08_25/data/{_O_DIR}/*__rep*.json")}))
_o_base = _agg(_O_DIR, "baseline")
for _row in _OT:
    _a = _o_arm(_row[0])
    _v = _agg(_O_DIR, _a)
    chk(f"v4 README O {_a}: three repeats", _v["reps"], 3)
    chk(f"v4 README O {_a} pooled", round(_v["pooled"], 1), _cellv4(_row[1]), 0.06)
    chk(f"v4 README O {_a} aggregate", round(_v["agg"], 1), _cellv4(_row[3]), 0.06)
    chk(f"v4 README O {_a} draft tokens", _v["drafted"], int(_cellv4(_row[6])))
    if _a == "baseline":
        chk(f"v4 README O {_a} is the reference, so no delta and no acceptance",
            (any(c.isdigit() for c in _row[2] + _row[4] + _row[5]),
             _v["acc"], _v["drafted"]),
            (False, None, 0))
        continue
    chk(f"v4 README O {_a} Δ pooled (%)",
        round(100 * (_v["pooled"] / _o_base["pooled"] - 1), 1), _cellv4(_row[2]), 0.06)
    chk(f"v4 README O {_a} Δ aggregate (%)",
        round(100 * (_v["agg"] / _o_base["agg"] - 1), 1), _cellv4(_row[4]), 0.06)
    chk(f"v4 README O {_a} acceptance (%)", round(_v["acc"], 1), _cellv4(_row[5]), 0.06)
chk("v4 README O: the table is sorted by pooled rate, descending",
    [_cellv4(r[1]) for r in _OT],
    sorted((_cellv4(r[1]) for r in _OT), reverse=True))
chk("v4 README O: the fastest and slowest arms, a factor of five apart",
    (round(max(_cellv4(r[1]) for r in _OT) / min(_cellv4(r[1]) for r in _OT), 1) >= 5.0,
     _o_arm(_OT[0][0]), _o_arm(_OT[-1][0])),
    (True, "spec-dflash-n2", "spec-draft-n1"))

# The per-prompt DFlash table, and the sentence it exists to support: "it wins
# on all ten prompts individually". Forty cells, none of them read until now.
_J2 = "matrix_J2_20260826_014750"


def _j2_prompt(arm):
    """Per prompt, the mean of that prompt's `predicted_per_second` over the repeats.

    Request mean, not pooled - checked against the published cells, which the
    pooled rate misses by 0.4-0.7 tok/s on every row. Every rate in this table
    is therefore low by `(n-1)/n`; see ERRATA B8. The comparison stays valid
    because all four arms are the same ten prompts under one cap.
    """
    rows = defaultdict(list)
    for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{_J2}/{arm}__rep*.json")):
        for _x in json.load(open(_f))["rows"]:
            rows[_x["tag"]].append(_x["predicted_per_second"])
    return {_t: st.mean(_v) for _t, _v in rows.items()}


_JPT = _v4_table("| prompt | no speculation | `dflash-n4` | `dflash-n8` | `dflash-n16` |")
_JP_ARMS = ["baseline", "spec-dflash-n4", "spec-dflash-n8", "spec-dflash-n16"]
_JP = {_a: _j2_prompt(_a) for _a in _JP_ARMS}
chk("v4 README J per-prompt table: ten prompts", len(_JPT), 10)
chk("v4 README J per-prompt table: the prompts run J measured",
    sorted(r[0] for r in _JPT), sorted(_JP["baseline"]))
for _row in _JPT:
    _t = _row[0]
    for _k, _a in enumerate(_JP_ARMS, start=1):
        chk(f"v4 README J {_a} on {_t}",
            round(_JP[_a][_t], 1), _cellv4(_row[_k]), 0.06)
chk("v4 README J: dflash-n4 beats no speculation on all ten prompts individually",
    sum(1 for _t in _JP["baseline"]
        if _JP["spec-dflash-n4"][_t] > _JP["baseline"][_t]), 10)
# n8 loses by 14.8 % on aggregate and still wins on three prompts. The table
# has said so since it was published; no sentence next to it did until the
# assertion above was written and this one failed.
_JP_W8 = sorted(_t for _t in _JP["baseline"]
                if _JP["spec-dflash-n8"][_t] > _JP["baseline"][_t])
chk("v4 README J: n8 wins on three prompts despite losing on aggregate",
    _JP_W8, ["code_small", "medium_rec", "reasoning"])
chk("v4 README J: n16 wins on none",
    [_t for _t in _JP["baseline"]
     if _JP["spec-dflash-n16"][_t] > _JP["baseline"][_t]], [])
_JP_A8 = defaultdict(lambda: [0, 0])
for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{_J2}/spec-dflash-n8__rep*.json")):
    for _x in json.load(open(_f))["rows"]:
        _JP_A8[_x["tag"]][0] += _x["draft_n_accepted"]
        _JP_A8[_x["tag"]][1] += _x["draft_n"]
_JP_ACC = {_t: 100 * _a / _d for _t, (_a, _d) in _JP_A8.items() if _d}
chk("v4 README J: acceptance separates n8's winners from its losers with no overlap",
    round(min(_JP_ACC[_t] for _t in _JP_W8)
          - max(_JP_ACC[_t] for _t in set(_JP_ACC) - set(_JP_W8)), 1), 9.5, 0.06)
for _t, _w in (("code_small", 71.8), ("reasoning", 48.6), ("medium_rec", 47.3)):
    chk(f"v4 README J: n8 acceptance on {_t}", round(_JP_ACC[_t], 1), _w, 0.06)
chk("v4 README J: the seven losers' acceptance range",
    (round(min(_JP_ACC[_t] for _t in set(_JP_ACC) - set(_JP_W8)), 1),
     round(max(_JP_ACC[_t] for _t in set(_JP_ACC) - set(_JP_W8)), 1)), (27.1, 37.8))

# Run K's draft-length sweep, and the two shape claims that rest on it: a
# plateau rather than a peak, and a sign change between 4 and 6.
_K1 = "matrix_K1_sweep_20260826_025615"


def _k1(arm):
    _rs = [json.load(open(_f))
           for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{_K1}/{arm}__rep*.json"))]
    _n = sum(x["predicted_n"] for r in _rs for x in r["rows"])
    _ms = sum(x["predicted_ms"] for r in _rs for x in r["rows"])
    _dn = sum(x["draft_n"] for r in _rs for x in r["rows"])
    _da = sum(x["draft_n_accepted"] for r in _rs for x in r["rows"])
    _ag = [r["aggregate_tok_s"] for r in _rs]
    return {"agg": st.mean(_ag), "sd": st.stdev(_ag), "pooled": 1000 * _n / _ms,
            "acc": 100 * _da / _dn if _dn else None, "reps": len(_rs)}


_KT = _v4_table("| `n_max` | aggregate | run-to-run SD | pooled | vs no speculation |")
chk("v4 README run K sweep: seven rows", len(_KT), 7)
_k_base = _k1("baseline")
_k_seen = []
for _row in _KT:
    _lab = _norm_early(_row[0]).replace("*", "").strip()
    _a = "baseline" if "no speculation" in _lab else f"spec-dflash-n{_lab}"
    _k_seen.append(_a)
    _v = _k1(_a)
    chk(f"v4 README K {_a}: three repeats", _v["reps"], 3)
    chk(f"v4 README K {_a} aggregate", round(_v["agg"], 1), _cellv4(_row[1]), 0.06)
    chk(f"v4 README K {_a} run-to-run SD", round(_v["sd"], 2), _cellv4(_row[2]), 0.006)
    chk(f"v4 README K {_a} pooled", round(_v["pooled"], 1), _cellv4(_row[3]), 0.06)
    if _a == "baseline":
        chk(f"v4 README K {_a} is the reference, so no delta and no acceptance",
            (any(c.isdigit() for c in _row[4] + _row[5]), _v["acc"]), (False, None))
        continue
    chk(f"v4 README K {_a} vs no speculation (%)",
        round(100 * (_v["agg"] / _k_base["agg"] - 1), 1), _cellv4(_row[4]), 0.06)
    chk(f"v4 README K {_a} acceptance (%)", round(_v["acc"], 1), _cellv4(_row[5]), 0.06)
chk("v4 README K sweep: every DFlash arm the run holds is in the table",
    sorted(_k_seen),
    sorted({_f.split("/")[-1].split("__rep")[0]
            for _f in glob.glob(f"v4_audit_2026_08_25/data/{_K1}/*__rep*.json")}))
# "a plateau and then a cliff": 2, 3 and 4 within the baseline's own run-to-run SD
_k_plateau = [_k1(f"spec-dflash-n{_n}")["agg"] for _n in (2, 3, 4)]
chk("v4 README K: n_max 2, 3 and 4 sit inside the baseline's run-to-run SD",
    max(_k_plateau) - min(_k_plateau) < _k_base["sd"], True)
chk("v4 README K: acceptance falls monotonically with draft length",
    [round(_k1(f"spec-dflash-n{_n}")["acc"], 1) for _n in (1, 2, 3, 4, 6, 8)],
    sorted((round(_k1(f"spec-dflash-n{_n}")["acc"], 1) for _n in (1, 2, 3, 4, 6, 8)),
           reverse=True))
chk("v4 README K: the sign change is between 4 and 6, not between 4 and 8",
    (_k1("spec-dflash-n4")["agg"] > _k_base["agg"],
     _k1("spec-dflash-n6")["agg"] < _k_base["agg"]), (True, True))

# Run N, and the paragraph beside it that mixed two spans: 3271 generate()
# calls is per repeat, 144 draft tokens is over all thirty requests.
_N = "matrix_N_ngrammap_20260826_081806"
_NCMP = [c for c in json.load(open(
    "v4_audit_2026_08_25/data/acceptance_counter_comparison.json"))
    if c["run"] == _N]
_NT = _v4_table("| arm | aggregate | vs baseline | draft tokens over 30 requests |")
chk("v4 README run N table: seven rows", len(_NT), 7)


def _n_arm(label):
    _t = _norm_early(label).replace("*", "").replace("`", "")
    _t = _t.split("(")[0].strip()
    return "baseline" if _t == "no speculation" else _t


chk("v4 README run N: every row is an arm the run measured",
    sorted(_n_arm(r[0]) for r in _NT),
    sorted({f.split("/")[-1].split("__rep")[0]
            for f in glob.glob(f"v4_audit_2026_08_25/data/{_N}/*__rep*.json")}))
_n_base = _agg(_N, "baseline")
for _row in _NT:
    _a = _n_arm(_row[0])
    _v = _agg(_N, _a)
    chk(f"v4 README N {_a}: three repeats", _v["reps"], 3)
    chk(f"v4 README N {_a} aggregate", round(_v["agg"], 1), _cellv4(_row[1]), 0.06)
    chk(f"v4 README N {_a} draft tokens over 30 requests",
        _v["drafted"], int(_cellv4(_row[3])))
    if _a == "baseline":
        chk(f"v4 README N {_a} is the reference and drafts nothing",
            (any(c.isdigit() for c in _row[2]), _v["drafted"]), (False, 0))
        continue
    chk(f"v4 README N {_a} vs baseline (%)",
        round(100 * (_v["agg"] / _n_base["agg"] - 1), 1), _cellv4(_row[2]), 0.06)
    _sv, _dr = (_cellv4(x) for x in _row[4].split("/"))
    _rows = [c for c in _NCMP if c["arm"] == _a]
    chk(f"v4 README N {_a}: three counter rows, one per repeat", len(_rows), 3)
    chk(f"v4 README N {_a} server-counter acceptance (%)",
        round(_v["acc"], 1), _sv, 0.06)
    chk(f"v4 README N {_a} drafter-counter acceptance (%)",
        sorted({r["drafter_pct"] for r in _rows}), [_dr])
# the spans the paragraph now names
_N_K = [c for c in _NCMP if c["arm"] == "ngram-map-k"]
chk("v4 README N: generate() calls, per repeat",
    sorted({c["drafter_calls_generate"] for c in _N_K}), [3271])
chk("v4 README N: and across all thirty requests",
    sum(c["drafter_calls_generate"] for c in _N_K), 9813)
chk("v4 README N: drafts returned, per repeat and over thirty",
    (sorted({c["drafter_drafts"] for c in _N_K})[0],
     sum(c["drafter_drafts"] for c in _N_K)), (2, 6))
chk("v4 README N: the three repeats are identical to the token",
    len({(c["server_drafted"], c["server_accepted"], c["drafter_drafted"],
          c["drafter_accepted"], c["drafter_calls_generate"]) for c in _N_K}), 1)
# 144/48, 24/8, 12/4 = three, i.e. one full-length hit per repeat
for _a, _m in (("ngram-map-k", 48), ("ngram-map-k-m8", 8), ("ngram-map-k-m4", 4)):
    chk(f"v4 README N {_a}: server-counted draft tokens over thirty requests are 3 x size_m",
        _agg(_N, _a)["drafted"], 3 * _m)
chk("v4 README N: the drafter counts more tokens than the server, in every arm",
    sorted({(c["arm"], c["drafter_drafted"] > c["server_drafted"]) for c in _NCMP
            if c["drafter_drafted"]})
    == sorted({(c["arm"], True) for c in _NCMP if c["drafter_drafted"]}), True)
chk("v4 README N: the drafter's per-repeat token counts, as the paragraph prints them",
    [sorted({c["drafter_drafted"] for c in _NCMP if c["arm"] == _a})[0]
     for _a in ("ngram-map-k", "ngram-map-k-m8", "ngram-map-k-m4")], [55, 15, 10])
chk("v4 README N: and the drafts they came in",
    [sorted({c["drafter_drafts"] for c in _NCMP if c["arm"] == _a})[0]
     for _a in ("ngram-map-k", "ngram-map-k-m8", "ngram-map-k-m4")], [2, 2, 3])

# The v1 representative table, 68 cells and the largest in the repository. Four
# of its twelve rows were checked against literals; the other eight were not,
# and the two range rows never had been. Table driven now, so a row added to
# the document is a row that has to reconcile with `analysis/summary.csv`.
_ROOT_TEXT = (pathlib.Path(__file__).resolve().parents[1]
              / "README.md").read_text(encoding="utf-8")
_ROOT_LINES = _ROOT_TEXT.splitlines()


def _root_table(header_startswith):
    _i = next(_i for _i, _l in enumerate(_ROOT_LINES)
              if _l.startswith(header_startswith))
    _rows = []
    for _l in _ROOT_LINES[_i + 2:]:
        if not _l.startswith("|"):
            break
        _rows.append([_c.strip() for _c in _norm_early(_l).strip("|").split("|")])
    return _rows


_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _v1_cell(x):
    """`135.5 (-0.1 %)` -> ([135.5], [-0.1]); a range row gives two of each.

    A dash between two digits is a range separator, not a sign - `7-9 / 10` is
    seven to nine, and `129.6 - 130.1` is a span. A dash that opens a number is
    a sign. Getting that backwards turns every range row into a negative.
    """
    _t = _norm_early(x).replace("*", "").replace("`", "")
    _t = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " to ", _t)
    _head, _, _tail = _t.partition("(")
    return ([float(v) for v in _NUM_RE.findall(_head)],
            [float(v) for v in _NUM_RE.findall(_tail.replace("%", ""))])


def _v1_label(x):
    """The configs a row stands for. `ngmod-n20 / n16 / n8 / n12` is four."""
    _t = _norm_early(x).replace("*", "").replace("`", "").split("(")[0].strip()
    _parts = [_p.strip() for _p in _t.split("/")]
    if len(_parts) == 1:
        return _parts
    _stem = _parts[0].rsplit("-", 1)[0]
    return [_parts[0]] + [f"{_stem}-{_p}" for _p in _parts[1:]]


def _span(vals):
    return sorted({round(v, 1) for v in vals})


def _v1agg(c):
    """`agg()` and `b` from the v1 section are both rebound by later loops -
    `agg` becomes a float at the run J table, `b` becomes a baseline dict.
    This block runs after both, so it computes its own."""
    _v = by[c]
    _r = [x["tok_s"] for x in _v]
    return {"mean": st.mean(_r),
            "pooled": 1000 * sum(x["predicted_n"] for x in _v)
                      / sum(x["predicted_ms"] for x in _v),
            "med": st.median(_r), "mn": min(_r),
            "act": sum(1 for x in _v if x["draft_n"] > 0)}


_v1_base = _v1agg("baseline")
_V1T = _root_table("| condition | request-mean | pooled | median | min |")
# The one number in that table's label column: the 0.6B drafter's vocabulary,
# which is why the draft never attached. It is not derivable from anything
# committed here - the 0.6B GGUF is not in the archive - so what can be checked
# is that the three documents quoting it quote the same number, and that it
# really differs from the target vocabulary they compare it against. Perturbing
# it produced no failure on 2026-08-29 because nothing read the cell at all.
_VOCAB_DRAFT, _VOCAB_TARGET = 151936, 248320
_ER_EARLY = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("ERRATA.md").read_text(encoding="utf-8")
_PRC_EARLY = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("pr_comment.md").read_text(encoding="utf-8")
chk("README labels the 0.6B drafter with its vocabulary",
    f"vocab {_VOCAB_DRAFT}, draft never attached" in _ROOT_TEXT, True)
chk("ERRATA quotes the same vocabulary, against the target's",
    f"vocab {_VOCAB_DRAFT} \u2260 {_VOCAB_TARGET}" in _ER_EARLY, True)
chk("and the PR comment that started it quotes the drafter's too",
    f"(vocab {_VOCAB_DRAFT})" in _PRC_EARLY, True)
chk("the target vocabulary is the one the README declares",
    f"declare `vocab_size = {_VOCAB_TARGET}`" in _ROOT_TEXT, True)
chk("and the two really differ, which is the point of the row",
    _VOCAB_DRAFT != _VOCAB_TARGET, True)
# The transposed n_max table: six draft lengths as columns, and the "six
# distinct draft lengths each reproduce this at r >= +0.996" sentence beside it.
# Its Pearson r row was never computed from anything - only the single +0.998
# from run B was, which is a different run and a different arm.
_NMAXH = next(_l for _l in _ROOT_LINES if _l.startswith("| `--spec-draft-n-max` |"))
_NMAX = [_c.strip().strip("`") for _c in _NMAXH.strip("|").split("|")][1:]
_NMT = {_r[0]: _r[1:] for _r in _root_table("| `--spec-draft-n-max` |")}
chk("README n_max table: six draft lengths", len(_NMAX), 6)
chk("README n_max table: its three rows", sorted(_NMT),
    ["Pearson r", "acceptance", "pooled tok/s"])


def _c_prompt_r(arm):
    """Within one arm, across the ten prompts: acceptance against decode rate."""
    _pn, _pms, _da, _dn = (defaultdict(int) for _ in range(4))
    for _f in sorted(glob.glob(_C % arm)):
        for _x in json.load(open(_f))["rows"]:
            _pn[_x["tag"]] += _x["predicted_n"]
            _pms[_x["tag"]] += _x["predicted_ms"]
            _da[_x["tag"]] += _x["draft_n_accepted"]
            _dn[_x["tag"]] += _x["draft_n"]
    _t = sorted(_pn)
    _xs = [100 * _da[_k] / _dn[_k] for _k in _t]
    _ys = [1000 * _pn[_k] / _pms[_k] for _k in _t]
    _mx, _my = st.mean(_xs), st.mean(_ys)
    return (sum((_a - _mx) * (_b - _my) for _a, _b in zip(_xs, _ys))
            / ((sum((_a - _mx) ** 2 for _a in _xs)
                * sum((_b - _my) ** 2 for _b in _ys)) ** 0.5))


for _k, _n in enumerate(_NMAX):
    _a = f"spec-draft-n{_n}"
    _v = _cstat(_a)
    chk(f"README n_max {_n} Pearson r across the ten prompts",
        round(_c_prompt_r(_a), 3), _cellv4(_NMT["Pearson r"][_k]), 0.0006)
    chk(f"README n_max {_n} pooled",
        round(_v["pooled"], 1), _cellv4(_NMT["pooled tok/s"][_k]), 0.06)
    chk(f"README n_max {_n} acceptance (%)",
        round(_v["acc"], 1), _cellv4(_NMT["acceptance"][_k]), 0.06)
chk("README n_max: all six reproduce it at r >= +0.996",
    min(round(_c_prompt_r(f"spec-draft-n{_n}"), 3) for _n in _NMAX) >= 0.996, True)
# "spanning acceptance from 5 % to 83 %" is the PER PROMPT span across the six
# arms, not the span of the six pooled figures in the row above, which is
# 8 % to 69 %. Checking the wrong one of those makes the document look wrong.
_nm_pp = []
for _n in _NMAX:
    _da, _dn = defaultdict(int), defaultdict(int)
    for _f in sorted(glob.glob(_C % f"spec-draft-n{_n}")):
        for _x in json.load(open(_f))["rows"]:
            _da[_x["tag"]] += _x["draft_n_accepted"]
            _dn[_x["tag"]] += _x["draft_n"]
    _nm_pp += [100 * _da[_t] / _dn[_t] for _t in _dn if _dn[_t]]
chk("README n_max: per prompt they span acceptance from 5 % to 83 %",
    (round(min(_nm_pp)), round(max(_nm_pp))), (5, 83))
chk("README n_max: the pooled row spans a narrower 8 % to 69 %",
    (round(min(_cstat(f"spec-draft-n{_n}")["acc"] for _n in _NMAX)),
     round(max(_cstat(f"spec-draft-n{_n}")["acc"] for _n in _NMAX))), (8, 69))

chk("README v1 table: twelve rows", len(_V1T), 12)
_v1_seen = []
for _row in _V1T:
    _cfgs = _v1_label(_row[0])
    _v1_seen += _cfgs
    _a = [_v1agg(_c) for _c in _cfgs]
    _name = "/".join(_cfgs)
    chk(f"README v1 {_name}: every config it names is in summary.csv",
        [_c in by for _c in _cfgs], [True] * len(_cfgs))
    for _col, _key in ((1, "mean"), (2, "pooled"), (3, "med"), (4, "mn")):
        _got, _delta = _v1_cell(_row[_col])
        # unrounded against the published cell, tolerance 0.06: 130.05 is a
        # rounding boundary and the document rounds half away from zero where
        # Python rounds half to even. Rounding first would fail on the tie.
        _vals = [x[_key] for x in _a]
        chk(f"README v1 {_name} {_key} low", min(_vals), min(_got), 0.06)
        chk(f"README v1 {_name} {_key} high", max(_vals), max(_got), 0.06)
        if _delta:
            _d = [100 * (x[_key] / _v1_base[_key] - 1) for x in _a]
            chk(f"README v1 {_name} {_key} vs baseline, low (%)",
                min(_d), min(_delta), 0.06)
            chk(f"README v1 {_name} {_key} vs baseline, high (%)",
                max(_d), max(_delta), 0.06)
    _cell5 = _v1_cell(_row[5])[0]
    _hits, _of = _cell5[:-1], _cell5[-1]
    _act = [x["act"] for x in _a]
    chk(f"README v1 {_name} requests with a counted draft round",
        (min(_act), max(_act), len(by[_cfgs[0]])),
        (int(min(_hits)), int(max(_hits)), int(_of)))
# the long-output rows, which use their own reference
_V1L = _root_table("| condition | request-mean | pooled | note |")
_v1_l_base = _v1agg("baseline-1000tok")
chk("README v1 long-output table: three rows", len(_V1L), 3)
chk("README v1 long-output: its reference, as the sentence above it prints",
    (round(_v1_l_base["mean"], 1), round(_v1_l_base["pooled"], 1)), (133.2, 133.1))
for _row in _V1L:
    _c = _v1_label(_row[0])[0]
    chk(f"README v1 long-output {_c}: a config summary.csv holds", _c in by, True)
    _v = _v1agg(_c)
    for _col, _key in ((1, "mean"), (2, "pooled")):
        _got, _delta = _v1_cell(_row[_col])
        chk(f"README v1 long-output {_c} {_key}", _v[_key], _got[0], 0.06)
        chk(f"README v1 long-output {_c} {_key} vs baseline-1000tok (%)",
            100 * (_v[_key] / _v1_l_base[_key] - 1), _delta[0], 0.06)
chk("README v1 long-output: it covers every 1000-token variant",
    sorted(_v1_label(_r[0])[0] for _r in _V1L),
    sorted(_c for _c in by if _c.endswith("-1000tok") and _c != "baseline-1000tok"))
chk("README v1 long-output: and ngcache-1000tok is the worst pooled in the matrix",
    min(by, key=lambda _c: _v1agg(_c)["pooled"]), "ngcache-1000tok")

chk("README v1 table: it covers every 300-token config and no other",
    sorted(_v1_seen),
    sorted(_c for _c in by if not _c.endswith("-1000tok")))

# Run L: the same five arms with thinking on and off, and the per-prompt table
# under it that carries the mechanism claim. Twenty of the latter's cells are
# two numbers in one cell, which is why neither was ever read.
_L_ON = "matrix_L_thinkon_20260826_032652"
_L_OFF = "matrix_L_thinkoff_20260826_032652"


def _l_pool(half, arm):
    _rs = [json.load(open(_f))
           for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{half}/{arm}__rep*.json"))]
    _n = sum(x["predicted_n"] for r in _rs for x in r["rows"])
    _ms = sum(x["predicted_ms"] for r in _rs for x in r["rows"])
    _dn = sum(x["draft_n"] for r in _rs for x in r["rows"])
    _da = sum(x["draft_n_accepted"] for r in _rs for x in r["rows"])
    return {"pooled": 1000 * _n / _ms, "acc": 100 * _da / _dn if _dn else None,
            "reps": len(_rs)}


def _l_prompt(half, arm):
    """Per prompt, pooled. The document's integer deltas do not separate this
    from the request mean - both reproduce all twenty cells - so the pooled
    rate is used because it is what the rest of the repository publishes."""
    _n, _ms, _da, _dn = (defaultdict(int) for _ in range(4))
    for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{half}/{arm}__rep*.json")):
        for _x in json.load(open(_f))["rows"]:
            _n[_x["tag"]] += _x["predicted_n"]
            _ms[_x["tag"]] += _x["predicted_ms"]
            _da[_x["tag"]] += _x["draft_n_accepted"]
            _dn[_x["tag"]] += _x["draft_n"]
    return ({_t: 1000 * _n[_t] / _ms[_t] for _t in _n},
            {_t: 100 * _da[_t] / _dn[_t] for _t in _dn if _dn[_t]})


_LT = [_r for _r in _v4_table("| arm | thinking ON | | thinking OFF | |")
       if _r[0]]
chk("v4 README run L table: five arms", len(_LT), 5)
for _row in _LT:
    _a = "baseline" if _row[0] == "no speculation" else _row[0]
    for _half, _col in ((_L_ON, 1), (_L_OFF, 3)):
        # not `_b`: that name is the M1 baseline further down this file
        _v, _lbase = _l_pool(_half, _a), _l_pool(_half, "baseline")
        _tag = "ON" if _col == 1 else "OFF"
        chk(f"v4 README L {_a} {_tag}: five repeats", _v["reps"], 5)
        chk(f"v4 README L {_a} {_tag} pooled",
            round(_v["pooled"], 1), _cellv4(_row[_col]), 0.06)
        if _a == "baseline":
            chk(f"v4 README L {_a} {_tag} is the reference",
                any(c.isdigit() for c in _row[_col + 1]), False)
        else:
            chk(f"v4 README L {_a} {_tag} vs base (%)",
                round(100 * (_v["pooled"] / _lbase["pooled"] - 1), 1),
                _cellv4(_row[_col + 1]), 0.06)
# the sentence under it
chk("v4 README L: the win survives and shrinks by two thirds at n_max 2",
    (round(100 * (_l_pool(_L_ON, "spec-dflash-n2")["pooled"]
                  / _l_pool(_L_ON, "baseline")["pooled"] - 1), 1),
     round(100 * (_l_pool(_L_OFF, "spec-dflash-n2")["pooled"]
                  / _l_pool(_L_OFF, "baseline")["pooled"] - 1), 1)), (21.1, 7.6))
chk("v4 README L: and goes negative at n_max 4",
    round(100 * (_l_pool(_L_OFF, "spec-dflash-n4")["pooled"]
                 / _l_pool(_L_OFF, "baseline")["pooled"] - 1), 1), -2.7, 0.06)
for _a, _on, _off in (("spec-dflash-n2", 72.8, 58.5), ("spec-dflash-n4", 55.6, 40.3)):
    chk(f"v4 README L {_a} acceptance, on then off",
        (round(_l_pool(_L_ON, _a)["acc"], 1), round(_l_pool(_L_OFF, _a)["acc"], 1)),
        (_on, _off))

# the per-prompt table: acceptance and delta, both halves, in one cell each
_LP = _v4_table("| prompt | ON: acc / Δ | OFF: acc / Δ |")
chk("v4 README L per-prompt table: ten prompts", len(_LP), 10)
_lp = {_h: _l_prompt(_h, "spec-dflash-n2") for _h in (_L_ON, _L_OFF)}
_lb = {_h: _l_prompt(_h, "baseline")[0] for _h in (_L_ON, _L_OFF)}
for _row in _LP:
    _t = _row[0]
    for _half, _col, _tag in ((_L_ON, 1, "ON"), (_L_OFF, 2, "OFF")):
        _acc_c, _d_c = (_cellv4(_x) for _x in _norm_early(_row[_col]).split("/"))
        chk(f"v4 README L {_t} {_tag} acceptance (%)",
            _lp[_half][1][_t], _acc_c, 0.06)
        chk(f"v4 README L {_t} {_tag} delta (%)",
            100 * (_lp[_half][0][_t] / _lb[_half][_t] - 1), _d_c, 0.6)
chk("v4 README L: ten of ten prompts win with thinking on, seven of ten with it off",
    (sum(1 for _t in _lb[_L_ON] if _lp[_L_ON][0][_t] > _lb[_L_ON][_t]),
     sum(1 for _t in _lb[_L_OFF] if _lp[_L_OFF][0][_t] > _lb[_L_OFF][_t])), (10, 7))
chk("v4 README L: the prompt that loses most is Traditional Chinese free prose",
    min(_lb[_L_OFF], key=lambda _t: _lp[_L_OFF][0][_t] / _lb[_L_OFF][_t]), "zh_hant")
chk("v4 README L: and its acceptance falls from 66 % to 29 %",
    (round(_lp[_L_ON][1]["zh_hant"]), round(_lp[_L_OFF][1]["zh_hant"])), (66, 29))

# "Every measurement of the same quantity, with its power" - six rows drawn
# from three different runs, which is why nothing read it: each row needs its
# own directory. The paragraph under it is the repository's own statement about
# which number to believe, so its two spans are derived too.
_PW_RUN = {"J": _J2, "K1": _K1, "L, thinking on": _L_ON}
_PWT = _v4_table("| run | arm | repeats | Δ aggregate | Δ pooled | configuration |")
chk("v4 README power table: six rows", len(_PWT), 6)
_pw_agg, _pw_pool = [], []
for _row in _PWT:
    _dir = _PW_RUN[_row[0]]
    _a = "spec-dflash-n" + _row[1].split()[-1]
    _v, _bv = _agg(_dir, _a), _agg(_dir, "baseline")
    _name = f"{_row[0]} {_a}"
    chk(f"v4 README power {_name}: repeats", _v["reps"], int(_cellv4(_row[2])))
    _da = 100 * (_v["agg"] / _bv["agg"] - 1)
    _dp = 100 * (_v["pooled"] / _bv["pooled"] - 1)
    chk(f"v4 README power {_name} Δ aggregate (%)", round(_da, 1), _cellv4(_row[3]), 0.06)
    chk(f"v4 README power {_name} Δ pooled (%)", round(_dp, 1), _cellv4(_row[4]), 0.06)
    _pw_agg.append(_da)
    _pw_pool.append(_dp)
# the configuration column was unread, so a row could have claimed any context
# or fitter margin; each is checked against that run's manifest
# from `common_args`, not the `ctx` field: run J2's manifest leaves `ctx` null
# and records the flag in the argument list, so reading the field would have
# called a correct row wrong
_PW_MAN = {}
for _tag, _dir in _PW_RUN.items():
    _m = json.load(open(f"v4_audit_2026_08_25/data/{_dir}/manifest.json"))
    _ca = _m.get("common_args") or []
    _ctx = _ca[_ca.index("-c") + 1] if "-c" in _ca else str(_m.get("ctx"))
    _PW_MAN[_tag] = (str(_ctx), str(_m.get("fit_target")))
for _row in _PWT:
    _ctx, _fit = _PW_MAN[_row[0]]
    chk(f"v4 README power {_row[0]}: the context its row names",
        f"-c {_ctx}" in _row[5], True)
    chk(f"v4 README power {_row[0]}: and the fitter margin",
        (f"--fit-target {_fit}" in _row[5]) if "fit-target" in _row[5]
        else _fit in ("None", "3072"), True)

chk("v4 README power: the aggregate span it quotes",
    (round(min(_pw_agg), 1), round(max(_pw_agg), 1)), (16.1, 18.7))
chk("v4 README power: and the pooled span",
    (round(min(_pw_pool), 1), round(max(_pw_pool), 1)), (20.9, 23.9))
chk("v4 README power: pooled exceeds aggregate on every row, by about 4 pp",
    (all(_p > _a for _p, _a in zip(_pw_pool, _pw_agg)),
     round(st.mean(_p - _a for _p, _a in zip(_pw_pool, _pw_agg)))), (True, 4))
chk("v4 README power: run J is the top of the aggregate range and has the fewest repeats",
    (_PWT[max(range(6), key=lambda _k: _pw_agg[_k])][0],
     min(int(_cellv4(_r[2])) for _r in _PWT)), ("J", 3))
chk("v4 README power: the five-repeat rows are run L's",
    sorted({_r[0] for _r in _PWT if int(_cellv4(_r[2])) == 5}), ["L, thinking on"])

# Run J's arm table. Four of its rows are checked against literals further up
# this file; the table itself was not, so the no-speculation row, the drafted
# counts and every SD in the aggregate column were unread.
_JT = _v4_table("| arm | pooled | aggregate | vs no speculation | drafted | acceptance |")
chk("v4 README run J table: five rows", len(_JT), 5)
_j_base = _agg(_J2, "baseline")


def _j_sd(arm):
    return st.stdev([json.load(open(_f))["aggregate_tok_s"]
                     for _f in sorted(glob.glob(
                         f"v4_audit_2026_08_25/data/{_J2}/{arm}__rep*.json"))])


for _row in _JT:
    _lab = _norm_early(_row[0]).replace("*", "").split("(")[0].strip()
    _a = "baseline" if _lab == "no speculation" else _lab
    _v = _agg(_J2, _a)
    chk(f"v4 README J table {_a}: three repeats", _v["reps"], 3)
    chk(f"v4 README J table {_a} pooled", round(_v["pooled"], 1), _cellv4(_row[1]), 0.06)
    _ag, _sd = (_cellv4(_x) for _x in _norm_early(_row[2]).split("±"))
    chk(f"v4 README J table {_a} aggregate", round(_v["agg"], 1), _ag, 0.06)
    chk(f"v4 README J table {_a} run-to-run SD", round(_j_sd(_a), 2), _sd, 0.006)
    chk(f"v4 README J table {_a} drafted", _v["drafted"], int(_cellv4(_row[4])))
    if _a == "baseline":
        chk(f"v4 README J table {_a} is the reference and drafts nothing",
            (any(c.isdigit() for c in _row[3] + _row[5]), _v["drafted"]), (False, 0))
        continue
    chk(f"v4 README J table {_a} vs no speculation (%)",
        round(100 * (_v["agg"] / _j_base["agg"] - 1), 1), _cellv4(_row[3]), 0.06)
    chk(f"v4 README J table {_a} acceptance (%)",
        round(_v["acc"], 1), _cellv4(_row[5]), 0.06)
chk("v4 README J table: drafted tokens rise with the draft window",
    [_agg(_J2, f"spec-dflash-n{_n}")["drafted"] for _n in (4, 8, 16)],
    sorted(_agg(_J2, f"spec-dflash-n{_n}")["drafted"] for _n in (4, 8, 16)))
chk("v4 README J table: and acceptance falls with it",
    [round(_agg(_J2, f"spec-dflash-n{_n}")["acc"], 1) for _n in (4, 8, 16)],
    sorted((round(_agg(_J2, f"spec-dflash-n{_n}")["acc"], 1) for _n in (4, 8, 16)),
           reverse=True))

# Runs A and B, the two tables the audit README opens with. Their headline
# figures are asserted further up this file against literals; the tables
# themselves were not read, so the `min` column and the completed counts - which
# are what "the abort is gone" rests on - were unchecked.
_AB_ARM = {"baseline": "baseline",
           "draft, translation fallback": "draft-max8-translate",
           "draft, matched vocabulary": "draft-max8-matched"}


def _ab_row(runs):
    _r = [x for _run in runs for x in _run["rows"]]
    _n = sum(x["predicted_n"] for x in _r)
    _ms = sum(x["predicted_ms"] for x in _r)
    return {"mean": st.mean(x["predicted_per_second"] for x in _r),
            "pooled": 1000 * _n / _ms,
            "min": min(x["predicted_per_second"] for x in _r),
            "drafted": sum(x["draft_n"] for x in _r),
            "accepted": sum(x["draft_n_accepted"] for x in _r),
            "complete": sum(1 for _run in runs if len(_run["rows"]) == 10),
            "reps": len(runs)}


# `_v4_table` finds the FIRST table with a header, and these two share one, so
# they are located by index rather than by name.
_AB_LINES = _V4R_LINES
_ab_seen = []
for _tag, _runs in (("A", A), ("B", B)):
    _hits = [_k for _k, _l in enumerate(_AB_LINES)
             if _l.startswith("| arm | request-mean | pooled | min |")]
    chk("v4 README: the run A and B tables both exist", len(_hits), 2)
    _start = _hits[0 if _tag == "A" else 1]
    _rows = []
    for _l in _AB_LINES[_start + 2:]:
        if not _l.startswith("|"):
            break
        _rows.append([_c.strip().strip("*` ").replace("**", "").strip()
                      for _c in _norm_early(_l).strip("|").split("|")])
    chk(f"v4 README run {_tag}: three arms", len(_rows), 3)
    for _row in _rows:
        _arm = _AB_ARM[_row[0]]
        _v = _ab_row(_runs[_arm])
        _ab_seen.append((_tag, _arm))
        chk(f"v4 README {_tag} {_arm} request-mean",
            round(_v["mean"], 1), _cellv4(_row[1]), 0.06)
        chk(f"v4 README {_tag} {_arm} pooled",
            round(_v["pooled"], 1), _cellv4(_row[2]), 0.06)
        chk(f"v4 README {_tag} {_arm} min", round(_v["min"], 1), _cellv4(_row[3]), 0.06)
        _comp, _of = (int(_cellv4(_x)) for _x in _row[5].split("/"))
        chk(f"v4 README {_tag} {_arm} completed",
            (_v["complete"], _v["reps"]), (_comp, _of))
        if _v["drafted"]:
            _acc, _dr, _pct = (_cellv4(_x) for _x in
                               _row[4].replace("=", "/").split("/"))
            chk(f"v4 README {_tag} {_arm} accepted / drafted",
                (_v["accepted"], _v["drafted"]), (int(_acc), int(_dr)))
            chk(f"v4 README {_tag} {_arm} acceptance (%)",
                round(100 * _v["accepted"] / _v["drafted"], 1), _pct, 0.06)
        else:
            chk(f"v4 README {_tag} {_arm} drafts nothing, so no ratio cell",
                any(_c.isdigit() for _c in _row[4]), False)
chk("v4 README: both runs' tables were read, six rows",
    sorted(_ab_seen), sorted(set(_ab_seen)))
chk("v4 README A: neither draft arm completed a single repeat",
    [_ab_row(A[_a])["complete"] for _a in
     ("draft-max8-translate", "draft-max8-matched")], [0, 0])
chk("v4 README B: both of them complete every repeat",
    [_ab_row(B[_a])["complete"] for _a in
     ("draft-max8-translate", "draft-max8-matched")], [3, 3])

# The thinking on/off comparison, which three documents carry and none read.
# Runs C and D, same ten prompts, and the claim it supports is that `ngram-mod`
# stops drafting altogether when the thinking block goes away.
_D = "v4_audit_2026_08_25/data/D_master_matrix_think_off/%s__rep*.json"


def _cd(pattern, arm):
    _rs = [json.load(open(_f)) for _f in sorted(glob.glob(pattern % arm))]
    _n = sum(x["predicted_n"] for r in _rs for x in r["rows"])
    _ms = sum(x["predicted_ms"] for r in _rs for x in r["rows"])
    _dn = sum(x["draft_n"] for r in _rs for x in r["rows"])
    _da = sum(x["draft_n_accepted"] for r in _rs for x in r["rows"])
    return {"pooled": 1000 * _n / _ms, "draft_per_gen": _dn / _n,
            "acc": 100 * _da / _dn if _dn else None, "reps": len(_rs)}


_TH_ARM = {"ngram-mod n=24": "ngram-mod-n24", "ngram-cache": "ngram-cache",
           "draft model n_max 8": "spec-draft-n8"}
_TH_BASE = {"C": _cd(_C, "baseline")["pooled"], "D": _cd(_D, "baseline")["pooled"]}
_TH_SEEN = {}
# ERRATA's own line list is built further down this file, so this reads it here
_TH_ER = (pathlib.Path(__file__).resolve().parents[1]
          / "ERRATA.md").read_text(encoding="utf-8").splitlines()
for _doc, _lines in (("v4 README", _V4R_LINES), ("README", _ROOT_LINES),
                     ("ERRATA", _TH_ER)):
    _k = next((_j for _j, _l in enumerate(_lines)
               if _l.startswith("| method | thinking on")), None)
    chk(f"{_doc}: it carries the thinking on/off table", _k is not None, True)
    _rows = []
    for _l in _lines[_k + 2:]:
        if not _l.startswith("|"):
            break
        _rows.append([_c.replace("`", "").replace("*", "").strip()
                      for _c in _norm_early(_l).strip("|").split("|")])
    chk(f"{_doc} thinking table: three methods", len(_rows), 3)
    for _row in _rows:
        # one copy writes `draft model, n_max 8` and another `draft model
        # n_max 8`; the comma is the only difference between them
        _a = _TH_ARM[_row[0].replace("draft model,", "draft model")]
        _on, _off = _cd(_C, _a), _cd(_D, _a)
        chk(f"{_doc} thinking {_a} on (%)",
            round(100 * (_on["pooled"] / _TH_BASE["C"] - 1), 1), _cellv4(_row[1]), 0.06)
        chk(f"{_doc} thinking {_a} off (%)",
            round(100 * (_off["pooled"] / _TH_BASE["D"] - 1), 1), _cellv4(_row[2]), 0.06)
        _a_on, _a_off = (_cellv4(_x) for _x in
                         _norm_early(_row[3]).replace("\u2192", "->").split("->"))
        chk(f"{_doc} thinking {_a} draft per generated token, on then off",
            (round(_on["draft_per_gen"], 2), round(_off["draft_per_gen"], 2)),
            (_a_on, _a_off), 0.006)
        _TH_SEEN.setdefault(_a, []).append(
            (_cellv4(_row[1]), _cellv4(_row[2]), _a_on, _a_off))
chk("the three copies of the thinking table agree cell for cell",
    sorted((_a, len(set(_v))) for _a, _v in _TH_SEEN.items()),
    sorted((_a, 1) for _a in _TH_ARM.values()))
chk("with thinking off ngram-mod drafts nothing at all",
    (_cd(_D, "ngram-mod-n24")["draft_per_gen"], _cd(_D, "ngram-mod-n24")["acc"]),
    (0.0, None))
# 23.09 %, which three documents printed as 23.0 until 2026-08-29. Every other
# percentage in this repository is rounded, and no definition of this one gives
# 23.0: pooled and the mean of the five repeats both give 23.09, the mean over
# the ten prompts gives 27.3. It was a truncation, in three places.
_TH_ACC = (round(_cd(_C, "spec-draft-n8")["acc"], 1),
           round(_cd(_D, "spec-draft-n8")["acc"], 1))
chk("the draft model's acceptance, thinking on then off", _TH_ACC, (29.7, 23.1))
for _doc in ("README.md", "ERRATA.md", "v4_audit_2026_08_25/README.md"):
    _t = (pathlib.Path(__file__).resolve().parents[1] / _doc) \
        .read_text(encoding="utf-8")
    chk(f"{_doc} prints the rounded value, not the truncated one",
        (f"to {_TH_ACC[1]} %" in _t, "to 23.0 %" in _t), (True, False))

# Run P against run O: the same four arms on two prompt sets sharing no prompt.
# Sixteen cells and the sentence "the result generalises" rest on them; the
# aggregate column is parenthesised as not comparable and was unread with the
# rest.
_P = "matrix_P_extended_20260826_110747"
_PT_O, _PT_P = _agg(_O_DIR, "baseline"), _agg(_P, "baseline")
_PGT = _v4_table("| arm | v1 ten, pooled | extended twenty, pooled | shift |")
chk("v4 README run P table: four arms", len(_PGT), 4)
for _row in _PGT:
    _a = _row[0]
    _o, _p = _agg(_O_DIR, _a), _agg(_P, _a)
    _d_o = 100 * (_o["pooled"] / _PT_O["pooled"] - 1)
    _d_p = 100 * (_p["pooled"] / _PT_P["pooled"] - 1)
    chk(f"v4 README P {_a}: three repeats on each set", (_o["reps"], _p["reps"]), (3, 3))
    chk(f"v4 README P {_a} on the v1 ten (%)", round(_d_o, 1), _cellv4(_row[1]), 0.06)
    chk(f"v4 README P {_a} on the extended twenty (%)",
        round(_d_p, 1), _cellv4(_row[2]), 0.06)
    chk(f"v4 README P {_a} shift (pp)",
        round(_d_p - _d_o, 1), _cellv4(_row[3]), 0.06)
    _ag_o, _ag_p = (_cellv4(_x) for _x in
                    _norm_early(_row[4]).replace("\u2192", "->").split("->"))
    chk(f"v4 README P {_a} aggregate, O then P (%)",
        (round(100 * (_o["agg"] / _PT_O["agg"] - 1), 1),
         round(100 * (_p["agg"] / _PT_P["agg"] - 1), 1)), (_ag_o, _ag_p), 0.06)
# "one arm moves upward" was two: spec-dflash-n4 gains 3.3 pp and
# spec-draft-n8 loses 0.6 pp less than it did. The sentence names the arm now.
chk("v4 README P: the largest shift, and how many move upward",
    (max(abs(_cellv4(_r[3])) for _r in _PGT),
     sum(1 for _r in _PGT if _cellv4(_r[3]) > 0)), (4.3, 2))
chk("v4 README P: the arm the sentence names is the one that gains most",
    max(_PGT, key=lambda _r: _cellv4(_r[3]))[0], "spec-dflash-n4")
chk("v4 README P: and the sentence no longer says one",
    "and one arm moves *upward*" in _V4R_TEXT, False)
# the same sentence is in the root README, and only the audit copy was fixed
# first; and the root README also had the acceptance ordering backwards
chk("README P: it does not say one arm either",
    "and one arm moves *upward*" in _ROOT_TEXT, False)

chk("v4 README P: on aggregate it would have read as the win halving, which is "
    "the artefact the section is about",
    round(100 * (_agg(_P, "spec-dflash-n2")["agg"] / _PT_P["agg"] - 1), 1)
    < round(100 * (_agg(_O_DIR, "spec-dflash-n2")["agg"] / _PT_O["agg"] - 1), 1) / 2 + 1,
    True)

# Run Q against runs M1 and M4: three of four reproduce, and the fourth is the
# measurement that created the anomaly. Twelve cells and the whole argument that
# M1's +10.5 % is the outlier.
_Q = {("spec-mtp-n2", "Q8_0"): ("matrix_M1_20260826_075816",
                                "matrix_Q_q8_20260826_110747"),
      ("spec-mtp-n2", "Q4_K_M"): ("matrix_M4_q4km_20260826_081806",
                                  "matrix_Q_q4km_20260826_110747"),
      ("spec-mtp-n4", "Q4_K_M"): ("matrix_M4_q4km_20260826_081806",
                                  "matrix_Q_q4km_20260826_110747"),
      ("spec-mtp-n4", "Q8_0"): ("matrix_M1_20260826_075816",
                                "matrix_Q_q8_20260826_110747")}


def _q_delta(run, arm):
    _v, _b = _agg(run, arm), _agg(run, "baseline")
    return 100 * (_v["pooled"] / _b["pooled"] - 1), _v["reps"]


_QT = _v4_table("| arm | drafter | 3 repeats (M1 / M4) |")
chk("v4 README run Q table: four rows", len(_QT), 4)
for _row in _QT:
    _key = (_row[0], _row[1])
    chk(f"v4 README Q {_key}: the row names a pair this repository ran",
        _key in _Q, True)
    _old_run, _new_run = _Q[_key]
    _d_old, _n_old = _q_delta(_old_run, _row[0])
    _d_new, _n_new = _q_delta(_new_run, _row[0])
    chk(f"v4 README Q {_key} repeats, three then five", (_n_old, _n_new), (3, 5))
    chk(f"v4 README Q {_key} the three-repeat figure (%)",
        round(_d_old, 1), _cellv4(_row[2]), 0.06)
    chk(f"v4 README Q {_key} the five-repeat figure (%)",
        round(_d_new, 1), _cellv4(_row[3]), 0.06)
    chk(f"v4 README Q {_key} difference (pp)",
        round(abs(_d_new - _d_old), 1), _cellv4(_row[4]), 0.06)
chk("v4 README Q: three of the four reproduce to within 0.5 pp",
    sum(1 for _r in _QT if _cellv4(_r[4]) <= 0.5), 3)
chk("v4 README Q: and the one that does not is the arm that made the anomaly",
    max(_QT, key=lambda _r: _cellv4(_r[4]))[:2], ["spec-mtp-n4", "Q8_0"])
# the third, independent reading of that arm, on the extended prompt set
chk("v4 README Q: run P reads the same arm at +2.7 %",
    round(_q_delta(_P, "spec-mtp-n4")[0], 1), 2.7, 0.06)
chk("v4 README Q: so +2.0, +2.7 and +3.6 cluster and +10.5 is the outlier",
    sorted(round(_x, 1) for _x in
           (_q_delta("matrix_Q_q8_20260826_110747", "spec-mtp-n4")[0],
            _q_delta(_P, "spec-mtp-n4")[0],
            _q_delta("matrix_Q_q4km_20260826_110747", "spec-mtp-n4")[0])),
    [2.0, 2.7, 3.6])

_M1T = {r[0]: r[1:] for r in
        _v4_table("| arm | aggregate | vs no speculation | acceptance |")}
chk("the v4 README's M1 table rows", len(_M1T), 7)
chk("and the no-speculation row is one of them", "no speculation" in _M1T, True)
chk("v4 README M1 no-speculation aggregate",
    round(_b["agg"], 1), _cellv4(_M1T["no speculation"][0]), 0.05)
for _arm, _row in sorted(_M1T.items()):
    if _arm == "no speculation":
        continue
    _v = _agg("matrix_M1_*", _arm)
    chk(f"v4 README M1 {_arm} aggregate", round(_v["agg"], 1), _cellv4(_row[0]), 0.05)
    chk(f"v4 README M1 {_arm} vs baseline (%)",
        round(100 * (_v["agg"] / _b["agg"] - 1), 1), _cellv4(_row[1].rstrip("\u2021 ")), 0.05)
    chk(f"v4 README M1 {_arm} acceptance (%)", round(_v["acc"], 1), _cellv4(_row[2]), 0.05)

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
_A13_NOCK, _A13_CK = list(_z), list(_n)
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
    """Unrounded. Rounding each side and then subtracting gave 8.5 where the
    measurements are 8.573 apart, which is the double rounding A12 already
    corrects elsewhere (four rounded components adding to 39.08 where the
    figure is 39.07). Every other difference column
    here - run P's shift, the O2/O3 replication table, A17's - subtracts first
    and rounds once; this was the one that did not."""
    b = _pl(pat, "baseline"); v = _pl(pat, arm)
    return 100 * (v["pooled"] / b["pooled"] - 1)
_m1, _q8, _pe = _d("matrix_M1_*"), _d("matrix_Q_q8_*"), _d("matrix_P_extended_*")
chk("A14 the non-replicating pair, gap (pp)", round(_m1 - _q8, 1), 8.6, 0.05)
chk("A14: rounding each side first is what gave 8.5",
    round(round(_m1, 1) - round(_q8, 1), 1), 8.5, 0.005)
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
_SCORECARD = _sc(_kept, 3)
_SC_KEPT = list(_kept)
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
_A12_EXCESS_S, _A12_CKPT_S = _exc, _ck
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
    ("ERRATA.md",   "8.57 pp",    "A14 the pair that did not replicate"),
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
    # whitespace-collapsed: a claim is the same claim wherever the line wraps,
    # and two of these broke on a re-wrap that changed no word
    txt = " ".join(_norm((root / f).read_text(encoding="utf-8")).split())
    ok = " ".join(_norm(needle).split()) in txt
    print(f"  {'PASS' if ok else 'FAIL'}  {f:32s} quotes {needle!r:20s} ({what})")
    RAN.append(f"{f}:{needle}")     # these are assertions too, and are counted
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
# A SHALLOW clone is not the same as no clone, and telling them apart matters.
# `tests/data_mutate.py` runs this in a mirror with no `.git` at all, which is
# legitimate and skipped. `actions/checkout` defaults to depth 1, where
# `rev-parse --git-dir` still succeeds and the history is simply absent - so
# every provenance assertion failed with a list of run directories and nothing
# said why. That cost one full evidence-workflow run to diagnose. A shallow
# clone in CI is a configuration error and says so.
if _HAS_GIT:
    _shallow = _sp2.run(["git", "-C", str(_repo), "rev-parse",
                         "--is-shallow-repository"],
                        capture_output=True, text=True).stdout.strip() == "true"
    if _shallow:
        sys.exit(
            "  ----  this is a SHALLOW clone. The harness-provenance checks "
            "resolve each run's `runner_sha256` against every version of\n"
            "        bench/retest_runner.py this repository has held, which a "
            "depth-1 clone cannot do.\n"
            "        In GitHub Actions add `with: { fetch-depth: 0 }` to the "
            "checkout step; see .github/workflows/audit.yml.\n"
            "        Refusing rather than reporting five assertion failures "
            "that do not name the cause.")
if not _HAS_GIT:
    print("  ----  no git history here (a mirror); "
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
# A harness version can also be archived rather than committed, which happened
# once: run W's runner was deployed from the working tree and edited twice
# before it was committed, so the source that produced 500 arm-runs was never a
# commit. Hiding that would be worse than recording it - the file is archived,
# byte for byte, and the pin resolves to something real.
_ARCH = {}
_arch_dir = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "harness"
if _arch_dir.is_dir():
    for _f in sorted(_arch_dir.glob("*.py")):
        _ARCH[hashlib.sha256(_f.read_bytes()).hexdigest()] = _f.name
chk("archived harness versions, for runs whose runner was never committed",
    sorted(_ARCH.values()), ["retest_runner_W_20260828_104222.py"])
if _HAS_GIT and _blobs:
    chk("every recorded harness hash resolves, in history or in the archive",
        sorted(k for k, v in _declared.items()
               if v not in _blobs and v not in _ARCH), [])
    # `matrix_W_` and `matrix_W2_`: W2 ran the same runner as W, which was
    # never committed, so it resolves through the archive for the same reason
    # rather than through a second copy of the same file.
    chk("and the archive is used only where history does not have it",
        sorted(k for k, v in _declared.items()
               if v in _ARCH and v not in _blobs),
        sorted(k for k in _declared
               if k.startswith(("matrix_W_", "matrix_W2_"))))
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
    dict(_stops["on"]), {"length": 5904})
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
# runs W and W2 add 1700 thinking-off arm-runs, half of them capped
chk("A17 thinking-off requests that stopped early", _stops["off"]["stop"], 9391)
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
_ER_LINES_TEXT = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("ERRATA.md").read_text(encoding="utf-8")
_ER_LINES = pathlib.Path(__file__).resolve().parents[1].joinpath("ERRATA.md") \
    .read_text(encoding="utf-8").splitlines()


def _md_table(header_startswith, quoted=False):
    """Read one ERRATA table. `quoted=True` strips the blockquote marker first,
    for tables that live inside a `> [!IMPORTANT]` note."""
    lines = ([l[2:] if l.startswith("> ") else l.rstrip(">") for l in _ER_LINES]
             if quoted else _ER_LINES)
    i = next(i for i, l in enumerate(lines) if l.startswith(header_startswith))
    if quoted:
        rows = []
        for l in lines[i + 2:]:
            if not l.startswith("|"):
                break
            rows.append([c.strip().strip("*`").replace("`", "").strip("* ").strip()
                         for c in l.strip("|").split("|")])
        return rows
    rows = []
    for l in _ER_LINES[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip().strip("*`").replace("`", "").strip("* ").strip()
                     for c in l.strip("|").split("|")])
    return rows


# --- A4's reconstruction table, whose every number was derived above and
# compared against a literal instead of against the table that prints it -----
# The chain length is the one figure that needed the log's ORDER: a run of
# consecutive verifications between two draft generations is one chain, and
# the longest is 2 because every partial accept was re-verified exactly once.
_a4_runs, _a4_cur = [], 0
for _m in re.finditer(
        r"called impl \w+, hist size = \d+, call_count = \d+, gen = \d+"
        r"|ignoring small draft: \d+ < \d+"
        r"|update_slots: n_draft=\d+, accepted=\d+", _A4["log"]):
    if _m.group(0).startswith("update_slots"):
        _a4_cur += 1
    else:
        if _a4_cur:
            _a4_runs.append(_a4_cur)
        _a4_cur = 0
if _a4_cur:
    _a4_runs.append(_a4_cur)
chk("A4: the verification chains account for every attempt",
    sum(_a4_runs), _A4["attempts"])
chk("A4: and there is one chain per draft that reached verification",
    len(_a4_runs), _A4["fresh"])
_A4_SIZE1 = 1
_A4_EXPECT = {
    "drafts generated": [_A4["gen_drafts"], _A4["gen_tokens"]],
    "dropped by --draft-min 2 before any verification": [
        _A4["min_flag"], _A4["dropped_lines"], _A4["dropped_tokens"],
        _A4["empty"], _A4["size1"], _A4_SIZE1, _A4["events"],
        _A4["dropped_lines"], _A4["fresh"]],
    "reached verification as a fresh draft": [
        _A4["fresh"], _A4["fresh_tokens"], _A4["gen_tokens"],
        _A4["dropped_tokens"], _A4["fresh_tokens"]],
    "verification attempts": [_A4["attempts"], _A4["fresh"], _A4["partial"]],
    "partially accepted -> discarded and redone": [_A4["partial"]],
    "longest verify chain": [max(_a4_runs)],
    "server counter": [_A4["acc_tokens"], _A4["counter"], _A4["full"]],
}
_A4T = _num_rows(_ER_LINES, "| quantity | value | independent check |")
chk("A4 reconstruction table: one row per quantity",
    sorted(_A4T), sorted(_A4_EXPECT))
for _q, _want in sorted(_A4_EXPECT.items()):
    _num_row_check(f"A4 table {_q}", _A4T[_q], _want)
chk("A4: every small draft that was not empty had exactly one token",
    sorted({_x for _x in _A4["small"] if _x}), [_A4_SIZE1])
chk("A4: the partial accepts equal the checkpoint restores",
    (_A4["partial"], _A4["restores"]), (20, 20))


# --- A16's six invocations: the clock they started at and what each measured
# Fifteen minutes apart by the manifests' own timestamps, so the start row is
# not decoration - it is the evidence that the scatter is not drift.
def _pool_dir(run_dir, arm):
    """Pooled decode rate over every repeat, without importing `length_mode`,
    whose name is not free this early in the file."""
    _ms = _n = 0
    for _f in sorted(glob.glob(os.path.join(run_dir, f"{arm}__rep*.json"))):
        _b = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
        if _b.get("crashed"):
            continue
        _ms += sum(_r["timings"]["predicted_ms"] for _r in _b["rows"])
        _n += sum(_r["timings"]["predicted_n"] for _r in _b["rows"])
    return 1000.0 * _n / _ms


_UDIRS = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_U*"))
_u_start, _u_shift = [], []
for _d in _UDIRS:
    _m = json.loads((pathlib.Path(_d) / "manifest.json").read_text(encoding="utf-8"))
    _hh, _mm = _m["created"][11:13], _m["created"][14:16]
    _u_start += [int(_hh), int(_mm)]
    _u_shift.append(round(100 * (_pool_dir(_d, "spec-dflash-n2")
                                 / _pool_dir(_d, "baseline") - 1), 1))
_UT = _num_rows(_ER_LINES, "| | U1 | U2 | U3 | U4 | U5 | U6 |")
chk("A16 invocation table: six invocations", len(_UDIRS), 6)
chk("A16 invocation table: its two rows", sorted(_UT),
    ["spec-dflash-n2 vs baseline", "start"])
_num_row_check("A16 invocation start times", _UT["start"], _u_start)
_num_row_check("A16 invocation shifts", _UT["spec-dflash-n2 vs baseline"],
               _u_shift)
chk("A16: they really are about a quarter of an hour apart end to end",
    (_u_start[-2] * 60 + _u_start[-1]) - (_u_start[0] * 60 + _u_start[1]), 15)


# --- B4's family minima: four rows of v1 per-prompt worst cases ------------
# Every one of them is a min over `analysis/summary.csv`, and the table was
# read by nothing. The band quoted under it is the min and max of the two
# families it names, so a wrong cell there had to stay consistent with both.
_B4MIN = {_c: min(float(_r["tok_s"]) for _r in _V1CSV if _r["config"] == _c)
          for _c in {_r["config"] for _r in _V1CSV}}
_B4_EXPECT = {
    "ngram-mod n = 8 / 12 / 16 / 20 / 24": [
        8, 12, 16, 20, 24,
        _B4MIN["ngmod-n8"], _B4MIN["ngmod-n12"], _B4MIN["ngmod-n16"],
        _B4MIN["ngmod-n20"], _B4MIN["ngram-mod-n24"]],
    "ngram-cache family": [
        _B4MIN["ngram-cache"], _B4MIN["ngcache-rerun"], _B4MIN["ngcache-1000tok"]],
    "ngcache-kv-fp16": [_B4MIN["ngcache-kv-fp16"]],
    "classic draft max 8 / 16 / 32": [
        8, 16, 32, _B4MIN["draft-q35-08b-max8"], _B4MIN["draft-q35-08b-max16"],
        _B4MIN["draft-q35-08b-max32"]],
}
_B4T = _num_rows(_ER_LINES, "| family | minimum |")
chk("B4 family table: one row per family", sorted(_B4T), sorted(_B4_EXPECT))
for _f, _want in sorted(_B4_EXPECT.items()):
    _num_row_check(f"B4 {_f}", _B4T[_f], [round(_x, 1) for _x in _want])
_b4_band = [_B4MIN[_c] for _c in ("ngram-cache", "ngcache-rerun",
                                  "ngcache-1000tok", "ngcache-kv-fp16",
                                  "draft-q35-08b-max8", "draft-q35-08b-max16",
                                  "draft-q35-08b-max32")]
chk("B4: the band under it is the range of the rows it attributes it to",
    f"The {int(min(_b4_band))}\u2013{round(max(_b4_band))} band belongs to specific"
    in " ".join(_ER_LINES_TEXT.split()), True)
_b4_ngmod = min(_B4MIN[_c] for _c in
                ("ngmod-n8", "ngmod-n12", "ngmod-n16", "ngmod-n20",
                 "ngram-mod-n24"))
chk("B4: and the ngram-mod family's worst really is inside 12 % of baseline",
    math.ceil(100 * (1 - _b4_ngmod / _B4MIN["baseline"])), 12)


# A7 contrast 1: the per-configuration Pearson r and the per-prompt acceptance
# range behind the README's "5 % to 83 %". Seven rows, twenty-one cells, and
# the paragraph under it that argues the seventh is not independent.
_A7R = _md_table("| configuration | Pearson r | acceptance range across prompts |")
chk("ERRATA A7 correlation table: seven rows", len(_A7R), 7)
# the seventh row's label carries v1's two draft-length flags, and they are in
# the manifest of the run that reproduced that configuration
_v1cfg = json.loads(pathlib.Path(
    "v4_audit_2026_08_25/data/C_master_matrix_think_on/manifest.json")
    .read_text(encoding="utf-8"))["arms"]["spec-draft-v1cfg"]
chk("A7: v1's configuration is max 8, min 4, as the run that reran it recorded",
    (_v1cfg[_v1cfg.index("--spec-draft-n-max") + 1],
     _v1cfg[_v1cfg.index("--spec-draft-n-min") + 1]), ("8", "4"))
chk("A7: and the row is labelled with both",
    f"v1's configuration (max {_v1cfg[_v1cfg.index('--spec-draft-n-max') + 1]}, "
    f"min {_v1cfg[_v1cfg.index('--spec-draft-n-min') + 1]})" in _ER_LINES_TEXT,
    True)
chk("A7: it differs from n_max 8 only in n_min",
    [_x for _x in _v1cfg
     if _x not in json.loads(pathlib.Path(
         "v4_audit_2026_08_25/data/C_master_matrix_think_on/manifest.json")
         .read_text(encoding="utf-8"))["arms"]["spec-draft-n8"]], ["4"])


def _c_prompt_acc(arm):
    _da, _dn = defaultdict(int), defaultdict(int)
    for _f in sorted(glob.glob(_C % arm)):
        for _x in json.load(open(_f))["rows"]:
            _da[_x["tag"]] += _x["draft_n_accepted"]
            _dn[_x["tag"]] += _x["draft_n"]
    return [100 * _da[_t] / _dn[_t] for _t in sorted(_dn) if _dn[_t]]


for _row in _A7R:
    _lab = _row[0]
    _a = ("spec-draft-v1cfg" if _lab.startswith("v1's")
          else f"spec-draft-n{_lab.split()[-1]}")
    chk(f"ERRATA A7 correlation {_a} Pearson r",
        round(_c_prompt_r(_a), 3), _cellv4(_row[1]), 0.0006)
    _lo, _hi = (_cellv4(_x) for _x in _norm_early(_row[2]).replace("%", "").split(" - "))
    _pp = _c_prompt_acc(_a)
    chk(f"ERRATA A7 correlation {_a} acceptance range, low", min(_pp), _lo, 0.06)
    chk(f"ERRATA A7 correlation {_a} acceptance range, high", max(_pp), _hi, 0.06)
chk("ERRATA A7 correlation: every row is at r >= +0.996",
    min(round(_c_prompt_r("spec-draft-v1cfg" if _r[0].startswith("v1's")
                          else f"spec-draft-n{_r[0].split()[-1]}"), 3)
        for _r in _A7R) >= 0.996, True)
# the seventh row is n_max 8 with a different n_min, and the paragraph says so
_A7_v1, _A7_n8 = _cstat("spec-draft-v1cfg"), _cstat("spec-draft-n8")
chk("ERRATA A7: v1's configuration against n_max 8, pooled",
    (round(_A7_v1["pooled"], 2), round(_A7_n8["pooled"], 2)), (32.27, 32.10))
chk("ERRATA A7: and their acceptance, to two places",
    (round(_A7_v1["acc"], 2), round(_A7_n8["acc"], 2)), (29.69, 29.67))
_A7_dt = {}
for _a in ("spec-draft-v1cfg", "spec-draft-n8"):
    _d = defaultdict(int)
    for _f in sorted(glob.glob(_C % _a)):
        for _x in json.load(open(_f))["rows"]:
            _d[_x["tag"]] += _x["draft_n"]
    _A7_dt[_a] = _d
chk("ERRATA A7: three of ten prompts draft a byte-identical number of tokens",
    sum(1 for _t in _A7_dt["spec-draft-n8"]
        if _A7_dt["spec-draft-n8"][_t] == _A7_dt["spec-draft-v1cfg"][_t]), 3)

# B1's table, the same v1 numbers as the README's but with the long-output row
# comparing against two different references in one row - the request-mean
# against the 300-token baseline and the pooled rate against
# `baseline-1000tok`. Which reference a cell uses is read out of the cell.
_B1 = _md_table("| config | request-mean | pooled | median | min |")
chk("ERRATA B1 table: five rows", len(_B1), 5)
for _row in _B1:
    _c = _row[0]
    chk(f"ERRATA B1 {_c}: a config summary.csv holds", _c in by, True)
    _v = _v1agg(_c)
    for _col, _key in ((1, "mean"), (2, "pooled"), (3, "med"), (4, "mn")):
        _got, _delta = _v1_cell(_row[_col])
        chk(f"ERRATA B1 {_c} {_key}", _v[_key], _got[0], 0.06)
        if _delta:
            _ref = ("baseline-1000tok" if "baseline-1000tok" in _row[_col]
                    else "baseline")
            chk(f"ERRATA B1 {_c} {_key} vs {_ref} (%)",
                100 * (_v[_key] / _v1agg(_ref)[_key] - 1), _delta[0], 0.06)
chk("ERRATA B1: the long-output row names its reference in both cells",
    [_r[0] for _r in _B1 if "baseline-1000tok" in _r[1] + _r[2]], ["ngcache-1000tok"])
chk("ERRATA B1: against the 300-token baseline that row would read -14.6 %",
    100 * (_v1agg("ngcache-1000tok")["mean"] / _v1agg("baseline")["mean"] - 1),
    -14.6, 0.06)
chk("ERRATA B1: and the divergence it exists to show, in percentage points",
    round(abs(100 * (_v1agg("ngram-cache")["mean"] / _v1agg("baseline")["mean"] - 1))
          - abs(100 * (_v1agg("ngram-cache")["pooled"] / _v1agg("baseline")["pooled"] - 1)),
          1), -5.8, 0.06)

# A7's draft-volume ordering table: six of run C's arms, sorted by draft volume,
# and the row the argument turns on is the fourth. Nothing read it, so the
# sentence "an external draft model proposing 0.50 tokens per generated token
# runs at 31.1 tok/s, while ngram-cache proposing 0.42 runs at 74.0" could have
# been built on two wrong numbers.
_A7V = _md_table("| arm | draft tokens per generated token | pooled tok/s | acceptance |")
chk("ERRATA A7 volume table: six arms", len(_A7V), 6)
for _row in _A7V:
    _a = _c_lookup(_row[0])
    _s = _cstat(_a)
    chk(f"ERRATA A7 volume {_a} draft tokens per generated token",
        round(_s["draft_per_gen"], 2), _cellv4(_row[1]), 0.006)
    chk(f"ERRATA A7 volume {_a} pooled", round(_s["pooled"], 1), _cellv4(_row[2]), 0.06)
    chk(f"ERRATA A7 volume {_a} acceptance (%)",
        round(_s["acc"], 1), _cellv4(_row[3]), 0.06)
chk("ERRATA A7 volume table: sorted by draft volume, ascending",
    [_cellv4(r[1]) for r in _A7V], sorted(_cellv4(r[1]) for r in _A7V))
# the inversion the section is about: more volume, and yet faster
_A7_ng, _A7_dm = _cstat("ngram-cache"), _cstat("spec-draft-n1")
chk("ERRATA A7: ngram-cache drafts less than the n_max 1 draft model",
    _A7_ng["draft_per_gen"] < _A7_dm["draft_per_gen"], True)
chk("ERRATA A7: and is more than twice as fast, which volume alone cannot explain",
    round(_A7_ng["pooled"] / _A7_dm["pooled"], 2) > 2.0, True)

# A18: the spread claim that was a range over the first six arms, and the two
# estimators the SD column could have been. Both endpoints of the corrected
# sentence are derived here, so the correction cannot rot the way the original
# did.
_A18 = _md_table("| arm | run-to-run SD, five repeats | with rep 0 dropped |")
chk("ERRATA A18: the arms it lists", len(_A18), 6)
for _row in _A18:
    _a = _c_lookup(_row[0])
    _r = _c_reps(_a)
    chk(f"ERRATA A18 {_a} SD of five repeats",
        round(st.stdev(_r), 2), _cellv4(_row[1]), 0.006)
    chk(f"ERRATA A18 {_a} SD with rep 0 dropped",
        round(st.stdev(_r[1:]), 2), _cellv4(_row[2]), 0.006)
chk("ERRATA A18: five of the six are strictly above the 0.48 the caption claimed",
    sum(1 for _row in _A18
        if round(st.stdev(_c_reps(_c_lookup(_row[0]))[1:]), 2) > 0.48), 5)
chk("ERRATA A18: and the sixth is 0.48 itself, which is where the claim came from",
    round(st.stdev(_c_reps("spec-draft-n8")[1:]), 2), 0.48, 0.006)
_c_all_sd = sorted((round(st.stdev(_c_reps(_a)[1:]), 2), _a)
                   for _a in set(_C_ARM.values()))
chk("ERRATA A18: 0.48 is the sixth-smallest of the eleven, the seventh of thirteen",
    ([_a for _v, _a in _c_all_sd if _a not in ("baseline", "ngram-cache")]
     .index("spec-draft-n8") + 1,
     [_a for _v, _a in _c_all_sd].index("spec-draft-n8") + 1), (6, 7))

_c_other = {_a: st.stdev(_c_reps(_a)[1:]) for _a in set(_C_ARM.values())
            if _a not in ("baseline", "ngram-cache")}
chk("ERRATA A18: eleven arms other than baseline and ngram-cache", len(_c_other), 11)
chk("ERRATA A18: their range with rep 0 dropped",
    (round(min(_c_other.values()), 2), round(max(_c_other.values()), 2)), (0.03, 1.01))
chk("ERRATA A18: the top of that range is ngram-cache-kvfp16",
    max(_c_other, key=_c_other.get), "ngram-cache-kvfp16")
chk("ERRATA A18: so ngram-cache leads by a factor of 1.8, not fourfold",
    round(st.stdev(_c_reps("ngram-cache")[1:]) / max(_c_other.values()), 1), 1.8, 0.05)

# which estimator the published column actually is - the claim the caption now
# makes, and the reason it is worth making
_sd_pub = {_c_lookup(_r[0]): _cellv4(_r[5]) for _r in _CT}
chk("v4 README C: a published SD for every arm", len(_sd_pub), 13)
_sd_hits = tuple(
    sum(1 for _a, _v in _sd_pub.items()
        if abs(round(st.stdev(_c_reps(_a, _how)), 2) - _v) <= 0.006)
    for _how in ("pooled", "mean"))
chk("v4 README C: the SD column is the request mean's, not the pooled rate's",
    _sd_hits, (3, 13))
chk("v4 README C: the caption says which of the two it is",
    "SD of the five repeats' **request means**" in _V4R_TEXT, True)
chk("v4 README C: and the published range of the column",
    (round(min(_sd_pub.values()), 2), round(max(_sd_pub.values()), 2)), (0.04, 2.48))
chk("v4 README C: ngram-cache's five repeats, as the caption prints them",
    ", ".join(f"{_v:.1f}" for _v in _c_reps("ngram-cache")) + " tok/s" in _V4R_TEXT,
    True)
chk("v4 README C: baseline's cold start removed",
    f"excluding it the SD is {st.stdev(_c_reps('baseline')[1:]):.2f}" in _V4R_TEXT,
    True)
chk("v4 README C: ngram-cache's cold start removed",
    f"still {st.stdev(_c_reps('ngram-cache')[1:]):.2f} after" in _V4R_TEXT, True)


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
chk("A12 table: and the row they are shares OF says so",
    _f(_acc["excess to account for"][1]), 100.0)
chk("A12 table: which is the excess divided by itself",
    round(100 * _excess / _excess, 1), _f(_acc["excess to account for"][1]))

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
chk("manifest entries", len(_mlines), 3043)
chk("every entry is a sha256 and a path",
    sorted({bool(re.fullmatch(r"[0-9a-f]{64}  \S.*", l)) for l in _mlines}), [True])
chk("logs in the manifest", sum(1 for l in _mlines if l.endswith(".log")), 3020)
chk("telemetry traces in the manifest",
    sum(1 for l in _mlines if l.endswith(".csv")), 23)
chk("no duplicate paths", len({l.split("  ", 1)[1] for l in _mlines}), len(_mlines))
# The manifest now has a published half and a pending one: W2's logs are hashed
# and not packaged. `sha256sum -c` over the whole file reports 1201 missing
# files on a tree where nothing is wrong, so `evidence.yml` splits on a marker.
# A marker a workflow depends on is a structural claim, and this is where it is
# checked without waiting for CI. Both directions, because moving a published
# entry below the line would hide a failed digest exactly as well as it hides a
# pending one.
_EVY = (pathlib.Path(__file__).resolve().parents[1] / ".github"
        / "workflows" / "evidence.yml").read_text(encoding="utf-8")
_mtext = _man.read_text(encoding="utf-8").splitlines()
_mmark = [i for i, l in enumerate(_mtext)
          if l.startswith("# --- PENDING TRANCHE")]
chk("manifest: exactly one pending-tranche marker", len(_mmark), 1)
_mpub = [l for l in _mtext[:_mmark[0]] if l and not l.startswith("#")]
_mpend = [l for l in _mtext[_mmark[0]:] if l and not l.startswith("#")]
chk("manifest: the published half is the three released tranches",
    (len(_mpub), sum(1 for l in _mpub if l.endswith(".log")),
     sum(1 for l in _mpub if l.endswith(".csv"))), (1842, 1820, 22))
# the needle is the run stamp, not "W2_20260830_220554": the log paths are
# `matrix_W2_s8_20260830_220554/...`, so the session number sits between the
# label and the stamp and the obvious needle matched only the trace
chk("manifest: the pending half is run W2 and nothing else",
    (len(_mpend), sorted({"20260830_220554" in l for l in _mpend})),
    (1201, [True]))
chk("manifest: and no W2 entry sits above the marker",
    [l for l in _mpub if "20260830_220554" in l], [])
chk("manifest: the two halves are the whole file",
    len(_mpub) + len(_mpend), len(_mlines))
chk("evidence.yml splits on that marker rather than reading the whole file",
    ("PENDING TRANCHE" in _EVY and "/tmp/pending" in _EVY
     and "sed '/^# --- PENDING TRANCHE/,$d'" in _EVY), True)
_v4r = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "README.md"
chk("the archive hash is recorded in both places",
    all("29c2401f100390268bbd52e43b5c2da9a61440bad3dabe502ca1684478771fd6" in t
        for t in (_man.read_text(encoding="utf-8"), _v4r.read_text(encoding="utf-8"))), True)
chk("and both record that it is published, and where",
    all("raw-evidence-2026-08-27" in t for t in
        (_man.read_text(encoding="utf-8"), _v4r.read_text(encoding="utf-8"))), True)
# What the release buys is a re-runnable extraction, so the counts it
# reproduces are a claim like any other and are checked against the dumps.
_acc_rows = json.loads((pathlib.Path(__file__).resolve().parents[1]
                        / "v4_audit_2026_08_25" / "data"
                        / "acceptance_counter_comparison.json").read_text(encoding="utf-8"))
_unreproducible = {"matrix_G_dflash_20260826_000124",
                   "matrix_I_conc1_20260826_012917",
                   "matrix_J_dflash_fit_20260826_014308"}
chk("the rows the archive alone cannot regenerate",
    sum(1 for r in _acc_rows if r["run"] in _unreproducible), 9)
chk("and the rest, which the workflow re-derives",
    sum(1 for r in _acc_rows if r["run"] not in _unreproducible), 526)
chk("the three runs are named where the count is",
    all(r in _v4r.read_text(encoding="utf-8") for r in _unreproducible), True)
chk("and none of them is a committed run directory",
    sorted(r for r in _unreproducible
           if (pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
               / "data" / r).is_dir()), [])
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


print("\n=== run V2: the crossover that identifies what run V could not ===")
# Run V measured `ignore_eos` as two whole runs sixteen minutes apart, and A16
# finds a DFlash-specific invocation effect of the same size. Run V2 is eight
# sessions of two halves in AB BA BA AB BA AB AB BA order, so the mode is
# balanced against the order and the session is the resampling unit.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import length_mode as _lm


_ER_V3 = _ER_V2 = re.sub(r"\s+", " ", _norm(
    (pathlib.Path(__file__).resolve().parents[1] / "ERRATA.md")
    .read_text(encoding="utf-8")))


def _cell(x):
    """One numeric table cell. `_pnum2` is defined further down the file."""
    t = _norm(x).replace("%", "").replace("pp", "").replace("*", "").replace("`", "")
    for _spx in (" ", "\u00a0", "\u2009", "\u202f"):
        t = t.replace(_spx, "")
    return float(t)

_V2 = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
              / "data").glob("matrix_V2_s*"))
chk("V2 halves on disk", len(_V2), 16)
chk("V2 arm-runs", sum(len(list(d.glob("*__rep*.json"))) for d in _V2), 400)
chk("every V2 half validated", sorted(d.name for d in _V2
                                      if not (d / "RUN_COMPLETE.json").exists()), [])
_v2_first = {}
_v2_sess = {}
for _d in _V2:
    _m = json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
    _s = _d.name.split("_")[2]
    _v2_sess.setdefault(_s, {})[("hardcap" if _m.get("ignore_eos") else "freerun")] = _d
    _v2_first.setdefault(_s, []).append(
        (_m["created"], "hardcap" if _m.get("ignore_eos") else "freerun"))
chk("V2 sessions", sorted(_v2_sess), [f"s{i}" for i in range(1, 9)])
chk("every session has both modes",
    sorted(s for s, h in _v2_sess.items() if set(h) != {"freerun", "hardcap"}), [])
# the order really is balanced, read from the manifests' own timestamps
_order = {s: sorted(v)[0][1] for s, v in _v2_first.items()}
chk("each mode ran first four times",
    sorted(Counter(_order.values()).items()),
    [("freerun", 4), ("hardcap", 4)])
_pos = {m: [i + 1 for i, s in enumerate(sorted(_order)) if _order[s] == m]
        for m in ("freerun", "hardcap")}
chk("and the two orders share a mean time position",
    [round(sum(v) / len(v), 2) for v in (_pos["freerun"], _pos["hardcap"])], [4.5, 4.5])
# every half must have run at the fit target that keeps the DFlash arms alive
chk("every V2 half ran at --fit-target 3072",
    sorted({json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("fit_target")
            for d in _V2}), ["3072"])
chk("every V2 half ran thinking off",
    sorted({json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("think")
            for d in _V2}), ["off"])
chk("and on a position-balanced schedule",
    sorted({json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            .get("schedule_is_position_balanced") for d in _V2}), [True])

# --- the numbers A17 now publishes -----------------------------------------
_v2_shift = defaultdict(list)
_v2_free = defaultdict(list)
_v2_cap = defaultdict(list)
_v2_base = {"freerun": [], "hardcap": []}
for _s, _h in sorted(_v2_sess.items()):
    _fr = {a: _lm.pooled(str(_h["freerun"]), a)["tok_s"]
           for a in _lm.arms_of(str(_h["freerun"]))}
    _cp = {a: _lm.pooled(str(_h["hardcap"]), a)["tok_s"]
           for a in _lm.arms_of(str(_h["hardcap"]))}
    _v2_base["freerun"].append(_fr["baseline"])
    _v2_base["hardcap"].append(_cp["baseline"])
    for _a, _v in _lm.contrast(_fr, _cp).items():
        _v2_shift[_a].append(_v["shift_pp"])
        _v2_free[_a].append(_v["free_pct"])
        _v2_cap[_a].append(_v["cap_pct"])
chk("V2 arms contrasted", sorted(_v2_shift),
    ["spec-dflash-n2", "spec-dflash-n4", "spec-draft-n8", "spec-mtp-n2"])
_A17T = {r[0]: r[1:] for r in _md_table(
    "| arm | freerun | hard cap | shift, 95 % t over 8 sessions |")}
chk("A17's crossover table rows", len(_A17T), 4)
def _iv2(cell):
    """`**-1.66 %** [-1.98, -1.35]` -> (point, low, high).

    `.split("[")[0]` reads the point estimate and stops. Every interval bound
    in this table, and in the three copies of it elsewhere, was unread until
    2026-08-29: twenty-four numbers in this one, found by perturbing each in
    turn rather than one per table."""
    _head, _, _rest = _norm(cell).partition("[")
    _pt = _cell(_head)
    if not _rest:
        return (_pt, None, None)
    _lo, _hi = (_cell(_x) for _x in _rest.rstrip("] ").split(","))
    return (_pt, _lo, _hi)


for _a, _row in sorted(_A17T.items()):
    _f = _lm.interval(_v2_free[_a]); _c = _lm.interval(_v2_cap[_a])
    _sh = _lm.interval(_v2_shift[_a])
    for _k, (_label, _iv) in enumerate((("freerun (%)", _f), ("hard cap (%)", _c),
                                        ("shift (pp)", _sh))):
        _pt, _lo, _hi = _iv2(_row[_k])
        chk(f"V2 {_a} {_label}", round(_iv[0], 2), _pt, 0.005)
        chk(f"V2 {_a} {_label} interval",
            (round(_iv[1], 2), round(_iv[2], 2)), (_lo, _hi))
    chk(f"V2 {_a} the hard cap moved it", _sh[1] > 0, True)
# the sign flip, which is what A17 was written about
chk("V2 spec-dflash-n4 is negative free-running", round(_lm.interval(_v2_free["spec-dflash-n4"])[2], 2) < 0, True)
chk("V2 spec-dflash-n4 is positive under the cap", round(_lm.interval(_v2_cap["spec-dflash-n4"])[1], 2) > 0, True)
# run V's single session against the eight
_RUNV = {"spec-dflash-n2": 9.26, "spec-mtp-n2": 9.68,
         "spec-dflash-n4": 11.90, "spec-draft-n8": 6.31}
_outside = sorted(a for a, v in _RUNV.items()
                  if not (_lm.interval(_v2_shift[a])[1] <= v <= _lm.interval(_v2_shift[a])[2]))
chk("run V's shift lies outside the eight-session interval for exactly this arm",
    _outside, ["spec-dflash-n2"])
chk("and above every one of the eight sessions",
    round(max(_v2_shift["spec-dflash-n2"]), 2) < 9.26, True)
chk("A17 says which arm and by how much", "about 3.3 pp on one arm" in _ER_V2, True)
# the between-session spread is arm-specific, which is A16 from another design
_sd = {a: st.stdev(v) for a, v in _v2_shift.items()}
chk("V2 between-session SD, spec-draft-n8 (pp)", round(_sd["spec-draft-n8"], 2), 0.02, 0.005)
chk("V2 between-session SD, spec-dflash-n2 (pp)", round(_sd["spec-dflash-n2"], 2), 1.27, 0.005)
chk("the A16 arm is the least stable contrast",
    max(_sd, key=lambda k: _sd[k]), "spec-dflash-n2")
chk("by at least a factor of two over the next",
    _sd["spec-dflash-n2"] > 2 * sorted(_sd.values())[-2], True)
chk("while the baseline holds under 0.25 % either way",
    [round(100 * st.stdev(v) / st.mean(v), 2) for v in
     (_v2_base["freerun"], _v2_base["hardcap"])], [0.21, 0.11])
chk("A17 records that no telemetry was captured for V2",
    "No GPU telemetry was recorded for run V2" in _ER_V2, True)

# The control that makes the spread interpretable: identical work, eight times.
_v2_sig = defaultdict(lambda: defaultdict(set))
_v2_work = defaultdict(lambda: defaultdict(set))
_v2_rows = 0
for _d in _V2:
    _mode = ("hardcap" if json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
             .get("ignore_eos") else "freerun")
    for _arm in sorted({f.name.split("__rep")[0] for f in _d.glob("*__rep*.json")}):
        _out, _dn, _da = [], 0, 0
        for _f in sorted(_d.glob(f"{_arm}__rep*.json")):
            for _r in json.loads(_f.read_text(encoding="utf-8"))["rows"]:
                _v2_rows += 1
                _out.append((_r.get("tag"), _r.get("content") or "",
                             _r["timings"]["predicted_n"]))
                _dn += _r.get("draft_n", 0) or 0
                _da += _r.get("draft_n_accepted", 0) or 0
        _v2_sig[_mode][_arm].add(
            hashlib.sha256(json.dumps(_out, sort_keys=True).encode()).hexdigest())
        _v2_work[_mode][_arm].add((_dn, _da))
chk("V2 request rows", _v2_rows, 4000)
chk("every arm produced one set of text in every session",
    sorted(f"{m}/{a}" for m in _v2_sig for a, v in _v2_sig[m].items() if len(v) != 1), [])
chk("and one drafted/accepted pair",
    sorted(f"{m}/{a}" for m in _v2_work for a, v in _v2_work[m].items() if len(v) != 1), [])
chk("the hard-cap halves generated exactly the cap every time",
    sorted({sum(_r["timings"]["predicted_n"]
                for _f in _d.glob("baseline__rep*.json")
                for _r in json.loads(_f.read_text(encoding="utf-8"))["rows"])
            for _d in _V2
            if json.loads((_d / "manifest.json").read_text(encoding="utf-8")).get("ignore_eos")}),
    [15000])
chk("A17 says the work was identical", "Identical to the token, in both modes" in _ER_V2, True)

print("\n=== run V3: both modes inside one square, and the arm that disagrees ===")
_V3 = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
              / "data").glob("matrix_V3_s*"))
chk("V3 sessions", len(_V3), 2)
chk("V3 arm-runs", sum(len(list(d.glob("*__rep*.json"))) for d in _V3), 200)
chk("both V3 sessions validated",
    sorted(d.name for d in _V3 if not (d / "RUN_COMPLETE.json").exists()), [])
for _d in _V3:
    _m = json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
    chk(f"{_d.name}: ten arms in a balanced square",
        (len(_m["arms"]), _m["repeats"], _m["schedule_is_position_balanced"]),
        (10, 10, True))
    chk(f"{_d.name}: five of them capped", len(_m["hardcap_arms"]), 5)
    chk(f"{_d.name}: the cap is per arm, not run-level", _m["ignore_eos"], False)
    chk(f"{_d.name}: a capped arm runs its base arm's flags",
        all(_m["arms"][a] == _m["arms"][a[:-4]] for a in _m["hardcap_arms"]), True)
    chk(f"{_d.name}: at the fit target that keeps DFlash alive",
        _m.get("fit_target"), "3072")

_v3_shift = defaultdict(list)
_v3_rate = defaultdict(dict)
for _d in _V3:
    _free = {a: _lm.pooled(str(_d), a)["tok_s"]
             for a in _lm.arms_of(str(_d)) if not a.endswith("-cap")}
    _cap = {a[:-4]: _lm.pooled(str(_d), a)["tok_s"]
            for a in _lm.arms_of(str(_d)) if a.endswith("-cap")}
    for _a, _v in _lm.contrast(_free, _cap).items():
        _v3_shift[_a].append(_v["shift_pp"])
    for _a in _free:
        _v3_rate[_a].setdefault("free", []).append(_free[_a])
        _v3_rate[_a].setdefault("cap", []).append(_cap[_a])
_V3T = {r[0]: r[1:] for r in _md_table(
    "| arm | V3 session 1 | V3 session 2 | V3 mean | V2, eight sessions |")}
chk("A17's V3 table rows", len(_V3T), 4)
# its last column is V2 again, and it was unread in both documents
for _a, _row in sorted(_V3T.items()):
    _pt, _lo, _hi = _iv2(_row[3])
    _v2i = _lm.interval(_v2_shift[_a])
    chk(f"V3 table {_a}: the V2 column is V2's own shift", round(_v2i[0], 2), _pt, 0.005)
    chk(f"V3 table {_a}: and V2's own interval",
        (round(_v2i[1], 2), round(_v2i[2], 2)), (_lo, _hi))
for _a, _row in sorted(_V3T.items()):
    chk(f"V3 {_a} session 1 (pp)", round(_v3_shift[_a][0], 2), _cell(_row[0]), 0.005)
    chk(f"V3 {_a} session 2 (pp)", round(_v3_shift[_a][1], 2), _cell(_row[1]), 0.005)
    chk(f"V3 {_a} mean (pp)", round(st.mean(_v3_shift[_a]), 2), _cell(_row[2]), 0.005)
    chk(f"V3 {_a} the cap moved it", min(_v3_shift[_a]) > 0, True)
# three agree with the crossover, one does not
_agree = sorted(a for a in _v3_shift
                if abs(st.mean(_v3_shift[a]) - st.mean(_v2_shift[a])) < 0.2)
chk("the arms on which the two designs agree to a fifth of a point",
    _agree, ["spec-dflash-n4", "spec-draft-n8", "spec-mtp-n2"])
chk("and the one they do not",
    sorted(set(_v3_shift) - set(_agree)), ["spec-dflash-n2"])
chk("V3's two sessions agree with each other on that arm to 0.06 pp",
    round(abs(_v3_shift["spec-dflash-n2"][0] - _v3_shift["spec-dflash-n2"][1]), 2),
    0.06, 0.005)
# The range was the honest answer while two designs disagreed and neither could
# attribute it. Run W attributes it, so A17 now states the within-invocation
# figure and names what the crossover's lower one is measuring instead.
chk("A17 no longer reports that arm as a bare range",
    "design-dependent between +5.9 and" in _ER_V3, False)
chk("A17 states the within-invocation figure",
    "+8.29 pp [+7.97, +8.60] when the two modes are" in " ".join(_ER_V3.split()), True)
chk("and says what the crossover's number is instead",
    "the two modes sit in different invocations" in " ".join(_ER_V3.split()), True)
chk("A17 reports the predecessor result as a null at this power, not as absence",
    "no detectable predecessor effect at" in " ".join(_ER_V3.split()), True)
chk("A17 rules out carryover as the explanation for the gap",
    "not first-order carryover" in " ".join(_ER_V3.split()), True)

# The per-repeat table under it: ten rates an arm, four arms, and the CV that
# each row's spread comes to. Forty-four numbers, none of which anything read,
# and it is the table the whole within-invocation argument rests on.
def _per_repeat(run_dir, arm):
    """One pooled rate per repeat file, in repeat order."""
    _out = []
    for _f in sorted(glob.glob(os.path.join(run_dir, f"{arm}__rep*.json"))):
        _b = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
        if _b.get("crashed"):
            continue
        _ms = sum(_r["timings"]["predicted_ms"] for _r in _b["rows"])
        _n = sum(_r["timings"]["predicted_n"] for _r in _b["rows"])
        _out.append(1000.0 * _n / _ms)
    return _out


_PRT = {r[0]: r[1:] for r in _md_table(
    "| arm, V3 session 1 | per-repeat pooled tok/s | CV |")}
chk("A17 per-repeat table: four rows", len(_PRT), 4)
_ARM_OF = {"no speculation": "baseline", "spec-mtp-n2": "spec-mtp-n2",
           "spec-dflash-n2": "spec-dflash-n2",
           "spec-dflash-n2-cap": "spec-dflash-n2-cap"}
chk("A17 per-repeat table: the rows it names", sorted(_PRT), sorted(_ARM_OF))
for _label, _row in sorted(_PRT.items()):
    _arm = _ARM_OF[_label]
    _pub = [float(_x) for _x in _row[0].replace("*", "").split()]
    _got = _per_repeat(str(_V3[0]), _arm)
    chk(f"A17 per-repeat {_label}: ten repeats", len(_pub), 10)
    chk(f"A17 per-repeat {_label}: every rate",
        [round(_x, 1) for _x in _got], _pub, 0.005)
    chk(f"A17 per-repeat {_label}: the CV of them",
        round(st.stdev(_got) / st.mean(_got) * 100, 2),
        _cell(_row[1]), 0.005)
# Session 2's CVs, quoted as prose under the table. It listed three, and the
# table above it has four rows, so a reader lining them up read the second
# arm's figure as the third's: the 0.33 % is `baseline`, not `spec-mtp-n2`,
# whose own session 2 CV is 0.49 %. All four are quoted now, in row order.
_s2_cv = [round(st.stdev(_v) / st.mean(_v) * 100, 2)
          for _a in ("baseline", "spec-mtp-n2", "spec-dflash-n2",
                     "spec-dflash-n2-cap")
          for _v in [_per_repeat(str(_V3[1]), _a)]]
chk("A17: session 2 repeats it, one figure per row of the table above",
    f"Session 2 repeats it, row for row: {_s2_cv[0]:.2f} %, {_s2_cv[1]:.2f} %, "
    f"{_s2_cv[2]:.2f} % and {_s2_cv[3]:.2f} %" in " ".join(_ER_V3.split()), True)
chk("A17: and there are as many figures as rows", (len(_s2_cv), len(_PRT)), (4, 4))
# and the sentence that reads the table: the spread it names, and the arm it
# contrasts against. The 6 is `spec-dflash-n2`, whose own row spans 6.2 tok/s.
_dn2 = _per_repeat(str(_V3[0]), "spec-dflash-n2")
chk("A17: the within-invocation spread it quotes",
    round(max(_dn2) - min(_dn2)), 6)
chk("A17: and it is stated as a spread between arm-runs of one configuration",
    "consecutive arm-runs of the same configuration inside one invocation "
    "differ by 6 tok/s" in " ".join(_ER_V3.split()).replace("**", ""), True)
_bl = _per_repeat(str(_V3[0]), "baseline")
chk("A17: the interleaved no-speculation arm holds 0.3 %",
    round(st.stdev(_bl) / st.mean(_bl) * 100, 1), 0.3, 0.005)
# The absolute-rate table was computed and compared against `_moved` below,
# and the TABLE itself was never read - the defect this repository has now
# found in six places. Probing it on 2026-08-28 by adding 7 to one cell
# produced no failure at all.
# V2's absolute rates were never computed here, only its percentages, so the
# table's first and third columns had nothing to compare against even in
# principle. Pool each arm over the eight sessions of each mode.
_v2_rate = defaultdict(lambda: defaultdict(list))
for _s, _h in sorted(_v2_sess.items()):
    for _mode, _key in (("freerun", "free"), ("hardcap", "cap")):
        _d = _h[_mode]
        for _a in _lm.arms_of(str(_d)):
            _pv = _lm.pooled(str(_d), _a)
            if _pv:
                _v2_rate[_a][_key].append(_pv["tok_s"])

_RT = {r[0]: r[1:] for r in _md_table(
    "| arm | V2 freerun | V3 freerun | V2 hard cap | V3 hard cap |")}
chk("A17's absolute-rate table rows", len(_RT), 5)
chk("and it names every arm the data has, with the baseline renamed",
    sorted(_RT), sorted(["no speculation"] + [a for a in _v3_rate if a != "baseline"]))
for _a, _row in sorted(_RT.items()):
    _key = "baseline" if _a == "no speculation" else _a
    chk(f"A17 rate table: {_a} V2 freerun",
        round(st.mean(_v2_rate[_key]["free"]), 2), _cell(_row[0]), 0.005)
    chk(f"A17 rate table: {_a} V3 freerun",
        round(st.mean(_v3_rate[_key]["free"]), 2), _cell(_row[1]), 0.005)
    chk(f"A17 rate table: {_a} V2 hard cap",
        round(st.mean(_v2_rate[_key]["cap"]), 2), _cell(_row[2]), 0.005)
    chk(f"A17 rate table: {_a} V3 hard cap",
        round(st.mean(_v3_rate[_key]["cap"]), 2), _cell(_row[3]), 0.005)

# only that arm's absolute rate moves between the designs
_moved = sorted(a for a in _v3_rate
                if abs(st.mean(_v3_rate[a]["free"]) / st.mean(
                    [_lm.pooled(str(d), a)["tok_s"] for d in _V2
                     if not json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                     .get("ignore_eos")]) - 1) > 0.005)
chk("the only arm whose free-running rate moves by more than half a percent",
    _moved, ["spec-dflash-n2"])
# and every V3 arm produced one output set across the two sessions
_v3_sig = defaultdict(set)
for _d in _V3:
    for _arm in sorted({f.name.split("__rep")[0] for f in _d.glob("*__rep*.json")}):
        _out = []
        for _f in sorted(_d.glob(f"{_arm}__rep*.json")):
            for _r in json.loads(_f.read_text(encoding="utf-8"))["rows"]:
                _out.append((_r.get("tag"), _r.get("content") or "",
                             _r["timings"]["predicted_n"]))
        _v3_sig[_arm].add(hashlib.sha256(
            json.dumps(_out, sort_keys=True).encode()).hexdigest())
chk("every V3 arm produced one output set across both sessions",
    sorted(a for a, v in _v3_sig.items() if len(v) != 1), [])
chk("all ten of them", len(_v3_sig), 10)
# the state is arm-run level: per-repeat CV inside one invocation
def _cv(d, arm):
    vs = []
    for f in sorted(d.glob(f"{arm}__rep*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
        vs.append(1000 * sum(r["timings"]["predicted_n"] for r in rows)
                  / sum(r["timings"]["predicted_ms"] for r in rows))
    return 100 * st.stdev(vs) / st.mean(vs)
chk("V3 s1 no-speculation CV inside the invocation (%)",
    round(_cv(_V3[0], "baseline"), 2), 0.27, 0.005)
chk("V3 s1 spec-mtp-n2 CV (%)", round(_cv(_V3[0], "spec-mtp-n2"), 2), 0.53, 0.005)
chk("V3 s1 spec-dflash-n2 CV (%)", round(_cv(_V3[0], "spec-dflash-n2"), 2), 1.82, 0.005)
chk("V3 s1 spec-dflash-n2-cap CV (%)",
    round(_cv(_V3[0], "spec-dflash-n2-cap"), 2), 2.31, 0.005)
chk("the unstable arm is at least three times the next inside one invocation",
    _cv(_V3[0], "spec-dflash-n2") > 3 * _cv(_V3[0], "spec-mtp-n2"), True)
chk("A17 says the state is finer-grained than A16 could see",
    "consecutive arm-runs of the same" in _ER_V3, True)
chk("A17 states that neither design varies the predecessor",
    "balance *position* and not *predecessor*" in _ER_V3, True)

print("\n=== run T4: the checkpoint boundary, split and measured ===")
# The third review's P0-2: the timers surround calls that begin with
# ctx->synchronize(), so 39.07 s might be an attribution to the API boundary
# rather than to the copying. T4 drains explicitly and times the drain.
_T4 = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
              / "data").glob("matrix_T4_split_*"))
chk("T4 run directories", len(_T4), 1)
_t4 = _T4[0]
chk("T4 arm-runs", len(list(_t4.glob("*__rep*.json"))), 18)
chk("T4 validated", (_t4 / "RUN_COMPLETE.json").exists(), True)
_t4m = json.loads((_t4 / "manifest.json").read_text(encoding="utf-8"))
chk("T4 arms", sorted(_t4m["arms"]), ["baseline", "spec-dflash-n2", "spec-draft-n8"])
chk("T4 repeats, balanced where 4 could not be",
    (_t4m["repeats"], _t4m["schedule_is_position_balanced"]), (6, True))
chk("T4 ran thinking on at the audited fit target",
    (_t4m["think"], _t4m["fit_target"]), ("on", "3072"))
chk("T4 ran on an instrumented library, not the stock one",
    _t4m["server_lib_sha256"]["libllama-server-impl.so"][:8], "0ff03b30")
chk("the split patch is archived",
    (pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "patches"
     / "checkpoint_timers_split.patch").is_file(), True)

# decode seconds, from the arm-run JSON
def _decode_s(arm):
    out = []
    for f in sorted(_t4.glob(f"{arm}__rep*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
        out.append(sum(r["timings"]["predicted_ms"] for r in rows) / 1000.0)
    return out
_t4_base = st.mean(_decode_s("baseline"))
_t4_draft = st.mean(_decode_s("spec-draft-n8"))
_t4_excess = _t4_draft - _t4_base
chk("T4 excess over no speculation (s)", round(_t4_excess, 1), 71.5, 0.05)
chk("and it reproduces A12's run T to under a fifth of a second",
    abs(_t4_excess - 71.4) < 0.2, True)

# The split itself, re-derived from the instrumented logs. Until 2026-08-27 the
# three figures below were compared with literals and the document was the only
# place they existed; `analysis/rederive_from_logs.py` now regenerates the dump
# from run T4's 18 server logs and this reads the dump.
_T4J = json.loads((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                   / "data" / "checkpoint_timers_20260827_split.json")
                  .read_text(encoding="utf-8"))
chk("T4's split dump covers all eighteen arm-runs", len(_T4J), 18)
_t4d = [r for r in _T4J if r["arm"] == "spec-draft-n8"]
chk("six of them are the arm that checkpoints", len(_t4d), 6)
chk("and the other twelve record no checkpoint events at all",
    sorted({(r["creates"], r["restores"]) for r in _T4J if r["arm"] != "spec-draft-n8"}),
    [(0, 0)])
chk("785 creates and 728 restores in every one of the six",
    sorted({(r["creates"], r["restores"]) for r in _t4d}), [(785, 728)])
_t4_call = st.mean(r["checkpoint_total_s"] for r in _t4d)
_t4_sync = st.mean(r["sync_total_s"] for r in _t4d)
_t4_state = st.mean(r["state_total_s"] for r in _t4d)
chk("the three parts add up", round(_t4_sync + _t4_state, 3), round(_t4_call, 3), 0.0015)

_A12T = {r[0]: r[1:] for r in _md_table("| | seconds | share of the 71.49 s excess |", quoted=True)}
chk("A12's split table rows", len(_A12T), 3)
chk("A12 quotes the whole call",
    round(_t4_call, 2), _cell(_A12T["inside the checkpoint calls"][0]), 0.005)
chk("A12 quotes the synchronisation wait",
    round(_t4_sync, 3), _cell(_A12T["of which, waiting on synchronize()"][0]), 0.0005)
chk("A12 quotes the state work",
    round(_t4_state, 2), _cell(_A12T["of which, state work"][0]), 0.005)
chk("the wait is under a hundredth of a percent of the excess",
    _cell(_A12T["of which, waiting on synchronize()"][1]) < 0.01, True)
chk("and the wait's own share is what A12 published",
    round(100 * _t4_sync / _t4_excess, 3),
    _cell(_A12T["of which, waiting on synchronize()"][1]), 0.0005)
chk("and the share is the one A12 published",
    round(100 * _t4_state / _t4_excess, 1), _cell(_A12T["of which, state work"][1]), 0.05)
chk("and so is the whole call's, which was compared against a literal only",
    round(100 * _t4_call / _t4_excess, 1),
    _cell(_A12T["inside the checkpoint calls"][1]), 0.05)
chk("which reproduces run T's 54.7 % on a different build",
    round(100 * _t4_call / _t4_excess, 1), 54.7, 0.05)
# the components A12 says reproduce, against run T's own dump
_T26 = {(r["arm"], r["repeat"]): r for r in json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "data"
     / "checkpoint_timers_20260826.json").read_text(encoding="utf-8"))}
_T26d = [v for k, v in _T26.items() if k[0] == "spec-draft-n8"]
for _f, _tol in (("update_tgt_s", 0.05), ("load_tgt_s", 0.05), ("load_dft_s", 0.05)):
    chk(f"T4 reproduces run T's {_f} to {_tol} s",
        abs(st.mean(r[_f] for r in _t4d) - st.mean(r[_f] for r in _T26d)) < _tol, True)
chk("A12 says the boundary question is answered by measurement",
    "answered by measurement rather than by wording" in _ER_V3, True)
chk("A12 records that the tree went back to stock",
    "a0cbe4d0" in _ER_V3 and "0ff03b30" in _ER_V3, True)

# A16's step, and the two things T4 could test
_A16T = {r[0]: r[1:] for r in _md_table("| arm | per-repeat pooled tok/s | CV |")}
chk("A16's T4 table rows", len(_A16T), 3)
def _rates(arm):
    out = []
    for f in sorted(_t4.glob(f"{arm}__rep*.json")):
        rows = json.loads(f.read_text(encoding="utf-8"))["rows"]
        out.append(1000 * sum(r["timings"]["predicted_n"] for r in rows)
                   / sum(r["timings"]["predicted_ms"] for r in rows))
    return out
for _label, _arm in (("no speculation", "baseline"),
                     ("spec-draft-n8", "spec-draft-n8"),
                     ("spec-dflash-n2", "spec-dflash-n2")):
    _v = _rates(_arm)
    _cv = 100 * st.stdev(_v) / st.mean(_v)
    chk(f"T4 {_arm} per-repeat values",
        [round(x, 2) for x in _v],
        [_cell(x) for x in _A16T[_label][0].split()])
    chk(f"T4 {_arm} CV (%)", round(_cv, 2), _cell(_A16T[_label][1]), 0.005)
_d = _rates("spec-dflash-n2")
chk("T4 the step: the last three are all above the first three",
    min(_d[3:]) > max(_d[:3]), True)
chk("and it is about four percent",
    round(100 * (st.mean(_d[3:]) / st.mean(_d[:3]) - 1), 1), 3.9, 0.05)
chk("while the arms beside it hold under one percent",
    max(100 * st.stdev(_rates(a)) / st.mean(_rates(a))
        for a in ("baseline", "spec-draft-n8")) < 1.0, True)
chk("A16 records that the predecessor does not explain it",
    "both predecessors produce both levels" in _ER_V3, True)
chk("and that the telemetry moves the wrong way",
    "moves the wrong way" in _ER_V3, True)

print("\n=== the same measurement, stored twice, must agree ===")
# Every arm-run row carries `predicted_ms`, `predicted_n` and
# `predicted_per_second` at the top level AND inside `timings`, and different
# analyses read different copies: paired_blocks.py, matrix_report.py, plot.py
# and plot_v4_runs.py take the top-level one, past_threshold_fit.py the nested
# one. Nothing compared them, so a change to one was invisible to whatever read
# the other - which is how a planted 5 % change to one request's decode time
# survived the whole perturbation suite.
_DUP = ("predicted_ms", "predicted_n", "predicted_per_second")
_armruns = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                   / "data").glob("*/*__rep*.json"))
_rows_seen = 0
_disagree = []
for _f in _armruns:
    for _r in json.loads(_f.read_text(encoding="utf-8")).get("rows") or []:
        _rows_seen += 1
        _t = _r.get("timings") or {}
        for _k in _DUP:
            if _k in _r and _k in _t and _r[_k] != _t[_k]:
                _disagree.append(f"{_f.parent.name}/{_f.name}: {_k} "
                                 f"{_r[_k]!r} at the top level, {_t[_k]!r} in timings")
chk("arm-run files scanned", len(_armruns) > 1200, True)
chk("request rows scanned", _rows_seen > 13000, True)
chk("the two copies of every measurement agree", _disagree[:3], [])

# B8: `predicted_per_second` is llama.cpp's own field and it reports a rate over
# n-1 tokens divided by the time for n. Every request-mean in this repository is
# the mean of it, so the relationship is pinned here rather than left to drift.
_nm1 = _nn = _neither = 0
for _f in _armruns:
    for _r in json.loads(_f.read_text(encoding="utf-8")).get("rows") or []:
        _t = _r.get("timings") or {}
        if not _t.get("predicted_ms") or "predicted_per_second" not in _t:
            continue
        _n, _ms, _ps = _t["predicted_n"], _t["predicted_ms"], _t["predicted_per_second"]
        if _n < 2:
            continue
        if abs(_ps - 1000 * (_n - 1) / _ms) < 1e-6 * max(1.0, 1000 * (_n - 1) / _ms):
            _nm1 += 1
        elif abs(_ps - 1000 * _n / _ms) < 1e-6 * max(1.0, 1000 * _n / _ms):
            _nn += 1
        else:
            _neither += 1
# What that numerator is WORTH, measured rather than described. The bias is
# 0.33 % at a fixed 300-token cap, which is why the pooled headline is
# unaffected; on the thinking-off freerun arms the lengths vary and it is 0.90
# to 1.75 %. Within a run it is nearly the same for every arm, so the ratios the
# study reports move far less than the absolute rates do -- and that is a
# measurement here, not an assumption.
# Selected by what the manifest says, not by what the directory is called.
# `glob("*thinkoff*")` matched three runs of twenty: it missed
# `D_master_matrix_think_off` for its underscore, every `matrix_V2_s*_freerun`
# half, `matrix_V_freerun`, and the freerun arms inside V3 and W. The published
# upper bound came from those three and was 1.57 %; the highest of the twenty
# is 1.75 %, in the run the underscore excluded.
_B8 = {}
for _d in sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                  / "data").glob("*")):
    if not _d.is_dir() or not (_d / "manifest.json").exists():
        continue
    _b8m = json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
    # thinking off, and not a wholly hard-capped half: the cap fixes every
    # length, which is the 0.33 % case this figure is contrasted against
    if str(_b8m.get("think")) != "off" or _b8m.get("ignore_eos"):
        continue
    _per = defaultdict(lambda: {"u": [], "c": []})
    for _f in _d.glob("*__rep*.json"):
        _b = json.loads(_f.read_text(encoding="utf-8"))
        _arm = _b.get("arm") or _f.name.split("__rep")[0]
        if _arm.endswith("-cap"):        # a capped arm inside a mixed run
            continue
        for _r in _b.get("rows") or []:
            _t = _r.get("timings") or {}
            if not _t.get("predicted_ms") or _t.get("predicted_n", 0) < 2:
                continue
            _per[_arm]["u"].append(_t["predicted_per_second"])
            _per[_arm]["c"].append(1000.0 * _t["predicted_n"] / _t["predicted_ms"])
    if _per:
        _B8[_d.name] = {_a: 100.0 * (st.mean(_v["c"]) / st.mean(_v["u"]) - 1.0)
                        for _a, _v in _per.items()}
_b8all = [x for v in _B8.values() for x in v.values()]
# Was `chk(..., sorted(_B8), sorted(_B8))`, which is true for any `_B8`
# including an empty one: it named a population and asserted nothing about it,
# and the self-audit below only refused two LITERALS, not one expression twice.
chk("B8: every thinking-off run with a freerun arm is in the population",
    (len(_B8), sum(len(_v) for _v in _B8.values())), (32, 158))
chk("B8: and the run the old filename glob dropped is one of them",
    "D_master_matrix_think_off" in _B8, True)
chk("B8: the numerator bias on thinking-off freerun arms",
    (round(min(_b8all), 2), round(max(_b8all), 2)), (0.90, 1.75))
# 0.26, not 0.12: the widest is `D_master_matrix_think_off`, the run the
# filename glob dropped, and it is nearly three times the widest of the three
# runs the figure used to be measured on.
_b8spread = {_k: round(max(_v.values()) - min(_v.values()), 2)
             for _k, _v in _B8.items() if len(_v) > 1}
chk("B8: within a run it is nearly the same for every arm, so ratios barely move",
    (max(_b8spread.values()), max(_b8spread, key=_b8spread.get)),
    (0.26, "D_master_matrix_think_off"))

chk("B8 rows where the server reports 1000*(n-1)/ms", _nm1, 30300)
chk("B8 rows where it reports 1000*n/ms", _nn, 44)
chk("B8 rows matching neither", _neither, 0)
chk("B8 the 44 are the legacy binary's",
    sorted({_f.parent.name for _f in _armruns
            if any(abs((_r.get("timings") or {}).get("predicted_per_second", 0)
                       - 1000 * (_r["timings"]["predicted_n"])
                       / _r["timings"]["predicted_ms"]) < 1e-9
                   for _r in json.loads(_f.read_text(encoding="utf-8")).get("rows") or []
                   if (_r.get("timings") or {}).get("predicted_ms")
                   and _r["timings"]["predicted_n"] > 1)}),
    ["A_bcb5eeb64_legacy"])
chk("ERRATA carries B8",
    "### B8. Every `request-mean` here counts one token fewer"
    in (pathlib.Path(__file__).resolve().parents[1] / "ERRATA.md")
    .read_text(encoding="utf-8"), True)

print("\n=== every published interval says what it covers ===")
# `paired_blocks.json` looked identical whether the schedule balanced or not,
# and it is the file the documents quote. Two of the seven came from schedules
# that are not position-balanced - run T rotates three arms over four repeats,
# and the head-to-head run has three blocks of a nine-arm design - and neither
# file said so. They record it now, and the tool refuses to write one without
# --allow-unbalanced.
_pbs = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
               / "data").glob("*/paired_blocks.json"))
chk("paired-block files on disk", len(_pbs), 7)
_pbj = {f.parent.name: json.loads(f.read_text(encoding="utf-8")) for f in _pbs}
chk("every one records whether its schedule balanced",
    sorted(k for k, v in _pbj.items() if "schedule_position_balanced" not in v), [])
chk("every one records the scope of its interval",
    sorted(k for k, v in _pbj.items()
           if "invocation" not in (v.get("interval_scope") or "")), [])
chk("the unbalanced ones are exactly these two, and they say so",
    sorted(k for k, v in _pbj.items() if not v["schedule_position_balanced"]),
    ["matrix_O_headtohead_20260826_081806", "matrix_T_timers_20260826_182639"])
chk("and each of those was written with the override, not silently",
    all(v["unbalanced_override"] for v in _pbj.values()
        if not v["schedule_position_balanced"]), True)
# the t critical value each file used, against the value for its block count
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paired_blocks as _pb
for _name, _v in sorted(_pbj.items()):
    chk(f"t critical for {_name} ({_v['blocks']} blocks)",
        round(_pb.t_critical_975(_v["blocks"] - 1), 4), _v["t_critical_975"], 0.0005)
# The tabulated fallback was wrong above df=10. No committed run reaches it, so
# no published interval moved when it was replaced - which is worth asserting
# rather than assuming.
chk("no committed run has enough blocks to have hit the 1.96 fallback",
    max(v["blocks"] for v in _pbj.values()) - 1 <= 10, True)

print("\n=== the past-threshold pre-registration ===")
# Until 2026-08-27 this was the only analysis in the repository with no
# committed code path: PREREGISTERED_PREDICTION.md was written from ad-hoc
# commands, and eight of its published figures were wrong — two baselines taken
# from repeat 0 alone, two traffic figures computed at the checkpoint size A12
# withdrew, a decode rate labelled end to end, an error column rounded twice, a
# checkpoint count attributed to one request instead of ten, and a scatter range
# nothing produces. The prediction section, written before the data existed,
# reproduced exactly, and so did the residual step and the amortisation
# deviation once their conventions were recovered. Everything the document
# prints comes from analysis/past_threshold_fit.py and is parsed back out of the
# document here.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import past_threshold_fit as _ptf

_PT = _ptf.build()
_PRE_P = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("v4_audit_2026_08_25", "PREREGISTERED_PREDICTION.md")
_PRE = _norm(_PRE_P.read_text(encoding="utf-8"))
_PRE_LINES = _PRE.splitlines()
# prose wraps, so sentences are matched against a whitespace-collapsed copy
_PRE_FLAT = re.sub(r"\s+", " ", _PRE)


def _pnum(x):
    """_f is rebound to a filename by an earlier loop; this is the same parse."""
    return float(_norm(x).replace("%", "").replace("pp", "").replace(" ", ""))


def _pre_table(header_startswith):
    i = next(i for i, l in enumerate(_PRE_LINES) if l.startswith(header_startswith))
    rows = []
    for l in _PRE_LINES[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip().strip("*`").replace("`", "").strip("* ").strip()
                     for c in l.strip("|").split("|")])
    return rows


# --- the model the prediction was made with ------------------------------
_f3 = _PT["fit3"]
chk("prereg fit: per-round coefficient", round(_f3["per_round_ms"], 2), 27.56, 0.005)
chk("prereg fit: per-draft-token coefficient", round(_f3["per_draft_token_ms"], 2), 5.54, 0.005)
chk("prereg fit: intercept", round(_f3["intercept_ms"], 2), 12.83, 0.005)
chk("prereg fit: R2", round(_f3["r2"], 4), 0.9956, 0.00005)
chk("prereg fit: draft volume alone", round(_f3["draft_volume_only_r2"], 4), 0.9471, 0.00005)
chk("prereg fit: leave-one-out MAE", round(_f3["loo_mae_ms"], 2), 1.63, 0.005)
chk("prereg fit: worst LOO point is the edge", _f3["loo_worst"][0], 32)
chk("prereg fit: worst LOO error", round(_f3["loo_worst"][1], 2), 4.99, 0.005)
chk("prereg fit: regressor correlation", round(_f3["regressor_r"], 3), -0.665, 0.0005)
chk("prereg fit: regressor VIF", round(_f3["regressor_vif"], 2), 1.79, 0.005)
for _s in ("27.56 * (rounds per generated token)", "5.54 * (draft tokens per generated token)",
           "+ 12.83", "R² = 0.9956", "R² = 0.9471", "1.63 ms/token",
           "r = -0.665 (VIF 1.79)", "`n_max` 32, +4.99"):
    chk(f"prereg document carries {_s!r}", _norm(_s) in _PRE_FLAT, True)
# the two figures the model is scaled against, and the one that was wrong
chk("prereg fit: intercept over the no-speculation cost (%)",
    round(_f3["intercept_over_baseline_pct"]), 58, 0.5)
chk("prereg: the no-speculation cost is the pooled one, not repeat 0",
    round(_PT["baseline"]["C_pooled_ms_per_token"], 2), 8.11, 0.005)
chk("prereg: repeat 0 alone is what 7.87 was",
    round(_PT["baseline"]["C_repeat0_ms_per_token"], 2), 7.87, 0.005)
chk("prereg: the per-round term removes this much of the remaining error (%)",
    round(_f3["error_removed_by_rounds_pct"]), 92, 0.5)
chk("prereg document says 92 %, not the truncated 91",
    "removing 92 % of the remaining error" in _PRE_FLAT, True)

# --- the table committed before the data existed -------------------------
_pred_rows = {int(r[0]): r[1:] for r in
              _pre_table("| `n_max` | predicted acceptance |")}
chk("prereg prediction table rows", sorted(_pred_rows), [64, 96, 128])
for _n, _row in _pred_rows.items():
    _p = _PT["predicted"][_n]
    chk(f"prereg n{_n} predicted acceptance", round(100 * _p["acceptance"], 1), _pnum(_row[0]), 0.05)
    chk(f"prereg n{_n} predicted draft/gen", round(_p["draft_per_gen"], 2), _pnum(_row[1]), 0.005)
    chk(f"prereg n{_n} predicted ms/token", round(_p["ms_per_token"], 2), _pnum(_row[2]), 0.005)
    chk(f"prereg n{_n} predicted tok/s", round(_p["tok_s"], 1), _pnum(_row[3]), 0.05)

# --- what the measurement did to it --------------------------------------
_out_rows = {int(r[0]): r[1:] for r in
             _pre_table("| `n_max` | registered tok/s |")}
chk("prereg outcome table rows", sorted(_out_rows), [64, 96, 128])
for _n, _row in _out_rows.items():
    _o = _PT["outcome"][_n]
    chk(f"prereg n{_n} registered rate is the one predicted in advance",
        _pnum(_row[0]), round(_PT["predicted"][_n]["tok_s"], 1), 0.05)
    chk(f"prereg n{_n} registered as published", _pnum(_row[0]), _o["registered_tok_s"], 0.005)
    chk(f"prereg n{_n} measured tok/s", round(_o["measured_tok_s"], 2), _pnum(_row[1]), 0.005)
    chk(f"prereg n{_n} error against the registered number",
        round(_o["error_pct"], 1), _pnum(_row[2]), 0.05)
    chk(f"prereg n{_n} the measurement is not above the registration",
        _o["error_pct"] < 0, True)
# the third row was published as an exact hit; it is 0.6 % low
chk("prereg n128 is not the exact hit it was published as",
    round(_PT["outcome"][128]["error_pct"], 1) != 0.0, True)
chk("prereg records what that column used to read",
    "-7.5 %, -5.7 % and 0.0 %" in _PRE_FLAT, True)
_over = [round(_PT["outcome"][n]["measured_input_over_prediction_pct"], 1)
         for n in (64, 96, 128)]
chk("prereg: measured inputs over-predict cost", _over, [12.3, 18.1, 25.2])
chk("prereg document carries the over-prediction",
    "+12.3 %, +18.1 % and +25.2 %" in _PRE_FLAT, True)

# --- the one-regressor law, and the absent knee ---------------------------
_law = _PT["law"]
chk("prereg law: slope", round(_law["slope_ms_per_draft_token"], 3), 4.040, 0.0005)
chk("prereg law: intercept", round(_law["intercept_ms"], 2), 27.00, 0.005)
chk("prereg law: R2", round(_law["r2"], 5), 0.99303, 0.000005)
chk("prereg law: slope is half a target step",
    round(_law["slope_over_baseline"], 2), 0.50, 0.005)
chk("prereg law: intercept over the target step",
    round(_law["intercept_over_baseline"], 1), 3.3, 0.05)
chk("prereg law: step at the coverage point (pp)", round(_law["step_pp"], 2), -0.39, 0.005)
chk("prereg law: mean residual below the threshold (%)",
    round(_law["mean_residual_below_pct"], 2), -0.27, 0.005)
chk("prereg law: mean residual at or above it (%)",
    round(_law["mean_residual_at_or_above_pct"], 2), -0.67, 0.005)
chk("prereg law: the step is dwarfed by the scatter",
    abs(_law["step_pp"]) < (_law["residual_arc_pct"][1] - _law["residual_arc_pct"][0]) / 50, True)
chk("prereg law: residual arc",
    [round(v, 1) for v in _law["residual_arc_pct"]], [-11.0, 10.9])
for _doc, _what in (("v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md", "the pre-registration"),
                    ("ERRATA.md", "A7"), ("README.md", "the README")):
    _t = _norm(pathlib.Path(__file__).resolve().parents[1].joinpath(_doc)
               .read_text(encoding="utf-8"))
    chk(f"{_what}: the residual step is -0.39 percentage points",
        "-0.39 percentage" in _t, True)
    chk(f"{_what}: 8.11 ms is the no-speculation step it quotes",
        "7.87 ms no-speculation" not in _t, True)

# --- the coverage arithmetic the whole test is about ----------------------
for _n, _want in ((1, 3.1), (32, 63.8), (64, 86.9), (96, 95.3), (128, 98.3)):
    chk(f"prereg coverage at n{_n} (%)", round(_PT["coverage_pct"][_n], 1), _want, 0.05)
chk("prereg: the threshold arm is the first at or past 95 %",
    _PT["coverage_pct"][_PT["threshold_nmax"]] >= 95.0
    and _PT["coverage_pct"][64] < 95.0, True)

# --- throughput: decode rate and wall clock, labelled apart ---------------
_dec = [round(_PT["arms"][n]["decode_tok_s"], 1) for n in (1, 2, 4, 8, 16, 32, 64, 96, 128)]
_wall = [round(_PT["arms"][n]["wall_tok_s"], 1) for n in (1, 2, 4, 8, 16, 32, 64, 96, 128)]
chk("prereg decode rates", _dec, [31.1, 34.2, 35.6, 32.1, 23.7, 17.3, 12.4, 10.0, 8.9])
chk("prereg wall-clock rates", _wall, [30.2, 33.2, 34.5, 31.1, 23.2, 17.0, 12.2, 9.9, 8.8])
chk("prereg: it peaks at n_max 4", max(range(len(_dec)), key=lambda i: _dec[i]), 2)
chk("prereg: and declines monotonically after",
    all(_dec[i] > _dec[i + 1] for i in range(2, len(_dec) - 1)), True)
chk("prereg baseline decode rate", round(_PT["baseline"]["C_decode_tok_s"], 1), 123.4, 0.05)
chk("prereg baseline wall-clock rate", round(_PT["baseline"]["C_wall_tok_s"], 1), 110.8, 0.05)
chk("prereg document labels the decode series as decode",
    "Pooled decode rate: 31.1" in _PRE_FLAT, True)
chk("prereg document publishes the wall-clock series too",
    "30.2, 33.2, **34.5**, 31.1, 23.2, 17.0, 12.2, 9.9, 8.8" in _PRE_FLAT, True)
chk("prereg document gives the wall-clock baseline", "**110.8** baseline" in _PRE_FLAT, True)

# --- hypothesis 1: marginal cost amortises -------------------------------
_am = _PT["amortisation"]
chk("prereg marginal cost at n1 (ms)",
    round(_am["marginal_ms_per_draft_token"][1], 1), 48.4, 0.05)
chk("prereg marginal cost at n128 (ms)",
    round(_am["marginal_ms_per_draft_token"][128], 1), 4.8, 0.05)
chk("prereg: it falls monotonically",
    all(_am["marginal_ms_per_draft_token"][a] > _am["marginal_ms_per_draft_token"][b]
        for a, b in zip((1, 2, 4, 8, 16, 32, 64, 96), (2, 4, 8, 16, 32, 64, 96, 128))), True)
chk("prereg: the fitted-six curve puts the three that came after this far below (%)",
    [round(_am["supra_deviation_pct"][n], 1) for n in (64, 96, 128)], [-24.5, -29.4, -34.1])
chk("prereg: fitting all nine shrinks the apparent mechanism",
    all(abs(_am["all_nine_deviation_pct"][n]) < abs(_am["supra_deviation_pct"][n])
        for n in (96, 128)), True)
chk("prereg document quotes the range and the three deviations",
    "24-34 % *below* the curve (-24.5 %, -29.4 %, -34.1 %)" in _PRE_FLAT, True)

# --- hypothesis 2: checkpoint traffic dominates --------------------------
_ck = _PT["checkpoints"]
chk("prereg: the checkpoint size is the one A12 left standing",
    _ck["checkpoint_mib"], 82.079)
chk("prereg: checkpoints in one n_max 1 arm-run", _ck["checkpoints_per_arm_run"][1], 1639)
chk("prereg: that arm-run is ten requests", _ck["requests_per_arm_run"], 10)
chk("prereg: and 3000 generated tokens", _ck["generated_per_arm_run"], 3000)
chk("prereg: checkpoints per request", round(_ck["checkpoints_per_request"], 1), 163.9, 0.05)
chk("prereg traffic at n1 (MiB per generated token)",
    round(_ck["mib_per_generated_token"][1], 1), 44.8, 0.05)
chk("prereg traffic at n128 (MiB per generated token)",
    round(_ck["mib_per_generated_token"][128], 1), 20.2, 0.05)
chk("prereg: traffic falls while cost rises",
    _ck["mib_per_generated_token"][1] > _ck["mib_per_generated_token"][128]
    and _PT["arms"][1]["ms_per_token"] < _PT["arms"][128]["ms_per_token"], True)
chk("prereg: the correlation that refutes it",
    round(_ck["correlation_with_cost"], 2), -0.52, 0.005)
# the withdrawn size is exactly what the old figures were computed at
chk("prereg: 55.4 and 24.9 were 101.345 MiB per checkpoint",
    [round(_ck["checkpoints_per_arm_run"][n] * 101.345 / 3000, 1) for n in (1, 128)],
    [55.4, 24.9])
chk("prereg document carries the corrected traffic", "44.8 MiB at `n_max` 1 to 20.2" in _PRE_FLAT, True)
chk("prereg document no longer attributes 1639 to one request",
    "for a single 300-token" not in _PRE_FLAT, True)

# --- the repeat-to-repeat scatter the outcome section quotes --------------
chk("prereg run-to-run SD (tok/s)",
    [round(_PT["repeat_sd_tok_s"][n], 2) for n in (64, 96, 128)], [0.05, 0.16, 0.11])
chk("prereg document quotes the SDs it measured",
    "0.05, 0.16 and 0.11 tok/s" in _PRE_FLAT, True)

# --- the tables A7 and A10 publish, parsed cell by cell -------------------
# A10 falsifies A7's law, and until 2026-08-27 its second half was fitted on
# repeat 0 - which it disclosed, giving a reason that had stopped being true
# when the counter dump was rebuilt over every repeat earlier the same week.
def _pnum2(x):
    """Parse one table cell. Markup and thousands separators both vary here."""
    t = _norm(x).replace("%", "").replace("pp", "").replace("*", "").replace("`", "")
    for _spc in (" ", "\u00a0", "\u2009", "\u202f"):
        t = t.replace(_spc, "")
    return float(t)


_sweep = {int(_pnum2(r[0])): r[1:] for r in
          _md_table("| n_max | pooled tok/s | vs baseline | drafted | acceptance |")}
chk("A7 sweep table rows", sorted(_sweep), [1, 2, 4, 8, 16, 32])
_base_c = _PT["baseline"]["C_decode_tok_s"]
for _n, _row in _sweep.items():
    _a = _PT["arms"][_n]
    chk(f"A7 sweep n{_n} pooled tok/s", round(_a["decode_tok_s"], 1), _pnum2(_row[0]), 0.05)
    chk(f"A7 sweep n{_n} vs baseline (%)",
        round((_a["decode_tok_s"] / _base_c - 1) * 100, 1), _pnum2(_row[1]), 0.05)
    chk(f"A7 sweep n{_n} drafted total",
        round(_a["draft_per_gen"] * _a["generated"]), int(_pnum2(_row[2])))
    chk(f"A7 sweep n{_n} acceptance (%)", round(100 * _a["acceptance"], 1), _pnum2(_row[3]), 0.05)

_covt = {int(_pnum2(r[0])): r[1:] for r in
         _md_table("| `n_max` | coverage | acceptance | draft/gen | pooled tok/s |")}
chk("A7 coverage table rows", sorted(_covt), [32, 64, 96, 128])
for _n, _row in _covt.items():
    _a = _PT["arms"][_n]
    chk(f"A7 coverage n{_n} (%)", round(_PT["coverage_pct"][_n], 1), _pnum2(_row[0]), 0.05)
    chk(f"A7 coverage n{_n} acceptance (%)", round(100 * _a["acceptance"], 1), _pnum2(_row[1]), 0.05)
    chk(f"A7 coverage n{_n} draft/gen", round(_a["draft_per_gen"], 2), _pnum2(_row[2]), 0.005)
    chk(f"A7 coverage n{_n} pooled tok/s", round(_a["decode_tok_s"], 1), _pnum2(_row[3]), 0.05)

_H = _PT["pmin"]
_ER_TXT = _norm(pathlib.Path(__file__).resolve().parents[1]
                  .joinpath("ERRATA.md").read_text(encoding="utf-8"))
_ER_FLAT = re.sub(r"\s+", " ", _ER_TXT)
_lbl = lambda x: re.sub(r"\s+", " ", x.replace("*", "").replace("`", "")).strip()
_A10 = {_lbl(r[0]): r[1:] for r in _md_table(
    "| arm | draft/gen | real acceptance | pooled tok/s | vs baseline | law residual |")}
chk("A10 table rows", len(_A10), 7)
chk("A10 baseline row", _pnum2(_A10["baseline"][2]),
    round(_H["baseline"]["decode_tok_s"], 1), 0.05)
_A10_KEY = {"n_max 8, p_min 0.75": "spec-draft-n8-pmin75",
            "n_max 8, p_min 0.90": "spec-draft-n8-pmin90",
            "n_max 128, p_min 0.75": "spec-draft-n128-pmin75",
            "n_max 32, p_min 0.75": "spec-draft-n32-pmin75",
            "n_max 8, p_min 0.50": "spec-draft-n8-pmin50",
            "n_max 8, p_min 0 (the whole audit matrix)": "spec-draft-n8"}
for _label, _arm in _A10_KEY.items():
    _row = _A10[_label]
    _a = _H["arms"][_arm]
    chk(f"A10 {_arm} draft/gen", round(_a["draft_per_gen"], 2), _pnum2(_row[0]), 0.005)
    chk(f"A10 {_arm} acceptance (%)", round(100 * _a["acceptance"], 1), _pnum2(_row[1]), 0.05)
    chk(f"A10 {_arm} pooled tok/s", round(_a["decode_tok_s"], 1), _pnum2(_row[2]), 0.05)
    chk(f"A10 {_arm} vs baseline (%)", round(_a["vs_baseline_pct"], 1), _pnum2(_row[3]), 0.05)
    chk(f"A10 {_arm} law residual (%)", round(_a["law_residual_pct"], 1), _pnum2(_row[4]), 0.05)
    chk(f"A10 {_arm} over-predicted by the law", _a["law_residual_pct"] < 0, True)
chk("A10 mean |residual| where p_min > 0 (%)",
    round(_H["mean_abs_residual_on_p_min_positive_pct"], 1), 19.4, 0.05)
chk("A10 quotes that mean", "mean of **19.4 %**" in _ER_FLAT, True)
_sd = [v for k, v in _H["repeat_sd_tok_s"].items()]
chk("A10 run-to-run SD over the speculative arms",
    [round(min(_sd), 2), round(max(_sd), 2)], [0.13, 0.36])
chk("A10 quotes the SD range it measured", "0.13 to 0.36 tok/s" in _ER_FLAT, True)
chk("A10 no-speculation scatter", round(_H["baseline_repeat_sd_tok_s"], 2), 3.16, 0.005)
chk("A10 says why the baseline scatters more", "3.16 tok/s on a mean" in _ER_FLAT, True)

_two = {_lbl(r[0]): r[1:] for r in _md_table("| configuration | draft/gen | rounds/gen | ms/token |")}
_TC = _H["two_configurations"]
for _label, _key in (("n_max 1, p_min 0", "n_max 1, p_min 0"),
                     ("n_max 8, p_min 0.90", "n_max 8, p_min 0.90")):
    _row = _two[_label]
    chk(f"A10 two-config {_key} draft/gen",
        round(_TC[_key]["draft_per_gen"], 2), _pnum2(_row[0]), 0.005)
    chk(f"A10 two-config {_key} rounds/gen",
        round(_TC[_key]["rounds_per_gen"], 2), _pnum2(_row[1]), 0.005)
    chk(f"A10 two-config {_key} ms/token",
        round(_TC[_key]["ms_per_token"], 2), _pnum2(_row[2]), 0.005)
chk("A10 two-config: volume differs (%)", round(_TC["volume_differs_pct"]), 9, 0.5)
chk("A10 two-config: cost differs (%)", round(_TC["cost_differs_pct"]), 37, 0.5)
chk("A10 two-config: rounds differ (%)", round(_TC["rounds_differ_pct"]), 186, 0.5)
chk("A10 quotes the corrected cost gap", "cost by **37 %**" in _ER_FLAT, True)

_refit = {_lbl(r[0]): r[1:] for r in _md_table(
    "| model | bias on `p_min > 0` arms | bias on `p_min = 0` arms | separation |")}
chk("A10 refit configurations", _H["refit"]["configurations"], 14)
for _label in ("volume only", "rounds + volume"):
    _m = _H["refit"]["models"][_label]
    _row = _refit[_label]
    chk(f"A10 refit {_label}: bias where p_min > 0 (%)",
        round(_m["bias_p_min_positive_pct"], 1), _pnum2(_row[0]), 0.05)
    chk(f"A10 refit {_label}: bias where p_min = 0 (%)",
        round(_m["bias_p_min_zero_pct"], 1), _pnum2(_row[1]), 0.05)
    chk(f"A10 refit {_label}: family separation (pp)",
        round(_m["separation_pp"], 1), _pnum2(_row[2]), 0.05)
chk("A10 refit: rounds shrink the separation but do not remove it",
    _H["refit"]["models"]["rounds + volume"]["separation_pp"]
    < _H["refit"]["models"]["volume only"]["separation_pp"] / 2 > 0, True)
_cut = (1 - _H["refit"]["models"]["rounds + volume"]["separation_pp"]
        / _H["refit"]["models"]["volume only"]["separation_pp"]) * 100
chk("A10 refit: how much of it rounds remove (%)", round(_cut), 56, 0.5)
chk("A10 quotes that", "cuts it by 56 %" in _ER_FLAT, True)
chk("A10 refit: volume alone still reaches this R2",
    round(_H["refit"]["models"]["volume only"]["r2"], 3), 0.988, 0.0005)
chk("A10 records that the rep-0 basis was lifted",
    "computed from repeat 0 alone until 2026-08-27" in _ER_FLAT, True)

# the same p_min table, in the README
_RMD_LINES = pathlib.Path(__file__).resolve().parents[1] \
    .joinpath("README.md").read_text(encoding="utf-8").splitlines()
_i = next(i for i, l in enumerate(_RMD_LINES)
          if l.startswith("| configuration | real acceptance | pooled tok/s | vs baseline |"))
_rp = {}
for _l in _RMD_LINES[_i + 2:]:
    if not _l.startswith("|"):
        break
    _c = [x.strip() for x in _norm(_l).strip("|").split("|")]
    _rp[_lbl(_c[0])] = _c[1:]
chk("README p_min table rows", len(_rp), 4)
chk("README p_min: the no-speculation row",
    _pnum2(_rp["no speculation"][1]), round(_H["baseline"]["decode_tok_s"], 1), 0.05)
for _label, _arm in (("n_max 8, p_min 0.75", "spec-draft-n8-pmin75"),
                     ("n_max 8, p_min 0.90", "spec-draft-n8-pmin90"),
                     ("n_max 8, p_min 0", "spec-draft-n8")):
    _a = _H["arms"][_arm]
    _row = _rp[_label]
    chk(f"README p_min {_arm} acceptance (%)",
        round(100 * _a["acceptance"], 1), _pnum2(_row[0]), 0.05)
    chk(f"README p_min {_arm} pooled tok/s",
        round(_a["decode_tok_s"], 1), _pnum2(_row[1]), 0.05)
    chk(f"README p_min {_arm} vs baseline (%)",
        round(_a["vs_baseline_pct"], 1), _pnum2(_row[2]), 0.05)
chk("README and A10 agree on the p_min rows",
    [_pnum2(_rp[k][1]) for k in ("n_max 8, p_min 0.75", "n_max 8, p_min 0.90", "n_max 8, p_min 0")],
    [_pnum2(_A10[k][2]) for k in ("n_max 8, p_min 0.75", "n_max 8, p_min 0.90",
                                  "n_max 8, p_min 0 (the whole audit matrix)")])

# --- and the thing that makes it a pre-registration at all ---------------
if _HAS_GIT:
    def _git(*a):
        return _sp2.run(["git", "-C", str(_repo)] + list(a),
                        capture_output=True, text=True).stdout.strip()
    _first_pre = _git("log", "--reverse", "--format=%H", "--",
                      "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md").split("\n")[0]
    _first_data = _git("log", "--reverse", "--format=%H", "--diff-filter=A", "--",
                       "v4_audit_2026_08_25/data/E_past_threshold").split("\n")[0]
    chk("prereg: the prediction table exists in the first commit of the file",
        "74.76" in _git("show", f"{_first_pre}:v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md"), True)
    chk("prereg: the outcome section does not",
        "# Outcome" in _git("show", f"{_first_pre}:v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md"), False)
    chk("prereg: and that commit is an ancestor of the one that adds the data",
        _sp2.run(["git", "-C", str(_repo), "merge-base", "--is-ancestor",
                  _first_pre, _first_data]).returncode, 0)
else:
    print("  ----  no git history here; the pre-registration ordering check is "
          "skipped, not passed")

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
    # 2026-08-27, the pre-registration's outcome section
    "7.87 ms": "the no-speculation step from repeat 0 alone (it is 8.11 pooled)",
    "55.4 MiB": "checkpoint traffic at the withdrawn 101.3 MiB size",
    "+11.9 %": "the measured-input over-prediction (it is +12.3 %)",
    "SD 0.03-0.17": "the past-threshold scatter (it is 0.05, 0.16 and 0.11)",
    "single 300-token request": "1639 checkpoints are one arm-run of ten",
}
_ALLOWED = ("ERRATA.md",)          # the entries that retract them
for _needle, _what in _WITHDRAWN.items():
    _hits = []
    for _f in ("README.md", "CHANGELOG.md", "RETEST_TODO.md", "BENCHMARK_ENV.md",
               "v4_audit_2026_08_25/README.md", "ERRATA.md",
               "v4_audit_2026_08_25/PREREGISTERED_PREDICTION.md"):
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


_o2_acc = {}
for _a in {_f.split("/")[-1].split("__rep")[0] for _f in glob.glob(f"{_O2}/*__rep*.json")}:
    _dn = _da = 0
    for _f in glob.glob(f"{_O2}/{_a}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _da += _x["draft_n_accepted"]
    if _dn:
        _o2_acc[_a] = 100 * _da / _dn
_o2_win = [_a for _a in _o2_acc if _pooled_arm(_O2, _a) > _pooled_arm(_O2, "baseline")]
chk("README: spec-draft-n1 is above one winning arm and below two",
    (sum(1 for _a in _o2_win if _o2_acc[_a] < _o2_acc["spec-draft-n1"]),
     sum(1 for _a in _o2_win if _o2_acc[_a] > _o2_acc["spec-draft-n1"])), (1, 2))
chk("README: and it is third-highest acceptance in that table",
    sorted(_o2_acc, key=_o2_acc.get, reverse=True).index("spec-draft-n1") + 1, 3)
chk("README no longer claims it beats all but one winner",
    "more than every winning arm but one" in _ROOT_TEXT, False)

# --- the experiment registry has to name every run that exists --------------
# It said "runs A-L" and listed eight while the data held thirty-five labels,
# including run O2, which the headline table above it is built from. A registry
# that stops before the headline is worse than none: it reads as complete.
_REG_DIRS = {}
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/")):
    if not glob.glob(f"{_d}*__rep*.json"):
        continue
    _name = os.path.basename(_d.rstrip("/"))
    _m = re.match(r"matrix_([A-Za-z]+[0-9]*)", _name)
    _tag = _m.group(1) if _m else _name.split("_")[0]
    _REG_DIRS.setdefault(_tag, []).append(_name)
_REG_ALL = [_d for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/"))
            if glob.glob(f"{_d}*__rep*.json")]
_REG_SMOKE = [_d for _d in _REG_ALL if "smoke" in _d]
chk("registry: directories carrying arm-runs, and how many are start-up checks",
    (len(_REG_ALL), len(_REG_SMOKE)), (77, 3))
_REG_ARMRUNS = sum(len(glob.glob(f"{_d}*__rep*.json"))
                   for _d in _REG_ALL if _d not in _REG_SMOKE)
chk("registry: arm-runs outside the start-up checks", _REG_ARMRUNS, 3002)
# counted, not tested for membership: the README says it twice, once in prose
# and once in the tier registry, and `in` is satisfied by either
chk("registry: and the README quotes that number, in both places it states it",
    " ".join(_ROOT_TEXT.split()).count(f"{_REG_ARMRUNS} arm-runs"), 2)
chk("registry: with the directory count beside it",
    f"{len(_REG_ALL) - len(_REG_SMOKE)} directories" in _ROOT_TEXT
    or f"{len(_REG_ALL) - len(_REG_SMOKE)} committed directories" in _ROOT_TEXT, True)
# the letter families the registry table is allowed to group under one row
_REG_FAMILY = {re.match(r"([A-Z]+)", _t).group(1) for _t in _REG_DIRS
               if re.match(r"[A-Z]", _t)}
_reg_text = _ROOT_TEXT.split("The v4 runs, and what each one is for:")[1] \
    .split("The v2 / Exp 2 / v3 files")[0]
_reg_named = set(re.findall(r"^\| ([A-Z][A-Za-z0-9 /]*?) \|", _reg_text, re.M))
_reg_named = {_x.strip() for _r in _reg_named for _x in _r.split("/")}
_reg_missing = sorted(_f for _f in _REG_FAMILY
                      if not any(_n == _f or _n.startswith(_f) for _n in _reg_named))
chk("registry: every run family the data holds is named in the table",
    _reg_missing, [])
chk("registry: and run W is one of them", "W" in _reg_named, True)
# The ROW, not the document. The span also appears in the blockquote near the
# top, so searching the whole README satisfied a check named for the registry
# even if the registry row had lost it: two occurrences, one assertion, and the
# one it was written for was not the one that could keep it green.
# Matched on the first CELL, not on a pipe-delimited literal: the census
# treats any `| ... |` string in this file as a table header that must resolve
# to a registered reader, and `"| **v4 audit** |"` is a row, so it tripped that
# invariant. The invariant is right to be strict; the literal was the wrong
# shape for what it names.
_reg_v4row = [_l for _l in _ROOT_TEXT.splitlines()
              if _l.startswith("|") and _l.count("|") > 2
              and "**v4 audit**" in _l.split("|")[1]]
chk("registry: there is exactly one v4 audit row to check",
    len(_reg_v4row), 1)
# The span is the manifests' own first and last `created`. Written as a
# literal it went stale the moment W2 landed, and a row that names a date
# range is exactly the kind of number nothing else re-derives.
_reg_created = sorted(
    json.loads(pathlib.Path(_p).read_text(encoding="utf-8"))["created"]
    for _p in glob.glob("v4_audit_2026_08_25/data/*/manifest.json"))
_reg_span = f"{_reg_created[0][:10]} to {_reg_created[-1][:10]}"
chk("registry: the tier description gives the full date span, in that row",
    _reg_span in " ".join(_reg_v4row[0].split()), True)
chk("registry: it no longer says runs A to L",
    ("runs A–L" in _ROOT_TEXT or "runs A-L" in _ROOT_TEXT), False)

# --- five statements the line-by-line README review found imprecise ---------
# Each was true-ish and narrower or wider than the data behind it. The pattern
# is the one A18 records: a range that does not contain what it describes.
chk("README: the thinking-on count separates verified from inferred",
    ("5770 requests recorded" in _ROOT_TEXT
     and "134 in runs A and B" in _ROOT_TEXT
     and "all 5904 thinking-on requests" not in _ROOT_TEXT), True)
_th_field = _th_cap = _no_field = _no_field_cap = 0
for _f in glob.glob("v4_audit_2026_08_25/data/*/*__rep*.json"):
    for _x in json.load(open(_f))["rows"]:
        if _x.get("thinking_suppressed") is False:
            _th_field += 1
            _th_cap += _x["predicted_n"] == 300
        elif "thinking_suppressed" not in _x:
            _no_field += 1
            _no_field_cap += _x["predicted_n"] == 300
chk("README: 5770 requests carry the field and all of them hit the cap",
    (_th_field, _th_cap), (5770, 5770))
chk("README: and 134 predate it, also all at the cap",
    (_no_field, _no_field_cap), (134, 134))

chk("README: run O2 is called a Latin square balanced for position, not balanced",
    ("**Latin square balanced for position**" in _ROOT_TEXT
     and "as a **balanced Latin square**" not in _ROOT_TEXT), True)
chk("README: and it says which run is carryover balanced",
    "Run W is the design that balances both" in _ROOT_TEXT, True)

_th_acc_vals = []
for _half in ("thinkon", "thinkoff"):
    _da, _dn = defaultdict(int), defaultdict(int)
    for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/matrix_L_{_half}_20260826_032652/spec-dflash-n2__rep*.json")):
        for _x in json.load(open(_f))["rows"]:
            _da[_x["tag"]] += _x["draft_n_accepted"]
            _dn[_x["tag"]] += _x["draft_n"]
    _th_acc_vals += [100 * _da[_t] / _dn[_t] for _t in ("reasoning", "code_small")]
chk("README: the constrained prompts' acceptance band contains all four values",
    (round(min(_th_acc_vals)) >= 82, round(max(_th_acc_vals)) <= 92), (True, True))
chk("README: and the band it prints is that one",
    ("between 82 % and 92 %" in _ROOT_TEXT
     and "stays ~85–90 %" not in _ROOT_TEXT), True)

_sa = json.load(open("v4_audit_2026_08_25/data/spec_accounting_20260826.json"))
_gen = {(_r["arm"], _r["run"]): _r["drafter_generate_s"] for _r in _sa
        if _r.get("drafter_generate_s") is not None}
chk("README: the dense drafter's generate() second, from the committed dump",
    _gen[("spec-draft-n8", "matrix_J2_20260826_014750")], 17.24, 0.005)
_self_spec = [_v for (_a, _r), _v in _gen.items() if _a != "spec-draft-n8"]
chk("README: the self-speculative heads span 1.89 s to 6.27 s, not 1.89 to 3.43",
    (round(min(_self_spec), 2), round(max(_self_spec), 2)), (1.89, 6.27))
chk("README: and it now says so",
    ("1.89 s to 6.27 s" in _ROOT_TEXT
     and "against 1.89–3.43 s for a head" not in _ROOT_TEXT), True)

# The band is given as the measured range, not as a fraction. "A fifth to a
# quarter" excluded both ends; "a sixth to a quarter" still excluded +26.7 %,
# because a quarter is 25. A fraction that has to contain 17.3 and 26.7 does not
# exist in small integers, so the sentence quotes the numbers.
chk("README: the headline band is the measured range, not a fraction",
    ("spans **+17.3 % to +26.7 %**" in _ROOT_TEXT
     and "a fifth to a quarter" not in _ROOT_TEXT
     and "sixth and a quarter" not in _ROOT_TEXT), True)
chk("README: the n-gram draft volume names both values it generalises over",
    ("a sixth to a fifth of tokens, 0.17 and 0.19" in " ".join(_ROOT_TEXT.split())), True)
_ng_dpg = {}
for _a in ("ngram-mod-n24", "ngram-cache"):
    _dn = _ng = 0
    for _f in glob.glob(f"{_O2}/{_a}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _ng += _x["predicted_n"]
    _ng_dpg[_a] = round(_dn / _ng, 2)
chk("README: and those two values are what run O2 measured",
    sorted(_ng_dpg.values()), [0.17, 0.19])

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
chk("README names the interval it publishes, and its scope",
    "95 % CI (t, blocks within this invocation)" in _README0, True)
# A16 shows the interval does not cover invocation-to-invocation variation, so
# the column must not read as though it were a configuration-level interval.
chk("README does not call it a plain 95 % CI over blocks",
    "95 % CI (t, over blocks)" in _README0, False)


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
    "barely overlap" in " ".join(_norm(
        pathlib.Path(__file__).resolve().parents[1]
        .joinpath("ERRATA.md").read_text(encoding="utf-8")).split()), True)


print("\n=== the same headline table in the audit README ===")
# 42 cells, the largest in that document. It is run O2 again, one column
# different from the copy in the root README: absolute draft tokens where that
# one prints draft per generated token. Both are parsed now, and required to
# agree where they overlap - two copies of one table drifting apart is a defect
# this repository has already had once, in the checkpoint cost table.
_V4O2 = {}
for _r in _v4_table("| arm | pooled tok/s | change | 95 % CI | acceptance"):
    if len(_r) >= 6:
        _V4O2[_r[0].replace("**", "").strip("` ")] = _r[1:6]
chk("v4 README O2 table: nine arms", len(_V4O2), 9)
for _arm, _c in sorted(_V4O2.items()):
    _key = "baseline" if _arm.startswith("no speculation") else _arm
    _dn = _da = 0
    for _f in glob.glob(f"{_O2}/{_key}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _da += _x["draft_n_accepted"]
    chk(f"v4 README O2 {_arm}: pooled",
        round(_pooled_arm(_O2, _key), 1), _cellv4(_c[0]), 0.06)
    chk(f"v4 README O2 {_arm}: draft tokens", _dn, int(_cellv4(_c[4])))
    if _key == "baseline":
        chk(f"v4 README O2 {_arm}: the reference has no change, interval or acceptance",
            (any(ch.isdigit() for ch in _c[1] + _c[2] + _c[3]), _dn), (False, 0))
        continue
    chk(f"v4 README O2 {_arm}: change (%)",
        round(_pb_o2[_key]["point_pct"], 1), _cellv4(_c[1]), 0.06)
    _lo, _hi = (_cellv4(_x) for _x in _norm_early(_c[2]).strip("[] ").split(","))
    chk(f"v4 README O2 {_arm}: interval",
        [round(_x, 1) for _x in _pb_o2[_key]["ci95_t_pct"]], [_lo, _hi], 0.06)
    chk(f"v4 README O2 {_arm}: acceptance (%)",
        round(100 * _da / _dn, 1), _cellv4(_c[3]), 0.06)
chk("v4 README O2 table: it is sorted by pooled rate, descending",
    [_cellv4(_c[0]) for _c in _V4O2.values()],
    sorted((_cellv4(_c[0]) for _c in _V4O2.values()), reverse=True))

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
            and "spec-dflash-n2" in (_mm.get("arms") or {})
            # A16 is "twelve times IN ONE DAY". Run T4 satisfies every other
            # criterion and ran on 2026-08-27, so it is a later measurement of
            # the same thing, reported in A16's addendum rather than folded into
            # the same-day statistics it would otherwise change.
            and str(_mm.get("created", "")).startswith("2026-08-26")):
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
chk("ERRATA reports run V",
    "Run V is a fixed-order sensitivity analysis" in _er_v, True)
chk("and says the two methods agree in direction, not magnitude",
    "agree in direction and not in magnitude" in _er_v, True)
# The order confound the third review found: both halves are position-balanced
# internally, and the mode itself was not interleaved at all.
chk("A17 states that the two halves were not interleaved",
    "were not interleaved" in _er_v, True)
chk("A17 gives both start stamps",
    "22:31:46" in _er_v and "22:48:08" in _er_v, True)
chk("A17 names the A16 effect it cannot separate from",
    "9.26 pp" in _er_v and "9.4 pp" in _er_v, True)
# The comparator that matches run V's design is not the full-day span but the
# closest pair of invocations, which is tighter and worse.
_uvals = {"U1": 22.3, "U2": 24.2, "U3": 17.3, "U4": 19.9, "U5": 25.6, "U6": 24.3}
chk("A17 uses the closest-pair comparator",
    "six minutes apart and differ by 8.30 pp" in re.sub(r"\s+", " ", _er_v), True)
chk("and that gap is what runs U3 and U5 actually differ by",
    round(_uvals["U5"] - _uvals["U3"], 2), 8.30, 0.005)
chk("A17 shows the two halves could not have overlapped",
    "938 s" in _er_v and "982 s gap" in re.sub(r"\s+", " ", _er_v), True)
chk("A17 closes only the crossover half of P1-3",
    "closes the crossover half of" in _er_v, True)
chk("and not the within-invocation half",
    "does **not** close" in _er_v, False)
chk("A17 separates the two estimands",
    "answer different questions" in _er_v, True)



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

# W2's telemetry, on the same definition of loaded. A first draft of A17's W2
# paragraph said "zero throttle samples"; the driver prints no such summary and
# nothing here had counted the trace, so the sentence rested on nothing.
_tW = [(u, r) for u, r in _trace("gpu_telemetry_W_20260828_104222.csv",
                                 "util", "throttle") if u >= 50]
_tW2 = [(u, r) for u, r in _trace("gpu_telemetry_W2_20260830_220554.csv",
                                  "util", "throttle") if u >= 50]
chk("W and W2 loaded telemetry samples", (len(_tW), len(_tW2)), (4297, 10271))
chk("W2: the throttle flags on those samples",
    (_bits(_tW2, "throttle", 0x4), _bits(_tW2, "throttle", 0x20),
     _bits(_tW2, "throttle", 0x40)), (1762, 13, 10))
chk("W: the same three, which A17 quotes beside them",
    (_bits(_tW, "throttle", 0x4), _bits(_tW, "throttle", 0x20),
     _bits(_tW, "throttle", 0x40)), (706, 4, 3))
chk("W2: and the card stays far below the throttle point",
    (round(min(float(r["temp"]) for _u, r in _tW2)),
     round(max(float(r["temp"]) for _u, r in _tW2))), (45, 73))
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
_A16_T3_SHARE = 100 * _ck3 / _ex3
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
            and _m.get("target_sha256") == _TGT and "baseline" in _m["arms"]
            # same-day, for the same reason as A16's set above: the footnote is
            # about twelve measurements on 2026-08-26, and run T4 is a later one
            and str(_m.get("created", "")).startswith("2026-08-26")):
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
chk("README names the lowest", "un U3 is, at +17.3 %" in _README, True)
chk("README says the interval describes the run, not the configuration",
    "describe run O2, not the configuration" in " ".join(_README.split()), True)
chk("README quotes the between-run range",
    "+17.3 % to +26.7 %" in _norm(_README), True)
chk("and the range is what the twelve runs actually span",
    [round(min(_dfl.values()), 1), round(max(_dfl.values()), 1)], [17.3, 26.7], 0.05)
chk("README no longer claims the lower of the two is quoted",
    "The lower of the two is quoted" in _README, False)


import ast as _ast0
import pathlib as _pl0
print("\n=== the pull request body (PULL_REQUEST.md) ===")
# The third review's P0-4 was errors in the PR body. Fixing them once fixes
# nothing durable: the body is a published document with four numeric tables in
# it and nothing was reading them, which is the same defect this audit has now
# found five times. The body lives in the tree and its tables are parsed here.
# `gh pr edit 2 --body-file PULL_REQUEST.md` is what publishes it.
_PR_LINES = (pathlib.Path(__file__).resolve().parents[1] / "PULL_REQUEST.md") \
    .read_text(encoding="utf-8").splitlines()


_PR = re.sub(r"\s+", " ", _norm("\n".join(_PR_LINES)))


def _pr_table(header_startswith):
    # tables inside a list item are indented; strip before matching
    lines = [l.strip() for l in _PR_LINES]
    i = next((i for i, l in enumerate(lines)
              if _norm(l).startswith(header_startswith)), None)
    if i is None:
        return []
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip().strip("*`").replace("`", "").replace("*", "").strip()
                     for c in _norm(l).strip("|").split("|")])
    return rows


# The body's own header comment says how many tables it has, which is one more
# unchecked figure in a file that exists because of unchecked figures.
def _pr_tables():
    lines = [l.strip() for l in _PR_LINES]

    def _rule(l):
        bare = l.replace("|", "").replace(" ", "")
        return bool(bare) and set(bare) <= set("-:")

    return [l for i, l in enumerate(lines)
            if l.startswith("|") and i + 1 < len(lines) and _rule(lines[i + 1])]


_PRN = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_hdr = next((w for w in _PRN if f"the {w} tables below" in "\n".join(_PR_LINES[:12])), None)
chk("PR body: its header says how many tables it has", _hdr is not None, True)
chk("PR body: and that is how many there are", _PRN.get(_hdr), len(_pr_tables()))

# --- the headline table: run O2's nine arms, pooled, against the baseline ----
def _prpool(d, arm):
    """`_pool` is defined three times in this file and returns a tuple by here."""
    n = ms = 0
    for f in glob.glob(f"{d}/{arm}__rep*.json"):
        r = json.load(open(f))
        n += sum(x["predicted_n"] for x in r["rows"])
        ms += sum(x["predicted_ms"] for x in r["rows"])
    return 1000 * n / ms


_PRH = {r[0]: r[1:] for r in
        _pr_table("| arm | pooled tok/s | change |")}
chk("the PR body's headline table rows", len(_PRH), 9)
chk("and the no-speculation row is one of them", "no speculation" in _PRH, True)
_pr_base = _prpool(_O2, "baseline")
chk("PR body headline: no-speculation pooled",
    round(_pr_base, 1), _cell(_PRH["no speculation"][0]), 0.05)
for _arm, _row in sorted(_PRH.items()):
    if _arm == "no speculation":
        continue
    _pv = _prpool(_O2, _arm)
    chk(f"PR body headline {_arm} pooled tok/s",
        round(_pv, 1), _cell(_row[0]), 0.05)
    chk(f"PR body headline {_arm} change (%)",
        round(100 * (_pv / _pr_base - 1), 1), _cell(_row[1]), 0.05)
    # The three columns to the right of `change` were unread until 2026-08-29.
    # Perturbing any of them left the checker at exit 0 with no new failure, in
    # the one document that is published outside this repository. Found by
    # perturbing every number of every table the census calls parsed, rather
    # than one number per table, which is what let it pass before.
    _lo, _hi = (_cell(_x) for _x in _norm(_row[2]).strip("[] ").split(","))
    chk(f"PR body headline {_arm} interval",
        [round(_x, 1) for _x in _pb_o2[_arm]["ci95_t_pct"]], [_lo, _hi], 0.05)
    _dn = _da = _ng = 0
    for _f in glob.glob(f"{_O2}/{_arm}__rep*.json"):
        for _x in json.load(open(_f))["rows"]:
            _dn += _x["draft_n"]
            _da += _x["draft_n_accepted"]
            _ng += _x["predicted_n"]
    chk(f"PR body headline {_arm} draft per generated token",
        round(_dn / _ng, 2), _cell(_row[3]), 0.005)
    chk(f"PR body headline {_arm} acceptance (%)",
        round(100 * _da / _dn, 1), _cell(_row[4]), 0.05)
# and the row that is the reference: it drafts nothing, so three of its cells
# are dashes and must stay that way
# the reference row: no delta, no interval, no acceptance - but a real 0.00 in
# the draft column, which is a measurement and is checked as one
_pr_ns = _PRH["no speculation"]
chk("PR body headline: the no-speculation row carries no delta, interval or "
    "acceptance",
    [_i for _i in (1, 2, 4) if any(_ch.isdigit() for _ch in _pr_ns[_i])], [])
chk("PR body headline: and it drafts nothing",
    (_cell(_pr_ns[3]),
     sum(x["draft_n"] for _f in glob.glob(f"{_O2}/baseline__rep*.json")
         for x in json.load(open(_f))["rows"])), (0.0, 0))
chk("PR body headline: every cell of it agrees with the audit README's copy",
    {_a: [_cell(_x) for _x in _r[:2]] for _a, _r in _PRH.items()
     if _a != "no speculation"},
    {_a: [_cellv4(_c[0]), _cellv4(_c[1])] for _a, _c in _V4O2.items()
     if not _a.startswith("no speculation")})
chk("PR body headline: the ordering it presents is by pooled rate, descending",
    [a for a, r in sorted(_PRH.items(), key=lambda kv: -_cell(kv[1][0]))],
    [r[0] for r in _pr_table("| arm | pooled tok/s | change |")])

# --- what it costs: the A12 accounting, as the body groups it ---------------
# The body merges A12's `load_tgt` and `load_dft` into one restore row and
# renames `update_tgt` to save, so the labels differ from ERRATA's and the
# numbers must not.
_PRC = {r[0]: r[1:] for r in _pr_table("| | seconds | share of the excess |")}
chk("the PR body's cost table rows", len(_PRC), 5)
chk("PR body cost: the excess over no speculation",
    round(_exc, 1), _cell(_PRC["excess over no speculation"][0]), 0.05)
chk("PR body cost: checkpoint save, seconds",
    round(st.mean([r["update_tgt_s"] for r in _ext]), 2),
    _cell(_PRC["speculative checkpoint save (785)"][0]), 0.005)
chk("PR body cost: checkpoint restore is load_tgt plus load_dft",
    round(_restore, 2), _cell(_PRC["speculative checkpoint restore (728)"][0]), 0.005)
# `_ck` is rebound further up this file; recompute from the rows themselves
_pr_ck = st.mean([r["checkpoint_total_s"] for r in _ext])
chk("PR body cost: and the save row is not the whole checkpoint",
    _cell(_PRC["speculative checkpoint save (785)"][0])
    + _cell(_PRC["speculative checkpoint restore (728)"][0]),
    round(_pr_ck, 2), 0.015)
for _row in ("speculative checkpoint save (785)",
             "speculative checkpoint restore (728)",
             "drafter generate()", "unattributed"):
    chk(f"PR body cost: {_row} share of the excess (%)",
        round(100 * _cell(_PRC[_row][0]) / _exc, 1), _cell(_PRC[_row][1]), 0.05)
chk("PR body cost: the excess row's own share is the whole of it",
    _cell(_PRC["excess over no speculation"][1]), 100.0, 0.05)
chk("PR body cost: the four rows add up to the excess",
    round(sum(_cell(_PRC[k][0]) for k in
              ("speculative checkpoint save (785)",
               "speculative checkpoint restore (728)",
               "drafter generate()", "unattributed")), 2),
    round(_exc, 2), 0.05)
chk("PR body cost: and the shares add up to 100 %",
    round(sum(_cell(_PRC[k][1]) for k in
              ("speculative checkpoint save (785)",
               "speculative checkpoint restore (728)",
               "drafter generate()", "unattributed")), 1), 100.0, 0.15)

# --- and the same table in the top-level README ----------------------------
# It was never parsed either, and it carried the same 30.5 %: the merged
# restore row is 21.74 / 71.4 = 30.4 %, and 30.5 was load_tgt's share plus
# load_dft's, each rounded first. With 30.4 the column adds to exactly 100.
_RML = (pathlib.Path(__file__).resolve().parents[1] / "README.md") \
    .read_text(encoding="utf-8").splitlines()
_i = next(i for i, l in enumerate(_RML) if l.startswith("| | seconds | share |"))
_RMC = {}
for _l in _RML[_i + 2:]:
    if not _l.startswith("|"):
        break
    _c = [x.strip().strip("*`").replace("`", "").strip("* ").strip()
          for x in _norm(_l).strip("|").split("|")]
    _RMC[_c[0]] = _c[1:]
chk("the README's cost table rows", len(_RMC), 4)
chk("README cost: checkpoint save, seconds",
    round(st.mean([r["update_tgt_s"] for r in _ext]), 2),
    _cell(_RMC["speculative checkpoint save (785)"][0]), 0.005)
chk("README cost: checkpoint restore, seconds",
    round(_restore, 2), _cell(_RMC["speculative checkpoint restore (728)"][0]), 0.005)
for _row in ("speculative checkpoint save (785)",
             "speculative checkpoint restore (728)",
             "drafter generate()", "unattributed"):
    chk(f"README cost: {_row} share (%)",
        round(100 * _cell(_RMC[_row][0]) / _exc, 1), _cell(_RMC[_row][1]), 0.05)
chk("README cost: the rows add up to the excess",
    round(sum(_cell(v[0]) for v in _RMC.values()), 2), round(_exc, 2), 0.05)
chk("README cost: and the shares add up to 100 %",
    round(sum(_cell(v[1]) for v in _RMC.values()), 1), 100.0, 0.05)
chk("README and the PR body publish the same cost table",
    {k: v for k, v in _RMC.items()},
    {k: v for k, v in _PRC.items() if k != "excess over no speculation"})
chk("README records that the boundary was measured",
    "0.002 s of 39.09 s" in _norm(pathlib.Path(__file__).resolve().parents[1]
                                  .joinpath("README.md").read_text(encoding="utf-8")),
    True)

# --- run T4: the split that answers the API-boundary objection --------------
_PRT4 = {r[0]: r[1:] for r in
         _pr_table("| | seconds | share of the 71.49 s excess |")}
chk("the PR body's T4 table rows", len(_PRT4), 3)
chk("PR body T4: the wait, in seconds",
    round(_t4_sync, 3), _cell(_PRT4["of which, waiting on synchronize()"][0]), 0.0005)
chk("PR body T4: inside the checkpoint calls",
    round(_t4_call, 2), _cell(_PRT4["inside the checkpoint calls"][0]), 0.005)
chk("PR body T4: and the share of the excess",
    round(100 * _t4_call / _t4_excess, 1),
    _cell(_PRT4["inside the checkpoint calls"][1]), 0.05)
chk("PR body T4: the wait's share, to three decimals",
    round(100 * _t4_sync / _t4_excess, 3),
    _cell(_PRT4["of which, waiting on synchronize()"][1]), 0.0005)
# the third row was unread in both copies: state work is the call time minus
# the wait, and it is the row the "API boundary" objection turns on
chk("PR body T4: the state-work row, seconds",
    round(_t4_call - _t4_sync, 2),
    _cell(_PRT4["of which, state work"][0]), 0.005)
chk("PR body T4: and its share",
    round(100 * (_t4_call - _t4_sync) / _t4_excess, 1),
    _cell(_PRT4["of which, state work"][1]), 0.05)
chk("PR body T4: the excess in its table header matches the data",
    round(_t4_excess, 1), 71.5, 0.05)

# --- run V2: the crossover ---------------------------------------------------
_PRV2 = {r[0]: r[1:] for r in
         _pr_table("| arm | freerun | hard cap | shift, 95 % t over 8 sessions |")}
chk("the PR body's V2 table rows", len(_PRV2), 4)
chk("and it names the same four arms the data has",
    sorted(_PRV2), sorted(_v2_shift))
# `.split("[")[0]` is the shape this body calls out two sections down, and it
# was still here: three of these cells carry an interval and only the point
# estimate was read, so the bounds were the last unguarded numbers in it.
for _a, _row in sorted(_PRV2.items()):
    for _k, (_what, _vals) in enumerate((("freerun (%)", _v2_free[_a]),
                                         ("hard cap (%)", _v2_cap[_a]),
                                         ("shift (pp)", _v2_shift[_a]))):
        _pt, _lo, _hi = _iv2(_row[_k])
        _mine = _lm.interval(_vals)
        chk(f"PR body V2 {_a} {_what}", round(_mine[0], 2), _pt, 0.005)
        if _lo is not None:
            chk(f"PR body V2 {_a} {_what}, both bounds",
                (round(_mine[1], 2), round(_mine[2], 2)), (_lo, _hi))

# --- run V3 against V2, which is the disagreement the body reports -----------
_PRV3 = {r[0]: r[1:] for r in
         _pr_table("| arm | V3, within invocation | V2, between invocations |")}
chk("the PR body's V3 table rows", len(_PRV3), 4)
for _a, _row in sorted(_PRV3.items()):
    chk(f"PR body V3 {_a} within-invocation mean (pp)",
        round(st.mean(_v3_shift[_a]), 2), _cell(_row[0]), 0.005)
    chk(f"PR body V3 {_a} against V2 (pp)",
        round(_lm.interval(_v2_shift[_a])[0], 2), _cell(_row[1].split("[")[0]), 0.005)

# --- the re-derivation table, against the dumps it names --------------------
_PRR = {r[0]: r[1:] for r in
        _pr_table("| derived file | records | identical | not regenerated |")}
chk("the PR body's re-derivation table rows", len(_PRR), 4)
for _f, _row in sorted(_PRR.items()):
    _n = len(json.loads((pathlib.Path(__file__).resolve().parents[1]
                         / "v4_audit_2026_08_25" / _f).read_text(encoding="utf-8")))
    chk(f"PR body re-derivation: {_f} record count", _n, int(_cell(_row[0])))
    chk(f"PR body re-derivation: {_f} identical plus not-regenerated",
        int(_cell(_row[1])) + int(_cell(_row[2])), _n)
chk("PR body re-derivation: it names the split dump run T4 produced",
    "data/checkpoint_timers_20260827_split.json" in _PRR, True)

# --- and the third copy of it, in the changelog, which nothing read ---------
# `--probe --covered` reported this one as surviving a wrong number while the
# census called it parsed: the census matched its header against a literal used
# on a different document. Both are fixed - the census binds each reader to its
# document, and this reads the changelog's own copy.
_CH_LINES = (pathlib.Path(__file__).resolve().parents[1]
             / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()


def _ch_table(header_startswith):
    _i = next((_i for _i, _l in enumerate(_CH_LINES)
               if _norm(_l).startswith(header_startswith)), None)
    if _i is None:
        return []
    _rows = []
    for _l in _CH_LINES[_i + 2:]:
        if not _l.startswith("|"):
            break
        _rows.append([_c.strip().strip("*`").replace("`", "").replace("*", "").strip()
                      for _c in _l.strip("|").split("|")])
    return _rows


_CHR = {r[0]: r[1:] for r in
        _ch_table("| derived file | records | identical | not regenerated |")}
chk("the changelog's re-derivation table rows", len(_CHR), 3)
for _f, _row in sorted(_CHR.items()):
    _n = len(json.loads((pathlib.Path(__file__).resolve().parents[1]
                         / "v4_audit_2026_08_25" / _f).read_text(encoding="utf-8")))
    chk(f"changelog re-derivation: {_f} record count", _n, int(_cell(_row[0])))
    chk(f"changelog re-derivation: {_f} identical plus not-regenerated",
        int(_cell(_row[1])) + int(_cell(_row[2])), _n)
    chk(f"changelog re-derivation: {_f} agrees with the PR body's copy",
        [_cell(_x) for _x in _row], [_cell(_x) for _x in _PRR[_f]])
chk("changelog re-derivation: it is the PR body's table minus the T4 split dump, "
    "which the entry predates",
    sorted(set(_PRR) - set(_CHR)), ["data/checkpoint_timers_20260827_split.json"])

# --- the same table in the audit README, which nothing read either ----------
_V4R = {r[0]: r[1:] for r in
        _v4_table("| derived file | records | identical | not reproducible |")}
chk("the audit README's re-derivation table rows", len(_V4R), 4)
chk("and it lists the same files the body does",
    sorted(_V4R), sorted(_PRR))
for _f, _row in sorted(_V4R.items()):
    _n = len(json.loads((pathlib.Path(__file__).resolve().parents[1]
                         / "v4_audit_2026_08_25" / _f).read_text(encoding="utf-8")))
    chk(f"audit README re-derivation: {_f} record count", _n, int(_cellv4(_row[0])))
    chk(f"audit README and the body agree on {_f}",
        [_row[0], _row[1]], [_PRR[_f][0], _PRR[_f][1]])
    # the third column is the remainder, and nothing read it
    _not_repro = _row[2].strip()
    chk(f"audit README re-derivation: {_f} accounts for every record",
        int(_cellv4(_row[1])) + (0 if _not_repro in ("-", "\u2014", "")
                                 else int(_cellv4(_not_repro))), _n)
# and both must say the workflow has not run, because it has not
_V4TXT = re.sub(r"\s+", " ", _norm((pathlib.Path(__file__).resolve().parents[1]
                / "v4_audit_2026_08_25" / "README.md").read_text(encoding="utf-8")))
# The workflow ran for the first time on 2026-08-29, from the `push` filter,
# and passed: it fetched the archive, checked it against the manifest, unpacked
# it, re-derived the committed JSON from the raw logs and ran this checker over
# the result. The three documents said it never had, which was true until that
# push, and they say what happened now.
chk("the audit README says the script produced the table and CI reproduced it",
    "produced by running the script. CI has now reproduced it" in _V4TXT, True)
chk("and it no longer says the workflow has never run",
    "has **never run**" in _V4TXT, False)
chk("and it still says why the other three triggers cannot fire",
    "**default branch**, and this file lives only on `audit-2026-08-25`"
    in _V4TXT, True)
chk("the body says the same", "CI has now reproduced it" in _PR, True)
chk("and the body no longer says it has never run",
    "it has **never run**" in _PR, False)
chk("and neither claims the evidence workflow runs in CI",
    ("evidence.yml` downloads them" in _PR
     or "evidence.yml` does it in CI" in _V4TXT
     or "evidence.yml` does exactly that" in _V4TXT), False)

# --- and the prose figures the body leads with ------------------------------
chk("PR body: the arm-run total it claims matches the three runs on disk",
    f"{sum(len(list(d.glob('*__rep*.json'))) for d in _V2) + 200 + 18} arm-runs" in _PR,
    True)
# Until run W the honest answer was a range, because two designs disagreed and
# neither could attribute it. W attributes it, so the body states the
# within-invocation figure and says what the crossover measures instead.
chk("PR body no longer reports that arm as a bare range",
    "design-dependent, +5.9 to +8.7 pp" in _PR, False)
chk("PR body says W's interval overlaps V3 and not V2",
    "overlaps V3's and does not" in _PR, True)
chk("PR body reports the predecessor null as a null at this power",
    "no\ndetectable effect at this power" in _PR
    or "no detectable effect at this power" in _PR, True)
# This used to assert that the body said the randomised-order run had NOT been
# run -- twice. Run W is that experiment and it is complete, so the same two
# sentences became the body's own contradiction of its opening section, and a
# check that pinned them kept them there. The assertion is now the other way
# round: a completed run may not be described as pending. The one surviving
# occurrence is inside a block quote of the earlier review, marked as such and
# answered underneath, which is a quotation rather than a claim.
chk("PR body: it does not describe the completed randomised-order run as pending",
    _PR.count("it has not been run"), 0)
chk("PR body: and it says W is that experiment",
    "W is that experiment" in _PR, True)
chk("PR body: it does not claim the mode order was the cause",
    "Order was not the cause" in _PR, True)
# Every test*.py, and only methods on a TestCase subclass, because that is what
# `unittest discover` runs. The version this replaced counted the string
# `    def test` in one named file, so a second test file - or a helper method
# named `test_` on a plain class - would have left the published count stale
# without failing.
_t_methods = 0
for _tf in sorted((_pl0.Path(__file__).resolve().parents[1] / "tests").glob("test*.py")):
    for _c in _ast0.walk(_ast0.parse(_tf.read_text(encoding="utf-8"))):
        if not (isinstance(_c, _ast0.ClassDef)
                and any("TestCase" in _ast0.unparse(_b) for _b in _c.bases)):
            continue
        _t_methods += sum(1 for _m in _c.body
                          if isinstance(_m, (_ast0.FunctionDef, _ast0.AsyncFunctionDef))
                          and _m.name.startswith("test"))
chk("PR body: the regression count it quotes is the suite's own",
    f"# {_t_methods} regressions" in _PR, True)
chk("PR body: and the count is not vacuous", _t_methods > 150, True)
chk("PR body: the run-directory count it quotes is what the checker walks",
    f"all {len([d for d in (_pl0.Path(__file__).resolve().parents[1] / 'v4_audit_2026_08_25' / 'data').iterdir() if d.is_dir()])} run directories" in _PR,
    True)
def _n_mutations(rel):
    """How many entries the MUTATIONS list in one of the two suites holds."""
    _t = _ast0.parse(_pl0.Path(__file__).resolve().parents[1].joinpath(rel)
                     .read_text(encoding="utf-8"))
    for _n in _t.body:
        if (isinstance(_n, _ast0.Assign) and len(_n.targets) == 1
                and getattr(_n.targets[0], "id", None) == "MUTATIONS"):
            return len(_n.value.elts)
    return None


chk("PR body: the mutation counts it quotes are the two suites' own",
    f"{_n_mutations('tests/mutate.py')} code and "
    f"{_n_mutations('tests/data_mutate.py')} data perturbations" in _PR,
    True)
# the same number, quoted a second time 90 lines earlier, where it had gone
# stale at 73 while the suite grew to 84
# --- the body's own prose about coverage, which went stale twice -----------
# the census section runs after this one, so it is computed here
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import table_coverage as _tcov_pr                                  # noqa: E402
_cov = _tcov_pr.census()
chk("PR body: the parsed count it quotes is the census's",
    f"{_cov['parsed']} are parsed" in _PR, True)
chk("PR body: the survivor count is reported against the census it was measured on",
    ("When 80 were still unparsed" in " ".join(_PR.split())
     and "left 67 that accept a wrong number" in " ".join(_PR.split())), True)
chk("PR body: and it says how many of those are parsed since",
    "all 80 of those are parsed now" in " ".join(_PR.split()), True)
chk("PR body: which is the difference the census records",
    80 - _cov["not_parsed"], 80)
chk("PR body: it no longer claims one cell per table proves a table guarded",
    "that are parsed catches all 44" in _PR, False)
chk("PR body: it names the per-number probe",
    "--probe --covered --every-cell" in _PR, True)
chk("PR body: its scope bullet reaches the last run",
    ("controlled tier is runs A to W" in _PR
     and "controlled tier is runs A to V3" not in _PR), True)
chk("PR body: and it does not call run O2 carryover balanced",
    ("Latin square balanced\nfor position" in _PR
     or "Latin square balanced for position" in " ".join(_PR.split())), True)
chk("PR body: the 618 arm-runs are V2, V3 and T4, and W is named beside them",
    (sum(len(glob.glob(f"{_d}/*__rep*.json"))
         for _p in ("matrix_V2_*", "matrix_V3_*", "matrix_T4_*")
         for _d in glob.glob(f"v4_audit_2026_08_25/data/{_p}")),
     "run\nW added 500 more" in _PR or "run W added 500 more" in " ".join(_PR.split())),
    (618, True))
chk("PR body: and W really is 500",
    sum(len(glob.glob(f"{_d}/*__rep*.json"))
        for _d in glob.glob("v4_audit_2026_08_25/data/matrix_W_*")), 500)

chk("PR body: and the second time it quotes the perturbation count",
    f"{_n_mutations('tests/data_mutate.py')} data and document perturbations" in _PR,
    True)

print("\n=== A16's O2-against-O3 table, which nothing read ===")
# Probed on 2026-08-28: adding 7 to `spec-dflash-n2`'s +26.3 % changed no
# verdict. The shift column is the whole of A16's "only this arm moves"
# argument and it was compared against literals in prose, never as a table.
_O2D = glob.glob("v4_audit_2026_08_25/data/matrix_O2_latin_*")[0]
_O3D = glob.glob("v4_audit_2026_08_25/data/matrix_O3_latin_*")[0]


def _pooled_arm(d, a):
    n = ms = 0
    for f in glob.glob(f"{d}/{a}__rep*.json"):
        r = json.load(open(f))
        if r.get("crashed") or not r.get("rows"):
            continue
        n += sum(x["predicted_n"] for x in r["rows"])
        ms += sum(x["predicted_ms"] for x in r["rows"])
    return (1000 * n / ms) if ms else None


_O2O3 = {r[0]: r[1:] for r in _md_table("| arm | O2 | O3 | shift |")}
chk("A16's O2/O3 table rows", len(_O2O3), 8)
_b2, _b3 = _pooled_arm(_O2D, "baseline"), _pooled_arm(_O3D, "baseline")
for _a, _row in sorted(_O2O3.items()):
    _d2 = 100 * (_pooled_arm(_O2D, _a) / _b2 - 1)
    _d3 = 100 * (_pooled_arm(_O3D, _a) / _b3 - 1)
    chk(f"A16 table: {_a} in O2 (%)", round(_d2, 1), _cell(_row[0]), 0.05)
    chk(f"A16 table: {_a} in O3 (%)", round(_d3, 1), _cell(_row[1]), 0.05)
    chk(f"A16 table: {_a} shift (pp)", round(_d3 - _d2, 1), _cell(_row[2]), 0.05)
chk("A16 table: the largest shift is the arm the section is about",
    min(_O2O3, key=lambda k: _cell(_O2O3[k][2])), "spec-dflash-n2")
chk("and every other arm moves less than half as far",
    max(abs(_cell(v[2])) for k, v in _O2O3.items() if k != "spec-dflash-n2")
    < abs(_cell(_O2O3["spec-dflash-n2"][2])) / 2, True)

print("\n=== run W: the carryover-balanced design ===")
_W = sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
             / "data").glob("matrix_W_s*_20260828_104222"))
chk("W sessions on disk", len(_W), 5)
chk("W arm-runs", sum(len(list(d.glob("*__rep*.json"))) for d in _W), 500)
chk("every W session validated",
    sorted(d.name for d in _W if not (d / "RUN_COMPLETE.json").exists()), [])
_WM = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in _W]
chk("W ran the Williams schedule", sorted({m["order_mode"] for m in _WM}), ["williams"])
chk("and every session claims all three balance properties",
    sorted({(m["schedule_is_position_balanced"],
             m["schedule_first_order_carryover_balanced"],
             m["schedule_randomized"]) for m in _WM}), [(True, True, True)])
chk("with a different seed each", len({m["schedule_seed"] for m in _WM}), 5)
chk("W is V3's configuration except the schedule",
    sorted({(m["think"], m["fit_target"], str(m["ctx"]), m["repeats"],
             m["hardcap_suffix"], len(m["arms"])) for m in _WM}),
    [("off", "3072", "8192", 10, "-cap", 10)])

# the balance is a property of what RAN, checked from t_start, not the label
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carryover as _co
_wbal = []
for _d in _W:
    _runs = _co.arm_runs(str(_d))
    _ok, _why = _co.is_balanced(_runs)
    _wbal.append(_ok)
chk("every arm preceded by every other exactly once, read from t_start",
    sorted(set(_wbal)), [True])

# Q1, against what A17 publishes
_wmode = defaultdict(list)
for _d in _W:
    _arms = _lm.arms_of(str(_d))
    _free = {a: _lm.pooled(str(_d), a)["tok_s"] for a in _arms
             if not a.endswith("-cap") and _lm.pooled(str(_d), a)}
    _cap = {a[:-4]: _lm.pooled(str(_d), a)["tok_s"] for a in _arms
            if a.endswith("-cap") and _lm.pooled(str(_d), a)}
    for _a, _v in _lm.contrast(_free, _cap).items():
        _wmode[_a].append(_v["shift_pp"])
_WT = {r[0]: r[1:] for r in _md_table(
    "| arm | V2, 8 sessions, between | V3, 2 sessions, within | "
    "**W, 5 sessions, within and carryover-balanced** |")}
chk("A17's four-design table rows", len(_WT), 4)


def _iv_cell(x):
    """`+12.03 [+11.67, +12.38]` -> (12.03, 11.67, 12.38); no bracket -> one value.

    The V2 and V3 columns of this table were never read - only the W column
    was, and only its point estimate. Perturbing V2's +12.03 to +19.03 passed
    every check, in both documents that carry the table.
    `analysis/table_coverage.py --probe --covered` is what found that: a table
    the census calls parsed is only covered in the columns someone parsed.
    """
    _head, _, _rest = _norm(x).partition("[")
    _pt = _cell(_head)
    if not _rest:
        return (_pt, None, None)
    _lo, _hi = (_cell(_y) for _y in _rest.rstrip("] ").split(","))
    return (_pt, _lo, _hi)


for _a, _row in sorted(_WT.items()):
    _m, _lo, _hi, _n = _lm.interval(_wmode[_a])
    chk(f"W {_a} shift (pp)", round(_m, 2), _iv_cell(_row[2])[0], 0.005)
    chk(f"W {_a} interval (pp)",
        (round(_lo, 2), round(_hi, 2)), _iv_cell(_row[2])[1:])
    chk(f"W {_a} sessions", _n, 5)
    _v2i = _lm.interval(_v2_shift[_a])
    chk(f"W table {_a}: its V2 column is V2's own shift",
        round(_v2i[0], 2), _iv_cell(_row[0])[0], 0.005)
    chk(f"W table {_a}: and V2's own interval",
        (round(_v2i[1], 2), round(_v2i[2], 2)), _iv_cell(_row[0])[1:])
    chk(f"W table {_a}: its V3 column is V3's two-session mean",
        round(st.mean(_v3_shift[_a]), 2), _iv_cell(_row[1])[0], 0.005)
chk("W's spec-dflash-n2 interval does not overlap V2's",
    _lm.interval(_wmode["spec-dflash-n2"])[1] > 6.99, True)
chk("and does overlap V3's",
    _lm.interval(_wmode["spec-dflash-n2"])[2] > 8.28, True)
chk("W's spec-dflash-n4 keeps the sign flip",
    _lm.interval(_wmode["spec-dflash-n4"])[1] > 0, True)

# Q2, the predecessor contrast the design exists for
_wcarry = defaultdict(list)
_wcarry_m = defaultdict(list)
for _d in _W:
    _runs = _co.arm_runs(str(_d))
    for _a, _v in _co.capped_contrast(_runs, "-cap").items():
        chk(f"W {_d.name[8:11]} {_a} predecessor split is the balanced one",
            _v["split_is_balanced"], True)
        _wcarry[_a].append(_v["delta_pct"])
    # The identity-matched version, which is the one that is a mode contrast.
    # The grouped estimator above puts an uncapped arm's own capped twin in the
    # capped group while the free group cannot contain the arm itself, so it
    # carries predecessor identity as well as predecessor mode.
    for _a, _v in _co.capped_contrast_matched(_runs, "-cap").items():
        _wcarry_m[_a].append(_v["delta_pct"])
_excl = sorted(a for a, v in _wcarry.items()
               if (lambda i: i[1] is not None and (i[1] > 0 or i[2] < 0))(_lm.interval(v)))
chk("no arm's predecessor interval excludes zero", _excl, [])
_wd2 = _lm.interval(_wcarry["spec-dflash-n2"])
chk("spec-dflash-n2 has the largest predecessor contrast",
    max(_wcarry, key=lambda k: abs(st.mean(_wcarry[k]))), "spec-dflash-n2")
chk("its point estimate (%)", round(_wd2[0], 2), -1.20, 0.005)
chk("and it points the way A17 guessed", _wd2[0] < 0, True)
chk("A17 reports it as a null at this power, with the interval",
    "[-2.61 %, +0.22 %]" in _norm(_ER_V3), True)

# Q3, and that the work was identical
_wcv = defaultdict(list)
for _d in _W:
    for _a in ("baseline", "spec-dflash-n2", "spec-dflash-n2-cap",
               "spec-mtp-n2", "spec-draft-n8"):
        _r = [1000 * sum(x["predicted_n"] for x in json.load(open(f))["rows"])
              / sum(x["predicted_ms"] for x in json.load(open(f))["rows"])
              for f in sorted(_d.glob(f"{_a}__rep*.json"))]
        _wcv[_a].append(100 * st.stdev(_r) / st.mean(_r))
# per-repeat CV over the ten arm-runs, which is what PROSPECTIVE_ANALYSIS_PLAN_W.md asked
# for and what V3's published 1.82 % is. `carryover.py`'s `spread` averages
# over predecessors first and reads 1.65 for the same arm; two definitions of
# the same word, and the table now says which one it publishes.
chk("W per-repeat CV, spec-dflash-n2 (%)",
    round(st.mean(_wcv["spec-dflash-n2"]), 2), 1.69, 0.005)
chk("W per-repeat CV, no speculation (%)",
    round(st.mean(_wcv["baseline"]), 2), 0.31, 0.005)
_WCVT = {r[0]: r[1:] for r in _md_table("| arm | mean per-repeat CV inside a session |")}
chk("A17's CV table rows", len(_WCVT), 5)
# all five rows, not the two the argument leans on: the other three were the
# last unread cells in this entry
_WCV_ARM = {"spec-dflash-n2": "spec-dflash-n2",
            "spec-dflash-n2-cap": "spec-dflash-n2-cap",
            "spec-mtp-n2": "spec-mtp-n2",
            "no speculation": "baseline",
            "spec-draft-n8": "spec-draft-n8"}
chk("A17's CV table names five of run W's ten arms",
    sorted(_WCVT), sorted(_WCV_ARM))
for _label, _arm in sorted(_WCV_ARM.items()):
    chk(f"A17 CV table {_label}",
        round(st.mean(_wcv[_arm]), 2), _cell(_WCVT[_label][0]), 0.005)
# and the other definition, so the two cannot be confused again
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carryover as _co_cv                                         # noqa: E402
_wspread = {_a: st.mean([_co_cv.spread_by_predecessor(
    _co_cv.arm_runs(str(_d)))[_a]["cv_pct"] for _d in _W])
    for _a in _WCV_ARM.values()}
chk("A17 CV: the other definition, which two of these cells used to carry",
    (round(_wspread["spec-dflash-n2-cap"], 2), round(_wspread["spec-mtp-n2"], 2)),
    (1.68, 0.50))
chk("A17 CV: and the entry now says so",
    "gives 1.68 % and 0.50 % for those" in " ".join(_ER_LINES_TEXT.split()), True)
chk("A17 CV table: the rows are in descending order, as printed",
    [_cell(_WCVT[_l][0]) for _l in
     ("spec-dflash-n2", "spec-dflash-n2-cap", "spec-mtp-n2", "no speculation",
      "spec-draft-n8")],
    sorted((_cell(_WCVT[_l][0]) for _l in _WCV_ARM), reverse=True))
chk("the unstable arm is five times the baseline",
    st.mean(_wcv["spec-dflash-n2"]) > 5 * st.mean(_wcv["baseline"]), True)
_wtxt = defaultdict(set)
_wdraft = defaultdict(set)
for _d in _W:
    for _f in _d.glob("*__rep*.json"):
        _j = json.loads(_f.read_text(encoding="utf-8"))
        _wtxt[_j["arm"]].add(tuple(r.get("content", "")[:200] for r in _j["rows"]))
        _wdraft[_j["arm"]].add((sum(r["draft_n"] for r in _j["rows"]),
                                sum(r["draft_n_accepted"] for r in _j["rows"])))
chk("every W arm produced exactly one distinct output set over five sessions",
    sorted({len(v) for v in _wtxt.values()}), [1])
# The prose says the drafted/accepted pairs match V2's and V3's exactly, and
# until 2026-08-28 nothing read them: halving one request's accepted count in
# `tests/data_mutate.py` changed no verdict. The claim is the counts, so the
# counts are what is asserted.
chk("and exactly one distinct drafted/accepted pair",
    sorted({len(v) for v in _wdraft.values()}), [1])
for _a, _pair in (("spec-dflash-n2", (1253, 732)), ("spec-dflash-n2-cap", (2556, 1710)),
                  ("spec-dflash-n4", (2061, 844)), ("spec-dflash-n4-cap", (4088, 1959)),
                  ("spec-mtp-n2", (1158, 782)), ("spec-mtp-n2-cap", (2367, 1801)),
                  ("spec-draft-n8", (2925, 677)), ("spec-draft-n8-cap", (5274, 1804))):
    chk(f"W {_a} drafted/accepted", sorted(_wdraft[_a]), [_pair])
chk("W's baseline arms draft nothing",
    sorted(_wdraft["baseline"] | _wdraft["baseline-cap"]), [(0, 0)])
# and the acceptance rates A17 quotes for the same arms in V2 and V3
for _a, _want in (("spec-dflash-n2", 58.4), ("spec-dflash-n2-cap", 66.9),
                  ("spec-mtp-n2", 67.5), ("spec-draft-n8", 23.1)):
    _dn, _da = sorted(_wdraft[_a])[0]
    chk(f"W {_a} acceptance (%)", round(100 * _da / _dn, 1), _want, 0.05)
chk("A17 says the pairs match the earlier runs",
    "matching V2's and V3's counts exactly" in " ".join(_ER_V3.split())
    or "the same\n1253/732" in _ER_V3
    or "1253/732" in _ER_V3, True)

# --- run W, four designs side by side ---------------------------------------
_PRW = {r[0]: r[1:] for r in _pr_table(
    "| arm | V2, between | V3, within | **W, within and carryover-balanced** |")}
chk("the PR body's four-design table rows", len(_PRW), 4)
chk("and it names the same arms the data has", sorted(_PRW), sorted(_wmode))
for _a, _row in sorted(_PRW.items()):
    _m, _lo, _hi, _n = _lm.interval(_wmode[_a])
    chk(f"PR body W {_a} shift (pp)", round(_m, 2), _iv_cell(_row[2])[0], 0.005)
    chk(f"PR body W {_a} interval (pp)",
        (round(_lo, 2), round(_hi, 2)), _iv_cell(_row[2])[1:])
    _v2i = _lm.interval(_v2_shift[_a])
    chk(f"PR body W {_a}: its V2 column is V2's own shift",
        round(_v2i[0], 2), _iv_cell(_row[0])[0], 0.005)
    chk(f"PR body W {_a}: and V2's own interval",
        (round(_v2i[1], 2), round(_v2i[2], 2)), _iv_cell(_row[0])[1:])
    chk(f"PR body W {_a}: its V3 column is V3's two-session mean",
        round(st.mean(_v3_shift[_a]), 2), _iv_cell(_row[1])[0], 0.005)
    # the two documents carry the same table; every cell of it must agree
    chk(f"PR body W {_a}: every cell agrees with A17's copy",
        [_iv_cell(_x) for _x in _row[:3]], [_iv_cell(_x) for _x in _WT[_a][:3]])
_PRV3 = {r[0]: r[1:] for r in
         _pr_table("| arm | V3, within invocation | V2, between invocations |")}
chk("the PR body's V3-against-V2 table rows", len(_PRV3), 4)
for _a, _row in sorted(_PRV3.items()):
    _v3m = round(st.mean(_v3_shift[_a]), 2)
    chk(f"PR body V3 {_a}: the within-invocation mean", _v3m, _cell(_row[0]), 0.005)
    _pt, _lo, _hi = _iv2(_row[1])
    _v2i = _lm.interval(_v2_shift[_a])
    chk(f"PR body V3 {_a}: the V2 column", round(_v2i[0], 2), _pt, 0.005)
    chk(f"PR body V3 {_a}: and its interval",
        (round(_v2i[1], 2), round(_v2i[2], 2)), (_lo, _hi))

# The evidence registry, which holds the re-derivation's expected coverage apart
# from the outputs it checks. Every scope in rederive_from_logs.py used to be
# read off the published artifact, so a run dropped from an output left the
# comparison rather than failing it.
_EREG = json.loads((_pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                    / "EVIDENCE_REGISTRY.json").read_text(encoding="utf-8"))
for _an, _av in _EREG["artifacts"].items():
    _ap = (_pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
           / "data" / _an)
    chk(f"evidence registry: {_an} exists", _ap.exists(), True)
    if not _ap.exists():
        continue
    _ad = json.loads(_ap.read_text(encoding="utf-8"))
    _ar = _ad if isinstance(_ad, list) else _ad.get("rows", [])
    _have = {r.get("run") for r in _ar if r.get("run")}
    if _have:                       # some artifacts carry no run column
        chk(f"evidence registry: {_an} covers exactly the runs it lists",
            sorted(_have), sorted(_av["expected_runs"]))
chk("evidence registry: it records what is NOT re-derived",
    "primary_benchmark_json" in _EREG["not_rederived"], True)
_EWF = (_pl0.Path(__file__).resolve().parents[1] / ".github" / "workflows"
        / "evidence.yml").read_text(encoding="utf-8")
chk("evidence workflow: its job name no longer claims the benchmark JSON",
    "name: raw logs to committed JSON" in _EWF, False)
chk("evidence workflow: and says what it does re-derive",
    "four log-derived audit files" in _EWF, True)

# The run registry. The PR body carried two contradictory narratives at once --
# W complete and W not run, the evidence workflow green and not green, two
# tranches and three -- because each fact lived in prose in more than one place.
# These are computed from the data and the manifest, and the documents are then
# checked against them rather than against each other.
_REG = json.loads((_pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                   / "RUN_REGISTRY.json").read_text(encoding="utf-8"))
_DATA = _pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "data"
for _rk, _pat in (("V2", "matrix_V2_*"), ("V3", "matrix_V3_*"), ("W", "matrix_W_*")):
    _ds = sorted(d for d in _DATA.glob(_pat) if d.is_dir())
    _ar = sum(len(list(d.glob("*__rep*.json"))) for d in _ds)
    chk(f"registry: {_rk} session count is the directories'",
        len(_ds), _REG["runs"][_rk]["sessions"])
    chk(f"registry: {_rk} arm-run count is the files'",
        _ar, _REG["runs"][_rk]["arm_runs"])
    chk(f"registry: {_rk} is attested by the driver in every session",
        [d.name for d in _ds if not (d / "RUN_COMPLETE.json").exists()], [])
_MANL = (_pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
         / "EVIDENCE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
chk("registry: the server-log count is the manifest's",
    sum(1 for x in _MANL if x.rstrip().endswith(".log")),
    _REG["raw_evidence"]["server_logs"])
chk("registry: the telemetry-trace count is the manifest's",
    sum(1 for x in _MANL if "telemetry" in x and x.rstrip().endswith(".csv")),
    _REG["raw_evidence"]["telemetry_traces"])
# A completed run may not be described as pending anywhere.
for _rk, _rv in _REG["runs"].items():
    if _rv["status"] == "complete" and _rk == "W":
        chk("registry: no document says the randomised-order run has not happened",
            [_d for _d, _t in (("PULL_REQUEST.md", _PR),)
             if "it has not been run" in _t], [])
chk("PR body: the evidence inventory is the registry's",
    (f"{_REG['raw_evidence']['server_logs']} " in _PR
     and f"{_REG['raw_evidence']['telemetry_traces']} telemetry traces" in _PR), True)
chk("PR body: it does not still say two tranches",
    "two tranches" in _PR.lower(), False)

# The V3-to-W runner diff, checked against the archived blobs rather than
# described. "V3 verbatim except BENCH_ORDER" was the claim; the manifests
# record different runner hashes, so the diff is archived and every number the
# note quotes about it is asserted here.
_HDIR = _pl0.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "harness"
_WRUN = _HDIR / "retest_runner_W_20260828_104222.py"
_DIFF = _HDIR / "V3_to_W_runner.diff"
_DNOTE = (_HDIR / "V3_TO_W_DIFF.md").read_text(encoding="utf-8")
import hashlib as _hl0                                              # noqa: E402
chk("W runner: the archived blob is the hash its manifests record",
    _hl0.sha256(_WRUN.read_bytes()).hexdigest(),
    "341a4a649c9215feb596561fd14c466274bc202c915455ab8d6a34fddd862f0b")
_dl = _DIFF.read_text(encoding="utf-8").splitlines()
_dadd = sum(1 for x in _dl if x.startswith("+") and not x.startswith("+++"))
_ddel = sum(1 for x in _dl if x.startswith("-") and not x.startswith("---"))
_dhunk = sum(1 for x in _dl if x.startswith("@@"))
chk("V3-to-W diff: the note's line counts are the diff's",
    (len(_dl), _dadd, _ddel, _dhunk), (189, 111, 3, 9))
for _n in ("189 lines", "111 added", "3 removed", "nine hunks"):
    chk(f"V3-to-W note quotes {_n!r}", _n in _DNOTE, True)
# The measurement path is what the note says is untouched, so that is asserted
# rather than asserted about: no hunk may mention these.
chk("V3-to-W diff: no hunk touches the request body or the timing extraction",
    [x for x in _dl if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))
     and any(k in x for k in ("def chat(", "predicted_ms", "predicted_n",
                              "ignore_eos", "max_tokens\"", "subprocess.Popen"))],
    [])
chk("PR body: it no longer says W is V3 verbatim",
    "verbatim except for `BENCH_ORDER`" in _PR, False)
chk("PR body: and it points at the archived diff",
    "V3_to_W_runner.diff" in _PR, True)

chk("PR body: the grouped predecessor result is the one computed",
    round(_lm.interval(_wcarry["spec-dflash-n2"])[0], 2), -1.20, 0.005)
_wm2 = _lm.interval(_wcarry_m["spec-dflash-n2"])
chk("PR body: the MATCHED predecessor result is the one computed",
    round(_wm2[0], 2), -1.05, 0.005)
chk("PR body: and its matched interval",
    (round(_wm2[1], 2), round(_wm2[2], 2)), (-2.97, 0.86))
chk("PR body quotes the matched figure and the grouped one it replaces",
    ("**-1.05 %**" in _norm(_PR) and "[-2.97, +0.86]" in _norm(_PR)
     and "-1.20 % [-2.61, +0.22]" in _norm(_PR)), True)
# The sentence that was wrong. -2.4 has to be inside the interval the document
# said it was far outside, or the correction in that paragraph is itself untrue.
_wg2 = _lm.interval(_wcarry["spec-dflash-n2"])
chk("PR body: -2.4 really is inside the grouped interval it was said to be outside",
    (_wg2[1] <= -2.4 <= _wg2[2]), True)
chk("PR body: and inside the matched one", (_wm2[1] <= -2.4 <= _wm2[2]), True)
chk("PR body: no matched interval excludes zero",
    [a for a, v in _wcarry_m.items()
     if not (_lm.interval(v)[1] <= 0 <= _lm.interval(v)[2])], [])
# The heading, not the phrase: the paragraph that replaces it quotes the old
# heading in order to withdraw it, and a bare substring test cannot tell the use
# from the mention.
chk("PR body: the withdrawn heading is gone",
    "**It is not the predecessor.**" in _PR, False)
chk("PR body: and the withdrawal names it",
    'heading "It is not the predecessor"' in _PR, True)
chk("PR body quotes the per-repeat CV pair",
    "**1.69 %**" in _PR and "**0.31 %**" in _PR, True)


print("\n=== do the canonical documents agree with each other? ===")
# The fourth review found five documents describing three different datasets:
# RETEST_TODO said the crossover was unrun while the PR body reported its
# results, CITATION.cff dated the controlled tier to 2026-08-26 and said the
# raw logs were unpublished, and the README's tier stopped at run V. Nothing
# compared them, because each was internally consistent.
_CFF = (pathlib.Path(__file__).resolve().parents[1] / "CITATION.cff") \
    .read_text(encoding="utf-8")
_TODO = _norm((pathlib.Path(__file__).resolve().parents[1] / "RETEST_TODO.md")
              .read_text(encoding="utf-8"))
_RDME = _norm((pathlib.Path(__file__).resolve().parents[1] / "README.md")
              .read_text(encoding="utf-8"))

# the newest run directory on disk is what the metadata has to cover
_all_runs = sorted((pathlib.Path(__file__).resolve().parents[1]
                    / "v4_audit_2026_08_25" / "data").glob("matrix_*"))
_stamps = sorted({d.name.rsplit("_", 2)[-2] for d in _all_runs
                  if d.name.rsplit("_", 2)[-2].startswith("2026")})
chk("the newest committed run is from 2026-08-30", _stamps[-1], "20260830")
chk("CITATION.cff's release date is not older than the newest run",
    re.search(r"^date-released:\s*(\S+)", _CFF, re.M).group(1) >= "2026-08-28", True)
chk("CITATION.cff no longer says the controlled-tier logs are unpublished",
    "logs themselves are not published" in _CFF, False)
chk("CITATION.cff says they are published",
    "published as release assets" in _CFF, True)
chk("the README's controlled tier reaches the last run, not V3",
    ("controlled tier** is runs A to W" in _RDME
     and "controlled tier** is runs A-V3" not in _RDME), True)
chk("RETEST_TODO no longer calls the crossover script unrun",
    "written and unrun" in _TODO, False)
chk("RETEST_TODO marks the crossover done",
    "mode order controlled.**~~ **Done 2026-08-27.**" in _TODO, True)
chk("RETEST_TODO marks the split timer done",
    "The checkpoint timers, split.**~~ **Done 2026-08-27.**" in _TODO, True)
chk("and it opens the carryover-balanced design instead",
    "carryover-balanced version of that design" in _TODO, True)
# a completed run must not be described as open anywhere
_completed = {"matrix_V2_s1_freerun_20260827_044442",
              "matrix_V3_s1_20260827_102614",
              "matrix_T4_split_20260827_175051"}
chk("each completed 2026-08-27 run is on disk with its validation",
    sorted(d.name for d in _all_runs
           if d.name in _completed and (d / "RUN_COMPLETE.json").exists()),
    sorted(_completed))

print("\n=== is every request-mean column labelled for what it is? ===")
# B8 documents that `predicted_per_second` divides n-1 tokens by the time for n,
# and the repository chose to keep the published figures rather than move dozens
# by a third of a percent. The fourth review's condition for that choice is that
# every place carrying the column says so, and that none of them is used for a
# thinking-off cross-arm comparison. That is now a rule, not a habit.
for _rel in ("README.md", "ERRATA.md", "v4_audit_2026_08_25/README.md"):
    _txt = (pathlib.Path(__file__).resolve().parents[1] / _rel) \
        .read_text(encoding="utf-8")
    _n = _txt.count("| request-mean |")
    chk(f"{_rel} publishes a request-mean column", _n > 0, True)
    chk(f"{_rel} says what request-mean is",
        "llama.cpp's own `predicted_per_second`, averaged" in _norm(_txt), True)
    chk(f"{_rel} says it must not carry a thinking-off cross-arm comparison",
        "must not carry a cross-arm comparison in the" in _norm(_txt), True)
    chk(f"{_rel} links the caveat to B8",
        "b8-every-request-mean-here-counts-one-token-fewer-than-it-timed" in _txt,
        True)
    # and the caveat must come before the first such table, not after it
    chk(f"{_rel} puts the caveat before the first such table",
        _txt.index("predicted_per_second`, averaged") < _txt.index("| request-mean |"),
        True)

print("\n=== run W's analysis plan was registered before its data ===")
# The same discipline PREREGISTERED_PREDICTION.md carries: an analysis chosen
# after seeing which answer the data gives is not evidence about the question.
# W exists to settle a disagreement this repository has been wrong about twice,
# so the estimators and the thresholds are committed first and the ordering is
# asserted, not asked for on trust.
_PW = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "PROSPECTIVE_ANALYSIS_PLAN_W.md"
chk("run W's plan is committed", _PW.is_file(), True)
_PWT = _norm(_PW.read_text(encoding="utf-8"))
chk("it names the disagreement it is there to settle",
    "+5.92 pp" in _PWT and "+8.65 pp" in _PWT, True)
chk("it names the alias V3 could not remove",
    "9 of 9" in _PWT, True)
chk("it fixes the estimand before the data",
    "absolute change in\npercentage points" in _PWT
    or "absolute change in percentage points" in " ".join(_PWT.split()), True)
chk("it says what a null result on the predecessor question would mean",
    "no detectable\npredecessor effect" in _PWT
    or "no detectable predecessor effect" in " ".join(_PWT.split()), True)
chk("it refuses in advance to pick a favourite among three readings",
    "not to\nchoose a favourite" in _PWT
    or "not to choose a favourite" in " ".join(_PWT.split()), True)
chk("it does not claim W will identify A16",
    "No claim will be made that W identifies" in " ".join(_PWT.split()), True)
if _HAS_GIT:
    # `--follow`, because the file was renamed. It was called PREREGISTERED_W.md
    # and said it had been committed before W's data existed, which git ancestry
    # cannot show and which the live PR body contradicts: the plan was finalized
    # at 360 of 500 arm-runs. Renaming it to a prospective plan must not lose
    # the ordering evidence, which is the part that IS checkable -- and a plain
    # `git log -- <new path>` returns the rename commit, not the original one.
    _pw_commit = _sp2.run(["git", "-C", str(_repo), "log", "--follow", "--format=%H",
                           "--", "v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W.md"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    _pw_commit = _pw_commit[-1] if _pw_commit else ""
    chk("the plan has a commit of its own", bool(_pw_commit), True)
    _wdirs = sorted((pathlib.Path(__file__).resolve().parents[1]
                     / "v4_audit_2026_08_25" / "data").glob("matrix_W_*"))
    _w_commit = ""
    if _wdirs:
        _w_commit = _sp2.run(["git", "-C", str(_repo), "log", "--format=%H", "-1", "--",
                              str(_wdirs[0].relative_to(_repo))],
                             capture_output=True, text=True).stdout.strip()
    if _w_commit:
        _anc = _sp2.run(["git", "-C", str(_repo), "merge-base", "--is-ancestor",
                         _pw_commit, _w_commit], capture_output=True)
        chk("and it is an ancestor of the commit that adds W's data",
            _anc.returncode == 0, True)
    else:
        # On disk but not yet committed is the state during the write-up. This
        # branch used to print instead of asserting, which made the number of
        # git-gated assertions depend on whether the data had been committed -
        # so committing it changed `_GITLESS_SKIPPED` from 10 to 11 and CI went
        # red on the commit that added the data. One chk either way keeps the
        # count a property of the code.
        chk("W's data is not committed yet, so the ancestry check is pending",
            _w_commit == "", True)

    # W2's plan makes a stronger claim than W's: not merely committed before
    # the data, but before the driver was invoked. Ancestry can show the first
    # half of that and nothing can show the second, which the plan itself says.
    _pw2 = _sp2.run(["git", "-C", str(_repo), "log", "--follow", "--format=%H",
                     "--", "v4_audit_2026_08_25/PROSPECTIVE_ANALYSIS_PLAN_W2.md"],
                    capture_output=True, text=True).stdout.strip().splitlines()
    _pw2 = _pw2[-1] if _pw2 else ""
    chk("W2's plan has a commit of its own", bool(_pw2), True)
    _w2dirs = sorted((pathlib.Path(__file__).resolve().parents[1]
                      / "v4_audit_2026_08_25" / "data").glob("matrix_W2_*"))
    _w2_commit = ""
    if _w2dirs:
        _w2_commit = _sp2.run(["git", "-C", str(_repo), "log", "--format=%H", "-1",
                               "--", str(_w2dirs[0].relative_to(_repo))],
                              capture_output=True, text=True).stdout.strip()
    if _w2_commit:
        chk("and it is an ancestor of the commit that adds W2's data",
            _sp2.run(["git", "-C", str(_repo), "merge-base", "--is-ancestor",
                      _pw2, _w2_commit], capture_output=True).returncode == 0,
            True)
    else:
        chk("W2's data is not committed yet, so its ancestry check is pending",
            _w2_commit == "", True)

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

# The same rule one level up. `chk(name, X, X)` cannot fail either, and this
# audit did not refuse it because neither side is a literal: `sorted(_B8)`
# against `sorted(_B8)` sat in B8 naming a population and asserting nothing
# about it, true for any `_B8` including an empty one. A name on both sides is
# not evidence that anything was compared.
_self_compare = []
for _n in _ast.walk(_tree):
    if not (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name)
            and _n.func.id == "chk" and len(_n.args) >= 3):
        continue
    if _ast.dump(_n.args[1]) == _ast.dump(_n.args[2]):
        _label = _n.args[0]
        _self_compare.append(getattr(_label, "value", None)
                             or _ast.unparse(_label)[:60])
print("\n=== A17's length-matched split table, and the cell that held the "
      "wrong quantity ===")
# Its `matrix_V_freerun` row published 11.90 pp in a column asking for the
# largest length-matching shift. 11.90 is run V's largest MODE contrast, hard
# cap against free running, which is a different measurement in a different
# entry; the length-matching shift is 14.02 pp on `spec-dflash-n4`. The cell
# looked right because the number was real, which is the hard kind to find.
_LMR = {_r["run"]: _r for _r in json.loads(
    (pathlib.Path(__file__).resolve().parent / "length_matching.json")
    .read_text(encoding="utf-8"))["runs"]}


def _lm_max_shift(run):
    return round(max(abs(_v.get("shift_pp") or 0.0)
                     for _v in _LMR[run]["arms"].values()), 2)


_lm_on = [_r for _r in _LMR.values() if _r["think"] != "off"]
_LMS_EXPECT = {
    "every thinking-on run with a computable comparison (31)": [
        len(_lm_on), 10, 20, 0.00],
    "matrix_V_hardcap, thinking off with the cap forced": [
        _LMR["matrix_V_hardcap_20260826_210956"]["prompts"], 0.00],
    "matrix_L_thinkoff": [10, 5, _lm_max_shift("matrix_L_thinkoff_20260826_032652")],
    "matrix_M3_thinkoff": [10, 5, _lm_max_shift("matrix_M3_thinkoff_20260826_081806")],
    "matrix_R_ext_thinkoff": [20, 6,
                              _lm_max_shift("matrix_R_ext_thinkoff_20260826_110747")],
    "D_master_matrix_think_off": [10, 5,
                                  _lm_max_shift("D_master_matrix_think_off")],
    "matrix_V_freerun": [10, 5, _lm_max_shift("matrix_V_freerun_20260826_210956")],
}
_LMST = _num_rows(_ER_LINES, "| run | thinking | prompts | length-matched |")
chk("A17 length-matched split: one row per run or class",
    sorted(_LMST), sorted(_LMS_EXPECT))
for _row, _want in sorted(_LMS_EXPECT.items()):
    _num_row_check(f"A17 split {_row}", _LMST[_row], _want)
chk("A17 split: every thinking-on run really is fully length-matched",
    sorted({_r["prompts"] == _r["length_matched_prompts"] for _r in _lm_on}),
    [True])
chk("A17 split: and they run either ten prompts or twenty",
    sorted({_r["prompts"] for _r in _lm_on}), [10, 20])
chk("A17 split: the thinking-off runs are the other six",
    sorted(_r["run"] for _r in _LMR.values() if _r["think"] == "off"),
    ["D_master_matrix_think_off", "matrix_L_thinkoff_20260826_032652",
     "matrix_M3_thinkoff_20260826_081806", "matrix_R_ext_thinkoff_20260826_110747",
     "matrix_V_freerun_20260826_210956", "matrix_V_hardcap_20260826_210956"])
chk("A17 split: the capped one is the only thinking-off run with no shift",
    [_r["run"] for _r in _LMR.values()
     if _r["think"] == "off" and _lm_max_shift(_r["run"]) == 0.0],
    ["matrix_V_hardcap_20260826_210956"])
# and the quantity that was in that cell, so the two cannot be confused again
_v_fr = {_a: _v["all_prompts_pct"] for _a, _v
         in _LMR["matrix_V_freerun_20260826_210956"]["arms"].items()}
_v_hc = {_a: _v["all_prompts_pct"] for _a, _v
         in _LMR["matrix_V_hardcap_20260826_210956"]["arms"].items()}
_v_mode = sorted(round(_v_hc[_a] - _v_fr[_a], 2) for _a in _v_fr)
chk("A17: 11.90 is run V's largest mode contrast, not its largest shift",
    (min(_v_mode), max(_v_mode)), (6.31, 11.90))
chk("A17: and the two really are different numbers for that run",
    _lm_max_shift("matrix_V_freerun_20260826_210956") != max(_v_mode), True)

# the paragraph under the table, which counts the comparisons and quotes six
# of them. It said the external drafter appears three times; `spec-draft-n8`
# appears in four of the five runs and `spec-draft-v1cfg` in a fifth.
_lm_off = [_r for _r in _LMR.values()
           if _r["think"] == "off" and not _r["run"].startswith("matrix_V_hardcap")]
_lm_pairs = [(_r["run"], _a, _v) for _r in _lm_off
             for _a, _v in sorted(_r["arms"].items())]
_lm_model = [_p for _p in _lm_pairs if not _p[1].startswith("ngram")]
_lm_ngram = [_p for _p in _lm_pairs if _p[1].startswith("ngram")]
chk("A17: five uncapped thinking-off runs", len(_lm_off), 5)
chk("A17: eighteen arm-vs-baseline comparisons in them", len(_lm_pairs), 18)
chk("A17: sixteen of them draft from a model", len(_lm_model), 16)
chk("A17: and every one of those moves the same way",
    sorted({_v["shift_pp"] > 0 for _, _, _v in _lm_model}), [True])
chk("A17: over the span the paragraph quotes",
    (round(min(_v["shift_pp"] for _, _, _v in _lm_model), 2),
     round(max(_v["shift_pp"] for _, _, _v in _lm_model), 2)), (2.52, 16.79))
_lm_ext = [_p for _p in _lm_model if _p[1].startswith("spec-draft")]
chk("A17: the external drafter appears five times, not three", len(_lm_ext), 5)
chk("A17: and the document says five",
    "each of its five appearances" in " ".join(_ER_LINES_TEXT.split()), True)
chk("A17: over the span it quotes for that arm",
    (round(min(_v["shift_pp"] for _, _, _v in _lm_ext), 1),
     round(max(_v["shift_pp"] for _, _, _v in _lm_ext), 1)), (2.5, 3.8))
chk("A17: the two exceptions are the n-gram arms and both are in run D",
    sorted((_a, _r.split("_")[0]) for _r, _a, _ in _lm_ngram),
    [("ngram-cache", "D"), ("ngram-mod-n24", "D")])
chk("A17: and they move the other way",
    sorted(round(_v["shift_pp"], 2) for _, _, _v in _lm_ngram), [-6.37, -0.17])
# the two run R prompts the entry opens with
_r_len = defaultdict(lambda: defaultdict(set))
for _f in sorted(glob.glob("v4_audit_2026_08_25/data/"
                           "matrix_R_ext_thinkoff_20260826_110747/*__rep*.json")):
    _b = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
    for _r in _b["rows"]:
        _r_len[_r["tag"]][_b["arm"]].add(_r["timings"]["predicted_n"])
_bash = {_a: sorted(_v) for _a, _v in _r_len["code_bash"].items()}
_rust = {_a: sorted(_v) for _a, _v in _r_len["code_rust"].items()}
chk("A17: run R's code_bash lengths, baseline against the three speculative arms",
    (_bash["baseline"], sorted(_v[0] for _a, _v in _bash.items() if _a != "baseline")),
    ([300], [187, 188, 188]))
chk("A17: and code_rust the other way",
    (_rust["baseline"], sorted({_v[0] for _a, _v in _rust.items() if _a != "baseline"})),
    ([203], [300]))
chk("A17: 38 % short in one direction",
    round(100 * (1 - min(_v[0] for _a, _v in _bash.items() if _a != "baseline")
                 / _bash["baseline"][0])), 38)
chk("A17: and 48 % long in the other",
    round(100 * (300 / _rust["baseline"][0] - 1)), 48)


print("\n=== C4b's thermal table, eighteen cells against the trace itself ===")
# The entry that corrects "stock clocks, measured once before the load"
# published its replacement as a table nothing read. `thermal_report.py` is
# the tool the entry names; every row is checked against the trace directly
# here, and against the tool's own output beside it, so a change in either
# that the other does not follow shows up as a disagreement.
# This trace is the `full` schema: values carry their units (` 87 %`, ` 350.00
# W`) and the throttle columns are ` Active` / ` Not Active` rather than bits.
# Reading them as bare numbers silently produced an empty sample set, which is
# a measurement of nothing that looks like a measurement of zero.
_C4B_TRACE = "v4_audit_2026_08_25/data/gpu_telemetry_20260825.csv"
_C4B_ALL = list(csv.DictReader(open(_C4B_TRACE, encoding="utf-8")))


def _c4b_num(cell):
    return float(re.search(r"-?[\d.]+", cell).group(0))


_c4b = [_r for _r in _C4B_ALL if _c4b_num(_r["util_pct"]) > 50]
_c4b_t = [int(_c4b_num(_r["temp_c"])) for _r in _c4b]
_c4b_g = [int(_c4b_num(_r["gfx_mhz"])) for _r in _c4b]


def _c4b_flag(key):
    return [_r for _r in _c4b if _r[key].strip() == "Active"]
_C4B_SLOWDOWN = 83                 # the card's documented slowdown point, not
                                   # a reading: marked in the table as such
_C4B_EXPECT = {
    "power.limit / power.default_limit / power.max_limit": [
        int(_c4b_num(_c4b[0]["power_limit_w"])),
        int(_c4b_num(_c4b[0]["power_default_w"])), 350],
    "GPU temperature": [min(_c4b_t), max(_c4b_t),
                        round(st.mean(_c4b_t), 1), _C4B_SLOWDOWN],
    "graphics clock": [min(_c4b_g), max(_c4b_g),
                       int(_c4b_num(_c4b[0]["gfx_max_mhz"])),
                       round(st.mean(_c4b_g))],
    "clocks_throttle_reasons.sw_power_cap": [
        len(_c4b_flag("thr_sw_power_cap")), len(_c4b)],
    "clocks_throttle_reasons.sw_thermal_slowdown": (
        [len(_c4b_flag("thr_sw_thermal")), len(_c4b)]
        + [int(_c4b_num(_r["temp_c"])) for _r in _c4b_flag("thr_sw_thermal")]),
    "clocks_throttle_reasons.hw_thermal_slowdown": (
        [len(_c4b_flag("thr_hw_thermal")), len(_c4b)]
        + [int(_c4b_num(_r["temp_c"])) for _r in _c4b_flag("thr_hw_thermal")]),
    "clocks_throttle_reasons.hw_power_brake_slowdown": [],
    "temperature.memory": [],
}
_C4BT = _num_rows(_ER_LINES, "| quantity | observed |")
chk("C4b thermal table: one row per quantity",
    sorted(_C4BT), sorted(_C4B_EXPECT))
for _q, _want in sorted(_C4B_EXPECT.items()):
    _num_row_check(f"C4b {_q}", _C4BT[_q], _want)
chk("C4b: the trace really has that many loaded samples, of that many",
    (len(_c4b), len(_C4B_ALL)), (1272, 1317))
chk("C4b: hw_power_brake never fired, which is why its row has no count",
    len(_c4b_flag("thr_hw_power_brake")), 0)
chk("C4b: the card does not report a memory junction temperature",
    sorted({_r.get("temp_mem_c", "N/A") for _r in _c4b}), ["N/A"])
chk("C4b: the three thermal-flag samples' clocks, as the sentence prints them",
    sorted(int(_c4b_num(_r["gfx_mhz"])) for _r in _c4b_flag("thr_sw_thermal")
           + _c4b_flag("thr_hw_thermal")), [1935, 1950, 1950])
chk("C4b: and none of them is near the run maximum's floor",
    max(_c4b_g), 1965)
chk("C4b: the slowdown point is marked as a datasheet figure",
    "throttle point \u2020" in _ER_LINES_TEXT, True)
chk("C4b: and the trace's own maximum is below it",
    max(_c4b_t) < _C4B_SLOWDOWN, True)


print("\n=== the three batching tables, thirty-nine cells nothing read ===")
# Run I's aggregate table, run K's collapse table, and the one under it that
# says the drafts do not get worse. Aggregate throughput is the harness's own
# `aggregate_tok_s`: generated tokens over wall clock for the whole ten-prompt
# set, which is the figure that moves under concurrency while the per-request
# decode rate does not.
def _aggregates(run_dir, arm):
    return [json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["aggregate_tok_s"]
            for _f in sorted(glob.glob(os.path.join(run_dir, f"{arm}__rep*.json")))]


def _ms(vals, dp=1):
    return [round(st.mean(vals), dp), round(st.stdev(vals), 2)]


_I_DIRS = {_c: f"v4_audit_2026_08_25/data/matrix_I2_conc{_c}_20260826_014750"
           for _c in (1, 4, 8)}
_IT_EXPECT = {}
for _c, _d in _I_DIRS.items():
    _bl, _spec = _aggregates(_d, "baseline"), _aggregates(_d, "spec-draft-n8")
    _IT_EXPECT[str(_c)] = ([_c] + _ms(_bl) + _ms(_spec)
                           + [round(st.mean(_spec) / st.mean(_bl), 2)])
_IT = _num_rows(_V4R_LINES, "| concurrency | no speculation | `spec-draft-n8` |")
chk("run I aggregate table: one row per level", sorted(_IT), sorted(_IT_EXPECT))
for _lvl, _want in sorted(_IT_EXPECT.items()):
    _num_row_check(f"run I aggregate c={_lvl}", _IT[_lvl], _want)

_K_DIRS = {1: "v4_audit_2026_08_25/data/matrix_K1_sweep_20260826_025615",
           4: "v4_audit_2026_08_25/data/matrix_K_conc4_20260826_025615",
           8: "v4_audit_2026_08_25/data/matrix_K_conc8_20260826_025615"}
_KB_EXPECT, _KL_EXPECT = {}, {}
for _c, _d in _K_DIRS.items():
    _bl, _spec = _aggregates(_d, "baseline"), _aggregates(_d, "spec-dflash-n4")
    _delta = round(100 * (st.mean(_spec) / st.mean(_bl) - 1), 1)
    # the c=1 row prints no spread; the other two do
    _KB_EXPECT["1 request in flight" if _c == 1 else f"{_c} in flight"] = (
        [_c, round(st.mean(_bl), 1), round(st.mean(_spec), 1), abs(_delta)]
        if _c == 1 else
        [_c] + _ms(_bl) + _ms(_spec) + [abs(_delta)])
    _dn = _da = _pn = 0
    for _f in sorted(glob.glob(os.path.join(_d, "spec-dflash-n4__rep*.json"))):
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            _dn += _r["draft_n"]
            _da += _r["draft_n_accepted"]
            _pn += _r["timings"]["predicted_n"]
    _KL_EXPECT[f"{_c} in flight"] = [_c, round(_dn / _pn, 3),
                                     round(100 * _da / _dn, 1),
                                     round(st.mean(_spec), 1)]
_KBT = _num_rows(_V4R_LINES, "| | no speculation | `spec-dflash-n4` | vs baseline |")
chk("run K batching table: one row per level", sorted(_KBT), sorted(_KB_EXPECT))
for _lvl, _want in sorted(_KB_EXPECT.items()):
    _num_row_check(f"run K batching {_lvl}", _KBT[_lvl], _want)
_KLT = _num_rows(_V4R_LINES,
                 "| level | drafted per generated token | acceptance | aggregate |")
chk("run K draft-quality table: one row per level",
    sorted(_KLT), sorted(_KL_EXPECT))
for _lvl, _want in sorted(_KL_EXPECT.items()):
    _num_row_check(f"run K draft quality {_lvl}", _KLT[_lvl], _want)
chk("run K: the aggregate column of the two tables is the same measurement",
    [_KL_EXPECT[_k][3] for _k in sorted(_KL_EXPECT)],
    [round(st.mean(_aggregates(_K_DIRS[_c], "spec-dflash-n4")), 1) for _c in (1, 4, 8)])
chk("run K: the drafts really do not get worse, the clock does",
    (_KL_EXPECT["8 in flight"][1] > _KL_EXPECT["1 in flight"][1],
     round(_KL_EXPECT["1 in flight"][2] - _KL_EXPECT["8 in flight"][2], 1)),
    (True, 4.4))


print("\n=== A14's M1-against-Q table, nine rows and none of them read ===")
# The entry's whole argument is that nothing differs between the two runs, and
# the table listing what was compared had no reader. Everything in it comes
# back from the manifests, the rep files and the telemetry except the two
# batch sizes, which were read from a server log this repository does not
# commit; those are held to the other place the file states them.
_M1D = "v4_audit_2026_08_25/data/matrix_M1_20260826_075816"
_QD = "v4_audit_2026_08_25/data/matrix_Q_q8_20260826_110747"
_M1M = json.loads((pathlib.Path(_M1D) / "manifest.json").read_text(encoding="utf-8"))
_QM = json.loads((pathlib.Path(_QD) / "manifest.json").read_text(encoding="utf-8"))
_M1R = json.loads((pathlib.Path(_M1D) / "spec-mtp-n4__rep0.json")
                  .read_text(encoding="utf-8"))
_QR = json.loads((pathlib.Path(_QD) / "spec-mtp-n4__rep0.json")
                 .read_text(encoding="utf-8"))


def _a14_draft_per_prompt(run_dir):
    """Counted draft tokens a prompt, per repeat, in the order the rows ran."""
    _fs = sorted(glob.glob(os.path.join(run_dir, "spec-mtp-n4__rep*.json")))
    _first = json.loads(pathlib.Path(_fs[0]).read_text(encoding="utf-8"))
    _per = defaultdict(int)
    for _f in _fs:
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            _per[_r["tag"]] += _r["draft_n"]
    return [_per[_r["tag"]] // len(_fs) for _r in _first["rows"]]


def _a14_acc(run_dir):
    _dn = _da = 0
    for _f in sorted(glob.glob(os.path.join(run_dir, "spec-mtp-n4__rep*.json"))):
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            _dn += _r["draft_n"]
            _da += _r["draft_n_accepted"]
    return round(100.0 * _da / _dn, 1)


def _a14_tel(pat, lo, hi):
    """Loaded telemetry samples in one wall-clock window, as (temp, SM clock).

    Its own copy of the reader: this file reuses short names freely and by
    here `_tel` is a string, `_c1` a dict and `_lm` a list. Every one of those
    was a TypeError before it was a wrong number, which is the lucky way round.
    """
    _out = []
    for _f in sorted(glob.glob(f"v4_audit_2026_08_25/data/{pat}")):
        for _r in csv.reader(open(_f, encoding="utf-8")):
            if len(_r) != 9 or not _r[1].isdigit():
                continue
            _m = re.search(r"(\d{2}):(\d{2}):(\d{2})", _r[0])
            if _m and lo <= _m.group(0) <= hi and int(_r[1]) > 50:
                _out.append((int(_r[3]), int(_r[5])))
    return _out


_a14_m1_tel = _a14_tel("gpu_telemetry_M_*.csv", "08:00:00", "08:15:00")
_a14_q_tel = _a14_tel("gpu_telemetry_chain2_*.csv", "11:33:00", "11:52:00")
chk("A14: both telemetry windows still hold loaded samples",
    (len(_a14_m1_tel) > 0, len(_a14_q_tel) > 0), (True, True))
_a14_place = re.search(r"placement (\d+)/(\d+), (\d+)/(\d+)",
                       pathlib.Path("v4_audit_2026_08_25/data/matrix_M.log")
                       .read_text(encoding="utf-8"))
_A14_EXPECT = {
    "server binary sha256": [],
    "drafter file sha256": [],
    "recorded argv": [len(_M1R["argv"])],
    "what the memory fitter chose": [_M1M["ctx"]]
        + [int(_x) for _x in _a14_place.groups()] + [2048, 512],
    "draft tokens per prompt": _a14_draft_per_prompt(_M1D),
    "draft acceptance": [_a14_acc(_M1D), _a14_acc(_QD)],
    "temperature under load": [round(st.mean([_x[0] for _x in _a14_m1_tel]), 1),
                               round(st.mean([_x[0] for _x in _a14_q_tel]), 1)],
    "SM clock under load": [round(st.mean([_x[1] for _x in _a14_m1_tel])),
                            round(st.mean([_x[1] for _x in _a14_q_tel]))],
    "the no-speculation baseline beside it": [round(_pool_dir(_M1D, "baseline"), 1),
                                              round(_pool_dir(_QD, "baseline"), 1)],
}
_A14T = _num_rows(_ER_LINES, "| checked | M1 | run Q |")
chk("A14 comparison table: one row per thing checked",
    sorted(_A14T), sorted(_A14_EXPECT))
for _row, _want in sorted(_A14_EXPECT.items()):
    if _row == "what the memory fitter chose":
        # the row prints n_ctx, then the placement, then the two batch sizes
        _num_row_check(f"A14 {_row}", _A14T[_row],
                       [_want[0]] + _want[1:5] + _want[5:])
    else:
        _num_row_check(f"A14 {_row}", _A14T[_row], _want)
chk("A14: the two runs ran the same binary and the same MTP head",
    (_M1M["server_sha256"] == _QM["server_sha256"],
     _M1M["mtp_sha256"] == _QM["mtp_sha256"]), (True, True))
chk("A14: and the table's abbreviated hashes are those files'",
    (_M1M["server_sha256"].startswith("b6a5c490"),
     _M1M["mtp_sha256"].startswith("5b1e4937")), (True, True))
# It said "byte-identical" and one of the thirty tokens differs: the listening
# port, 18861 against 18902. Irrelevant to the measurement and still not what
# the word means, so the row says what is actually true.
_a14_argv_diff = [_k for _k, (_x, _y) in enumerate(zip(_M1R["argv"], _QR["argv"]))
                  if _x != _y]
chk("A14: the recorded argv differs in exactly one position",
    len(_a14_argv_diff), 1)
chk("A14: and that position is the port the server listened on",
    _M1R["argv"][_a14_argv_diff[0] - 1], "--port")
chk("A14: which is what the row now says",
    "identical but for the listening port" in _ER_LINES_TEXT, True)
chk("A14: and the draft tokens a prompt really are identical",
    _a14_draft_per_prompt(_M1D) == _a14_draft_per_prompt(_QD), True)
chk("A14: the two batch sizes are marked as not re-derivable here",
    "`n_batch 2048, n_ubatch 512` \u2020" in _ER_LINES_TEXT, True)
chk("A14: and they agree with the only other place this file states them",
    "`n_batch` 2048, `n_ubatch` 512"
    in " ".join(_ER_LINES_TEXT.split()), True)


print("\n=== the README's length-matched table, eight cells nothing read ===")
# The paragraph around it was corrected once already - it used to say
# acceptance falls with thinking off, which was the short outputs and not the
# workload. The table that carries the correction had no reader.
# Computed from the run data at full precision, not from
# `length_matching.json`, whose fields are already rounded to two places:
# ngram-cache's length-matched acceptance is 1.8469 %, the file says 1.85, and
# rounding that again gives 1.9 against the published 1.8. That is A12's double
# rounding, and it appeared here in the checker written to catch it.
def _lm_cells(run_dir):
    """(present prompts, length-matched prompts, per-arm figures) for one run."""
    _man = json.loads((pathlib.Path(run_dir) / "manifest.json")
                      .read_text(encoding="utf-8"))
    _rows = defaultdict(lambda: defaultdict(list))
    for _f in sorted(glob.glob(os.path.join(run_dir, "*__rep*.json"))):
        _r = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
        if _r.get("crashed"):
            continue
        for _x in _r["rows"]:
            _rows[_x["tag"]][_r["arm"]].append(
                (_x["predicted_n"], _x["predicted_ms"],
                 _x["draft_n"], _x["draft_n_accepted"]))
    _arms = [_a for _a in _man["arms"] if _a != "baseline"]
    _present = [_t for _t, _d in _rows.items()
                if "baseline" in _d and all(_a in _d for _a in _arms)]
    _matched = [_t for _t in _present
                if len({_r[0] for _a in _rows[_t] for _r in _rows[_t][_a]}) == 1]

    def _p(tags, arm):
        return (1000 * sum(_r[0] for _t in tags for _r in _rows[_t][arm])
                / sum(_r[1] for _t in tags for _r in _rows[_t][arm]))

    def _acc(tags, arm):
        _d = sum(_r[2] for _t in tags for _r in _rows[_t][arm])
        if not _d:                     # an arm that never drafted, such as a
            return None                # no-speculation control under another name
        return 100 * sum(_r[3] for _t in tags for _r in _rows[_t][arm]) / _d

    _out = {}
    for _a in _arms:
        _out[_a] = {
            "pct": 100 * (_p(_present, _a) / _p(_present, "baseline") - 1),
            "acc": _acc(_present, _a),
            "m_pct": 100 * (_p(_matched, _a) / _p(_matched, "baseline") - 1)
                     if _matched else None,
            "m_acc": _acc(_matched, _a) if _matched else None}
    return len(_present), len(_matched), _out


_LM_ON_N, _LM_ON_M, _LM_ON = _lm_cells(
    "v4_audit_2026_08_25/data/C_master_matrix_think_on")
_LM_OFF_N, _LM_OFF_M, _LM_OFF = _lm_cells(
    "v4_audit_2026_08_25/data/D_master_matrix_think_off")


def _lm_row(arm, key):
    _mk = "m_pct" if key == "pct" else "m_acc"
    return [round(_LM_ON[arm][key], 1), round(_LM_OFF[arm][key], 1),
            round(_LM_OFF[arm][_mk], 1)]


_LMT = _num_rows_seq(_RM_LINES, "| | thinking on | thinking off, as above |")
chk("README length-matched table: four rows", len(_LMT), 4)
chk("README length-matched table: the rows it names",
    [_c[0].strip("`* ") for _c, _ in _LMT],
    ["draft model `n_max 8`".strip("`* "), "its acceptance",
     "ngram-cache", "its acceptance"])
_LM_WANT = [
    ("draft model n_max 8", [8] + _lm_row("spec-draft-n8", "pct")),
    ("its acceptance, draft", _lm_row("spec-draft-n8", "acc")),
    ("ngram-cache", _lm_row("ngram-cache", "pct")),
    ("its acceptance, ngram-cache", _lm_row("ngram-cache", "acc")),
]
for (_lbl, _want), (_cells, _nums) in zip(_LM_WANT, _LMT):
    _num_row_check(f"README length-matched {_lbl}", _nums,
                   [abs(_x) for _x in _want])
chk("README length-matched: thinking on had nothing to match, so all ten count",
    (_LM_ON_N, _LM_ON_M), (10, 10))
chk("and thinking off matched five of its ten", (_LM_OFF_N, _LM_OFF_M), (10, 5))
chk("README length-matched: the paragraph's own two acceptance figures",
    (f"{round(_LM_OFF['spec-draft-n8']['m_acc'], 1)} % against "
     f"{round(_LM_ON['spec-draft-n8']['acc'], 1)} % with thinking on")
    in " ".join(_ROOT_TEXT.replace("*", "").split()), True)


print("\n=== the BOS-override table, published twice and read in neither ===")
# A2's central claim: the drafted and accepted totals are identical whether the
# vocabulary is matched by override or reached through the translation
# fallback. Eight numbers a copy, in two documents, and both copies were
# unread; `--override-kv` is what the row labels differ by.
_BOS_ARMS = [("A_bcb5eeb64_legacy", "draft-max8-translate"),
             ("A_bcb5eeb64_legacy", "draft-max8-matched"),
             ("B_master_3737e4137", "draft-max8-translate"),
             ("B_master_3737e4137", "draft-max8-matched")]
_bos = []
for _run, _arm in _BOS_ARMS:
    _rows = []
    for _f in sorted(glob.glob(
            f"v4_audit_2026_08_25/data/{_run}/{_arm}__rep*.json")):
        _rows += json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]
    _bos.append([round(st.mean(_x["timings"]["predicted_per_second"]
                               for _x in _rows), 1),
                 sum(_x["draft_n"] for _x in _rows),
                 sum(_x["draft_n_accepted"] for _x in _rows)])
for _doc, _lines in (("README", _RM_LINES), ("ERRATA", _ER_LINES)):
    _rows_pub = _num_rows_seq(
        _lines, "| binary | arm | request-mean | drafted / accepted |")
    chk(f"{_doc} BOS table: four rows", len(_rows_pub), 4)
    for _k, ((_cells, _nums), _want) in enumerate(zip(_rows_pub, _bos)):
        _num_row_check(f"{_doc} BOS row {_k + 1} ({_cells[1][:28]})",
                       _nums, _want)
    chk(f"{_doc} BOS table: the two binaries, in order",
        [_c[0].replace("master ", "").strip("`* ") for _c, _ in _rows_pub],
        ["bcb5eeb64", "bcb5eeb64", "3737e4137", "3737e4137"])
chk("A2: the drafted and accepted totals really are equal within each binary",
    [(_bos[0][1], _bos[0][2]) == (_bos[1][1], _bos[1][2]),
     (_bos[2][1], _bos[2][2]) == (_bos[3][1], _bos[3][2])], [True, True])
chk("A2: and the two documents publish the same four rows",
    [_n for _, _n in _num_rows_seq(
        _RM_LINES, "| binary | arm | request-mean | drafted / accepted |")],
    [_n for _, _n in _num_rows_seq(
        _ER_LINES, "| binary | arm | request-mean | drafted / accepted |")])


print("\n=== run I's acceptance-under-batching table, which nothing read ===")
# Upstream issue #27572 reports acceptance collapsing to zero under `-np N`.
# The table that says it did not happen here was itself unchecked.
_I2_EXPECT = {}
for _c in (1, 4, 8):
    _d = f"v4_audit_2026_08_25/data/matrix_I2_conc{_c}_20260826_014750"
    _dn, _da, _ratio, _zero, _rows = [], [], [], [], []
    for _f in sorted(glob.glob(os.path.join(_d, "spec-draft-n8__rep*.json"))):
        _r = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]
        _dn.append(sum(_x["draft_n"] for _x in _r))
        _da.append(sum(_x["draft_n_accepted"] for _x in _r))
        _ratio.append(round(100.0 * _da[-1] / _dn[-1], 1))
        _zero.append(sum(1 for _x in _r if _x["draft_n"] == 0))
        _rows.append(len(_r))
    chk(f"run I c={_c}: the three repeats all ran ten requests",
        sorted(set(_rows)), [10])
    chk(f"run I c={_c}: no request went without a draft", sorted(set(_zero)), [0])
    _dspan = ([min(_dn)] if len(set(_dn)) == 1 else [min(_dn), max(_dn)])
    _aspan = ([min(_da)] if len(set(_da)) == 1 else [min(_da), max(_da)])
    _rspan = ([min(_ratio)] if len(set(_ratio)) == 1
              else [min(_ratio), max(_ratio)])
    _I2_EXPECT[f"c={_c}"] = ([_c] + _dspan + _aspan + _rspan
                             + [_zero[0], _rows[0]])
_I2T = _num_rows(_V4R_LINES, "| level | drafted | accepted | counted ratio |")
chk("run I table: one row per concurrency level",
    sorted(_I2T), sorted(_I2_EXPECT))
for _lvl, _want in sorted(_I2_EXPECT.items()):
    _num_row_check(f"run I {_lvl}", _I2T[_lvl], _want)
chk("run I: the acceptance really is flat across the three levels",
    round(max(_x for _v in _I2_EXPECT.values() for _x in _v if 20 < _x < 40)
          - min(_x for _v in _I2_EXPECT.values() for _x in _v if 20 < _x < 40), 1),
    1.4, 0.05)
chk("run I: and the issue it answers is named",
    "issue #27572" in _V4R_TEXT, True)


print("\n=== the archives' own tables: v3's ranking and Exp 2's variance ===")
# The v3 cross-method ranking in CHANGELOG.md is a mean over five per-prompt
# `llama-cli` logs a config, and the logs are committed, so it re-derives.
_V3LOG = (pathlib.Path(__file__).resolve().parents[1] / "v3_dflash_2026_05_07"
          / "data" / "out_20260507_183341")
_V3GEN = re.compile(r"Generation:\s*([\d.]+) t/s")


def _v3_mean(sub):
    _v = []
    for _f in sorted((_V3LOG / sub).glob("p*.log")):
        _m = _V3GEN.findall(_f.read_text(errors="replace"))
        if _m:
            _v.append(float(_m[-1]))
    return st.mean(_v), len(_v)


_V3_ROWS = {"no spec (baseline)": "01_baseline",
            "Oleg draft-spec max=32": "03_oleg_draft_2_32",
            "Oleg draft-spec max=16": "04_oleg_draft_2_16",
            "DFlash --draft-max 16": "05_dflash_max16",
            "DFlash --draft-max 8": "06_dflash_max8",
            "DFlash --draft-max 4": "07_dflash_max4"}
_v3_base, _v3_n = _v3_mean("01_baseline")
chk("v3 ranking: five prompt logs behind every row",
    sorted({_v3_mean(_d)[1] for _d in _V3_ROWS.values()}), [5])
_V3T = _num_rows(_CH_LINES, "| method | tok/s (mean) | vs baseline |")
chk("v3 ranking table: one row per method", sorted(_V3T), sorted(_V3_ROWS))
for _label, _sub in sorted(_V3_ROWS.items()):
    _m, _ = _v3_mean(_sub)
    _want = [round(_m, 1)]
    # the draft length is in the label of the four that name one
    _want = ([float(_x) for _x in re.findall(r"(?<![\w.])\d+(?![\w.])", _label)]
             + _want)
    if _sub != "01_baseline":
        _want.append(abs(round(100 * (_m / _v3_base - 1), 1)))
    _num_row_check(f"v3 ranking {_label}", _V3T[_label], _want)
chk("v3 ranking: the best DFlash row is the one the table marks",
    max(_V3_ROWS, key=lambda _k: _v3_mean(_V3_ROWS[_k])[0] if "DFlash" in _k else -1),
    "DFlash --draft-max 8")

# Exp 2's variance decomposition. Its cells come from the RAW list in
# `exp2_codejson_n3/results.json`, not from the summary file beside it, whose
# `configs` hold a different set of numbers entirely.
_X2 = json.loads((pathlib.Path(__file__).resolve().parents[1] / "v2_3090_followup"
                  / "exp2_codejson_n3" / "results.json").read_text(encoding="utf-8"))


def _x2_sd(name):
    _raw = _X2["configs"][name]["raw"]          # (trial, prompt, wall_ms, tok_s)
    _bt, _bp = defaultdict(list), defaultdict(list)
    for _t, _pr, _w, _v in _raw:
        _bt[_t].append(_v)
        _bp[_pr].append(_v)
    return [round(st.stdev([_r[3] for _r in _raw]), 3),
            round(st.stdev([st.mean(_v) for _v in _bt.values()]), 3),
            round(st.mean([st.stdev(_v) for _v in _bp.values()]), 3)]


_X2T = _num_rows(_ER_LINES, "| config | SD over all 15 cells |")
chk("Exp 2 variance table: one row per configuration",
    sorted(_X2T), sorted(_X2["configs"]))
for _cfg in sorted(_X2["configs"]):
    _num_row_check(f"Exp 2 variance {_cfg}", _X2T[_cfg], _x2_sd(_cfg))
chk("Exp 2: fifteen cells a configuration, three trials by five prompts",
    sorted({len(_X2["configs"][_c]["raw"]) for _c in _X2["configs"]}), [15])
chk("Exp 2: and the mean the paragraph quotes beside the spread",
    (round(st.mean([_r[3] for _r
                    in _X2["configs"]["02_oleg_draft_2_32"]["raw"]]), 2),
     _x2_sd("02_oleg_draft_2_32")[0]), (66.57, 7.566))


print("\n=== the EAGLE3 cross-check, which is a quotation and still checks ===")
# Not this repository's data: it is quoted from an upstream PR comment. What
# can be checked is that the table is internally consistent - every speedup is
# its own row's ratio - and that the range the paragraph draws from it is the
# range of those speedups.
_EGT = _num_rows_seq(_ER_LINES,
                     "| prompt | baseline | EAGLE3, draft 8 | speedup |")
chk("EAGLE3 table: three prompts", len(_EGT), 3)
_eg_speedups = []
for _cells, _nums in _EGT:
    _base, _d8, _s8, _d3, _s3 = _nums
    chk(f"EAGLE3 {_cells[0][:28]}: draft 8 speedup is its own row's ratio",
        round(float(_d8) / float(_base), 2), float(_s8))
    chk(f"EAGLE3 {_cells[0][:28]}: and draft 3's",
        round(float(_d3) / float(_base), 2), float(_s3))
    _eg_speedups += [float(_s8), float(_s3)]
chk("EAGLE3: the net loss the paragraph quotes is the span of those six",
    (round(100 * (1 - max(_eg_speedups))), round(100 * (1 - min(_eg_speedups)))),
    (29, 54))
chk("EAGLE3: and the table says whose measurement it is",
    "pull/18039" in _ER_LINES_TEXT, True)


print("\n=== A11's two identity tables, and the thermals across the day ===")
_A11J = _num_rows(_ER_LINES, "| arm | token streams identical to no speculation |")
chk("A11 run J identity table: four arms", len(_A11J), 4)
for _arm, _n in (("spec-dflash-n4", 3), ("spec-dflash-n8", 0),
                 ("spec-dflash-n16", 3), ("spec-draft-n8", 0)):
    _num_row_check(f"A11 run J identity {_arm}", _A11J[_arm], [_n, 30])
_A11X = _num_rows(_ER_LINES, "| run | output length | token streams identical")
chk("A11 cross-run identity table: four rows", len(_A11X), 4)
for _row, _want in (("J", [300, 6, 120]), ("K1", [300, 0, 180]),
                    ("L, thinking on", [300, 0, 200]),
                    ("L, thinking off", [96, 85, 200])):
    _num_row_check(f"A11 cross-run identity {_row}", _A11X[_row], _want)

_TH_RUNS = {"I + J": "gpu_telemetry_IJ_*", "K": "gpu_telemetry_K_*",
            "L": "gpu_telemetry_L_*"}
_TH_EXPECT = {}
for _lbl, _pat in _TH_RUNS.items():
    _f = sorted(glob.glob(f"v4_audit_2026_08_25/data/{_pat}.csv"))[0]
    _rows = [_r for _r in csv.reader(open(_f, encoding="utf-8"))
             if len(_r) == 9 and _r[1].isdigit() and int(_r[1]) > 50]
    _t = [int(_r[3]) for _r in _rows]
    _g = [int(_r[5]) for _r in _rows]
    _h = len(_g) // 2
    _drift = round(100 * (st.mean(_g[_h:]) / st.mean(_g[:_h]) - 1), 2)
    _TH_EXPECT[_lbl] = ([len(_rows), min(_t), max(_t), round(st.mean(_t), 1),
                         min(_g), max(_g), round(st.mean(_g))]
                        + ([] if _lbl == "I + J" else [abs(_drift)]))
_THT = _num_rows(_V4R_LINES, "| run | loaded samples | temperature | SM clock |")
chk("v4 thermal table: one row per run", sorted(_THT), sorted(_TH_EXPECT))
for _lbl, _want in sorted(_TH_EXPECT.items()):
    _num_row_check(f"v4 thermals {_lbl}", _THT[_lbl], _want)
chk("v4 thermals: the I+J row prints no drift, and the trace is one run's",
    _THT["I + J"][-1], "1939")


print("\n=== the twelve measurements of one arm, and the memory policy ===")
# README's twelve-cell table. Its inclusion rule is in the sentence above it:
# the same memory policy, the same ten prompts, thinking on, one at a time, on
# 2026-08-26. Applying that rule to the data returns exactly those twelve.
_TWELVE = []
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/")):
    _d = _d.rstrip("/")
    _fs = glob.glob(f"{_d}/spec-dflash-n2__rep*.json")
    if not _fs:
        continue
    _m = json.loads((pathlib.Path(_d) / "manifest.json").read_text(encoding="utf-8"))
    if _m.get("fit_target") != "3072" or _m.get("ignore_eos"):
        continue
    if _m.get("think") not in (None, "on", "think_on"):
        continue
    if not _m["created"].startswith("2026-08-26"):
        continue
    if len({_r["tag"] for _r
            in json.loads(pathlib.Path(_fs[0]).read_text(encoding="utf-8"))["rows"]}) != 10:
        continue
    _TWELVE.append((_m["created"][11:16], os.path.basename(_d),
                    100 * (_pool_dir(_d, "spec-dflash-n2")
                           / _pool_dir(_d, "baseline") - 1),
                    _pool_dir(_d, "baseline")))
_TWELVE.sort()
chk("README twelve-measurement table: twelve of them", len(_TWELVE), 12)
_TWT = _num_rows_seq(_RM_LINES, "| | | | | | |")
chk("README twelve-measurement table: two rows of six", len(_TWT), 2)
_tw_pub = [_x for _cells, _nums in _TWT for _x in _nums]
_tw_want = []
for _t, _n, _v, _b in _TWELVE:
    _tw_want += [round(abs(_v), 1), int(_t[:2]), int(_t[3:])]
chk("README twelve measurements: every shift and every clock time",
    [float(_x) for _x in _tw_pub], [float(_x) for _x in _tw_want])
_tw_v = [_v for _, _, _v, _ in _TWELVE]
_tw_b = [_b for _, _, _, _b in _TWELVE]
chk("README twelve: the range and SD the sentence quotes",
    (round(max(_tw_v) - min(_tw_v), 1), round(st.stdev(_tw_v), 1)), (9.4, 2.9))
chk("README twelve: the baseline band and its CV",
    (round(min(_tw_b), 2), round(max(_tw_b), 2),
     round(100 * st.stdev(_tw_b) / st.mean(_tw_b), 2)), (115.72, 117.25, 0.42))

_BE_LINES = (pathlib.Path(__file__).resolve().parents[1]
             / "BENCHMARK_ENV.md").read_text(encoding="utf-8").splitlines()
# BENCHMARK_ENV's memory-policy table says it is derived from the manifests,
# so it is. It had run N in the `-fit on` row when N ran pinned at 16384, and
# it had no row mentioning run W at all.
_POL = defaultdict(set)
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/")):
    _d = _d.rstrip("/")
    _name = os.path.basename(_d)
    if not glob.glob(f"{_d}/*__rep*.json"):
        continue
    _m = json.loads((pathlib.Path(_d) / "manifest.json").read_text(encoding="utf-8"))
    _ca = _m.get("common_args", [])
    _ngl = _ca[_ca.index("-ngl") + 1] if "-ngl" in _ca else "unset"
    _ctx = _ca[_ca.index("-c") + 1] if "-c" in _ca else None
    _ft = _m.get("fit_target") or ("1024" if _m.get("fit") else None)
    # the start-up checks carry the run letter, which is how the table names
    # run J: one matrix directory, `matrix_J2`, and a `smoke_J` beside it
    _tag = re.match(r"(?:matrix|smoke)_([A-Za-z]+[0-9]*)", _name)
    _POL[(_ngl, _ctx, _ft)].add(_tag.group(1) if _tag else _name.split("_")[0])
chk("memory policy: the manifests fall into four groups", len(_POL), 4)
_POLT = _num_rows(_BE_LINES, "| runs | `-ngl` | `-c` | `--fit-target` | why |")
chk("memory policy table: four rows", len(_POLT), 4)
_pol_named = {tuple(sorted(_k.replace(" ", "").split(","))): _v
              for _k, _v in _POLT.items()}
chk("memory policy: every row names exactly the runs with that policy",
    sorted(tuple(sorted(_v)) for _v in _POL.values()),
    sorted(_k for _k in _pol_named))
# the `why` column carries numbers too: the fit-target the row is contrasted
# against, and run J's measured headroom
_ij_rows = [_r for _r in csv.reader(open(sorted(glob.glob(
    "v4_audit_2026_08_25/data/gpu_telemetry_IJ_*.csv"))[0], encoding="utf-8"))
    if len(_r) == 9 and _r[1].isdigit()]
_CARD_MIB = 24576
_headroom = _CARD_MIB - max(int(_r[2]) for _r in _ij_rows)
chk("memory policy: run J's measured headroom on a 24 GiB card", _headroom, 630)
_POL_WHY = {("999", "16384", None): [],
            ("unset", "16384", "1024"): [],
            ("unset", "8192", "2048"): [1024],
            ("unset", "8192", "3072"): [2048, _headroom, 3072]}
for _key, _grp in sorted(_POL.items()):
    _row = _pol_named[tuple(sorted(_grp))]
    _want = (([999] if _key[0] != "unset" else []) + [int(_key[1])]
             + ([int(_key[2])] if _key[2] else []) + _POL_WHY[_key])
    _num_row_check(f"memory policy {sorted(_grp)[0]}...", _row, _want)
chk("memory policy: run N is in the pinned group, where its manifest puts it",
    sorted(_k for _k, _v in _POL.items() if "N" in _v)[0][0], "999")
chk("memory policy: and run W is in the 3072 group",
    sorted(_k for _k, _v in _POL.items() if "W" in _v)[0][2], "3072")


print("\n=== the two file maps, which count what is on disk ===")
# Both are lists of paths with counts in them, and both counts had gone stale:
# the v4 map's lead-in said 41 directories while the row under it said 65, and
# the README's map said 62 v2 logs where the three `v2_*` directories hold 61,
# one of which is the verbose trace it counts separately.
_RUN_DIRS = [_d for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/"))
             if glob.glob(f"{_d}*__rep*.json")]
_SMOKE = [_d for _d in _RUN_DIRS if "smoke" in _d]
chk("v4 file map: the lead-in counts the directories under data/",
    f"There are\n{len(_RUN_DIRS)}, of which {len(_RUN_DIRS) - len(_SMOKE)} are "
    f"runs and three are start-up checks." in _V4R_TEXT, True)
_FM = _num_rows(_V4R_LINES, "| Path | Contents |")
chk("v4 file map: its first row counts the same directories and every arm-run",
    [float(_x) for _x in _FM["data/<run>/"]],
    [float(len(_RUN_DIRS)),
     float(sum(len(glob.glob(f"{_d}*__rep*.json")) for _d in _RUN_DIRS))])
_v2_logs = sorted(_p for _d in ("v2_controls", "v2_master_cross_check",
                                "v2_oleg_suggestions")
                  for _p in glob.glob(f"v2_3090_followup/{_d}/**/*.log",
                                      recursive=True))
_v2_verbose = [_p for _p in _v2_logs if os.path.basename(_p) == "verbose.log"]
chk("README data map: the v2 log count is what the three directories hold",
    f"{len(_v2_logs) - len(_v2_verbose)} v2 raw `llama-cli` logs + one "
    f"`--verbose` trace" in _ROOT_TEXT, True)
chk("README data map: and exactly one of them is the verbose trace",
    len(_v2_verbose), 1)
chk("README data map: the v1 label count is summary.csv's",
    f"v1 raw per-request JSON, {len({_r['config'] for _r in _V1CSV})} run labels"
    in _ROOT_TEXT, True)


print("\n=== the A4 terms and counters, in the four places they appear ===")
# One reconstruction, four tables: ERRATA's counter table and its cost table,
# and the README's copy of each. Every figure is already derived above; none of
# the four tables had a reader.
_A4_COUNTERS = {
    "server draft_n_accepted / draft_n": [_A4["counter"], _A4["acc_tokens"],
                                          round(100.0, 1)],
    "drafter #acc tokens / #gen tokens": [
        _A4["acc_tokens"], _A4["gen_tokens"],
        round(100 * _A4["acc_tokens"] / _A4["gen_tokens"], 1)],
    "drafter #acc drafts / #gen drafts": [
        _A4["full"], _A4["gen_drafts"],
        round(100 * _A4["full"] / _A4["gen_drafts"], 1)],
}
_A4C_ER = _num_rows(_ER_LINES, "| counter | value | meaning |")
chk("A4 counter table (ERRATA): three counters",
    sorted(_A4C_ER), sorted(_A4_COUNTERS))
for _k, _want in sorted(_A4_COUNTERS.items()):
    _num_row_check(f"A4 counter (ERRATA) {_k}", _A4C_ER[_k], _want)
_A4C_RM = _num_rows(_RM_LINES, "| counter | value | what it counts |")
chk("A4 counter table (README): the same three, one relabelled",
    len(_A4C_RM), 3)
chk("A4 counter table (README): and the same numbers in the same order",
    [_x for _v in _A4C_RM.values() for _x in _v],
    [_x for _v in _A4C_ER.values() for _x in _v])

_A4_TERMS = [
    [round(_A4["gen_ms"], 1), round(100 * _A4["gen_ms"] / _A4["wall_ms"], 1)],
    [_A4["ckpts"], round(_A4["ckpt_mib"], 1), round(_A4["gib_written"], 2)],
    [_A4["restores"], round(_A4["gib_read"], 2)],
    [_A4["partial"], _A4["attempts"],
     round(100 * _A4["partial"] / _A4["attempts"], 1)],
]
# spelled out rather than looped: the census resolves a reader's document from
# the variable passed to it, and a loop variable tells it nothing
for _doc, _rows in (
        ("ERRATA", _num_rows_seq(_ER_LINES, "| term | value | source |")),
        ("README", _num_rows_seq(_RM_LINES, "| term | measured |"))):
    chk(f"A4 cost table ({_doc}): four terms", len(_rows), 4)
    for _k, ((_cells, _nums), _want) in enumerate(zip(_rows, _A4_TERMS)):
        _num_row_check(f"A4 cost ({_doc}) {_cells[0][:30]}", _nums, _want)

print("\n=== B8's understatement table, which is one over n ===")
_B8T = _num_rows(_ER_LINES, "| tokens generated | understated by |")
chk("B8 table: four lengths", len(_B8T), 4)
# the row key IS the first cell, so deriving the second from it checks the
# cell against itself: perturbing 300 to 307 leaves 100/307 rounding to the
# published 0.33 and the check passes. The lengths are pinned to what the
# repository measured instead, and three of the four are in the data.
_B8_LENGTHS = [20, 187, 300, 1000]
chk("B8 table: the lengths it tabulates",
    sorted(int(_k) for _k in _B8T), _B8_LENGTHS)
# Was "300 is the cap every v4 request runs to", checked over ONE file of one
# thinking-on run, where it is true. Across the corpus it is not: 9391 of
# 30344 rows stop short, every one of them thinking-off, which is exactly the
# condition the documents attach to this figure ("on a run where every request
# hits the same cap"). Measured over every committed row, so the condition is
# supported rather than assumed.
_b8cap_at = _b8cap_short = 0
_b8cap_think = set()
for _d in sorted((pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25"
                  / "data").glob("*")):
    if not _d.is_dir() or not (_d / "manifest.json").exists():
        continue
    _mth = str(json.loads((_d / "manifest.json").read_text(encoding="utf-8"))
               .get("think"))
    for _f in _d.glob("*__rep*.json"):
        for _r in json.loads(_f.read_text(encoding="utf-8")).get("rows") or []:
            _n300 = (_r.get("timings") or {}).get("predicted_n")
            if _n300 is None:
                continue
            if _n300 == 300:
                _b8cap_at += 1
            else:
                _b8cap_short += 1
                _b8cap_think.add(_mth)
chk("B8: committed rows that reach the 300-token cap, and those that stop short",
    (_b8cap_at, _b8cap_short, _b8cap_at + _b8cap_short), (20953, 9391, 30344))
chk("B8: and every row that stops short is a thinking-off one, which is the "
    "condition the documents attach",
    sorted(_b8cap_think), ["off"])
chk("B8: the run the old one-file check looked at does run every request to it",
    sorted({_r["timings"]["predicted_n"] for _f in
            sorted(glob.glob("v4_audit_2026_08_25/data/matrix_O2_latin_*/"
                             "baseline__rep0.json"))
            for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]}),
    [300])
chk("B8: 1000 is v1's long-output cap",
    sorted({int(_r["max_tokens"]) for _r in _V1CSV
            if _r["config"] == "baseline-1000tok"}), [1000])
chk("B8: 187 is the shortest run R generated on the prompt A17 opens with",
    min(_r["timings"]["predicted_n"] for _f in
        sorted(glob.glob("v4_audit_2026_08_25/data/matrix_R_ext_thinkoff_*/"
                         "spec-*__rep*.json"))
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]
        if _r["tag"] == "code_bash"), 187)
for _n, _row in sorted(_B8T.items(), key=lambda _x: int(_x[0])):
    _num_row_check(f"B8 at {_n} tokens", _row,
                   [int(_n), round(100.0 / int(_n), 2)])
chk("B8: the 300-token figure is the one the note beside every table quotes",
    f"{round(100.0 / 300, 2)} % at 300 tokens" in " ".join(_ER_LINES_TEXT.split()),
    True)


# named here because three sections below use them and the order they run in
# is the order they were written, which is not the order they depend on
_JD = "v4_audit_2026_08_25/data/matrix_J2_20260826_014750"
_M1D2 = "v4_audit_2026_08_25/data/matrix_M1_20260826_075816"
_M3D = "v4_audit_2026_08_25/data/matrix_M3_thinkoff_20260826_081806"
_RD = "v4_audit_2026_08_25/data/matrix_R_ext_thinkoff_20260826_110747"

print("\n=== the threshold scorecard and the two run tables beside it ===")
_SCT = _num_rows(_V4R_LINES, "| family | sign predicted correctly |")
_SC_ROWS = {"self-speculative (DFlash and MTP)": "self",
            "drafter-free n-gram": "ngram",
            "external 0.8 B drafter": "external"}
chk("threshold scorecard table: three families and a total",
    sorted(_SCT), sorted(list(_SC_ROWS) + ["all"]))
for _label, _fam in sorted(_SC_ROWS.items()):
    _got, _tot = _SCORECARD[_fam]
    _num_row_check(f"threshold scorecard {_label}", _SCT[_label],
                   ([0.8] if "0.8 B" in _label else []) + [_got, _tot])
_num_row_check("threshold scorecard all", _SCT["all"],
               [sum(_v[0] for _v in _SCORECARD.values()),
                sum(_v[1] for _v in _SCORECARD.values())])
chk("threshold scorecard: the families partition the kept arm-runs",
    sum(_v[1] for _v in _SCORECARD.values()), len(_SC_KEPT))

# the two misses the text calls informative, and run C's sweep of the drafter
_MISS = _num_rows(_V4R_LINES, "| arm | acceptance | measured | why it is interesting |")
chk("threshold misses table: two arms", len(_MISS), 2)
_m_mtp = [_x for _x in _SC_KEPT if _x[2] == "spec-mtp-n4"
          and (_x[3] >= _BRK) != (_x[5] > 0)][0]
_num_row_check("threshold miss spec-mtp-n4, thinking off",
               _MISS["spec-mtp-n4, thinking off"],
               [round(_m_mtp[3], 1), abs(round(_m_mtp[5], 1)),
                round(_m_mtp[3] - _BRK, 1)])
_m_d1 = [_x for _x in _SC_KEPT if _x[2] == "spec-draft-n1"
         and (_x[3] >= _BRK) != (_x[5] > 0)]
chk("threshold miss spec-draft-n1: it misses in three runs", len(_m_d1), 3)
# the row quotes run O's head-to-head, which is the largest of the three and
# the run the section is about; the other two are O2 and O3 at -74.8 and -75.0
_m_o = [_x for _x in _m_d1 if _x[1].startswith("matrix_O_headtohead")][0]
_num_row_check("threshold miss spec-draft-n1", _MISS["spec-draft-n1"],
               [round(_m_o[3], 1), round(_m_o[4], 1), abs(round(_m_o[5], 1))])
chk("threshold miss spec-draft-n1: the three runs it misses in",
    sorted(_x[1].split("_2026")[0] for _x in _m_d1),
    ["matrix_O2_latin", "matrix_O3_latin", "matrix_O_headtohead"])

_CSWEEP = _num_rows(_V4R_LINES, "| arm | acceptance | measured | threshold says |")
chk("run C drafter sweep table: three arms", len(_CSWEEP), 3)
# run C is the one directory whose name in the counter dump (`matrix_C_...`)
# is not the name on disk (`C_master_matrix_think_on`), so the scorecard's own
# join drops it and its acceptance has to be read from the dump directly
_CD = "v4_audit_2026_08_25/data/C_master_matrix_think_on"
_c_base = _pool_dir(_CD, "baseline")
_c_acc = {_k[1]: _v2["server_pct"] for _k, _v2 in _cc2.items()
          if _k[0].startswith("matrix_C_")}
chk("run C sweep: the counter dump carries run C under its own name",
    len(_c_acc) > 0, True)
for _n_max in (1, 2, 4):
    _arm = f"spec-draft-n{_n_max}"
    _delta = 100 * (_pool_dir(_CD, _arm) / _c_base - 1)
    _num_row_check(f"run C sweep {_arm}", _CSWEEP[_arm],
                   [round(_c_acc[_arm], 1), abs(round(_delta, 1))])
chk("run C sweep: the threshold calls the first two faster and they are not",
    [(_c_acc[f"spec-draft-n{_n}"] >= _BRK,
      _pool_dir(_CD, f"spec-draft-n{_n}") > _c_base) for _n in (1, 2, 4)],
    [(True, False), (True, False), (False, False)])


print("\n=== four more v4 tables: the head, the batching pair, and the sets ===")
_Q8D = "v4_audit_2026_08_25/data/matrix_M1_20260826_075816"
_Q4D = "v4_audit_2026_08_25/data/matrix_M4_q4km_20260826_081806"


def _q_pair(run_dir, arm):
    _v = 100 * (st.mean(_aggregates(run_dir, arm))
                / st.mean(_aggregates(run_dir, "baseline")) - 1)
    _dn = _da = 0
    for _f in sorted(glob.glob(os.path.join(run_dir, f"{arm}__rep*.json"))):
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            _dn += _r["draft_n"]
            _da += _r["draft_n_accepted"]
    return [abs(round(_v, 1)), round(100 * _da / _dn, 1)]


_QT = _num_rows(_V4R_LINES, "| arm | Q8_0 drafter | Q4_K_M drafter |")
chk("run Q drafter-quantisation table: two arms", len(_QT), 2)
for _arm in ("spec-mtp-n2", "spec-mtp-n4"):
    _num_row_check(f"run Q {_arm}", _QT[_arm],
                   _q_pair(_Q8D, _arm) + _q_pair(_Q4D, _arm))

_MB = _num_rows(_V4R_LINES, "| requests in flight | DFlash `n4` | MTP `n2` |")
chk("run M batching table: three levels", sorted(_MB), ["1", "4", "8"])
_MB_DIRS = {1: ("v4_audit_2026_08_25/data/matrix_K1_sweep_20260826_025615",
                "v4_audit_2026_08_25/data/matrix_M1_20260826_075816"),
            4: ("v4_audit_2026_08_25/data/matrix_K_conc4_20260826_025615",
                "v4_audit_2026_08_25/data/matrix_M2_conc4_20260826_081806"),
            8: ("v4_audit_2026_08_25/data/matrix_K_conc8_20260826_025615",
                "v4_audit_2026_08_25/data/matrix_M2_conc8_20260826_081806")}
for _c, (_kd, _md) in sorted(_MB_DIRS.items()):
    _dfl = 100 * (st.mean(_aggregates(_kd, "spec-dflash-n4"))
                  / st.mean(_aggregates(_kd, "baseline")) - 1)
    _mtp = 100 * (st.mean(_aggregates(_md, "spec-mtp-n2"))
                  / st.mean(_aggregates(_md, "baseline")) - 1)
    _num_row_check(f"run M batching c={_c}", _MB[str(_c)],
                   [_c, abs(round(_dfl, 1)), abs(round(_mtp, 1))])
chk("run M batching: only one of the two falls off a cliff",
    (100 * (st.mean(_aggregates(_MB_DIRS[8][0], "spec-dflash-n4"))
            / st.mean(_aggregates(_MB_DIRS[8][0], "baseline")) - 1) < -50,
     100 * (st.mean(_aggregates(_MB_DIRS[8][1], "spec-mtp-n2"))
            / st.mean(_aggregates(_MB_DIRS[8][1], "baseline")) - 1) > -20), (True, True))

_JKT = _num_rows(_V4R_LINES, "| `n_max` | run J | run K |")
chk("J-against-K table: the two shared draft lengths", sorted(_JKT), ["4", "8"])
for _n_max in (4, 8):
    _j = 100 * (st.mean(_aggregates(_JD, f"spec-dflash-n{_n_max}"))
                / st.mean(_aggregates(_JD, "baseline")) - 1)
    _k = 100 * (st.mean(_aggregates(_MB_DIRS[1][0], f"spec-dflash-n{_n_max}"))
                / st.mean(_aggregates(_MB_DIRS[1][0], "baseline")) - 1)
    _num_row_check(f"J against K at n_max {_n_max}", _JKT[str(_n_max)],
                   [_n_max, abs(round(_j, 1)), abs(round(_k, 1))])

_LOFF = "v4_audit_2026_08_25/data/matrix_L_thinkoff_20260826_032652"
_SETT = _num_rows(_V4R_LINES, "| arm | v1 ten (run M3) | extended twenty (run R) |")
chk("prompt-set table: three arms", len(_SETT), 3)
chk("prompt set: run M3 really does not carry the external drafter",
    glob.glob(os.path.join(_M3D, "spec-draft-n8__rep*.json")), [])
for _arm, _v1 in (("spec-dflash-n2", _M3D), ("spec-mtp-n2", _M3D),
                  ("spec-draft-n8", _LOFF)):
    _a = 100 * (_pool_dir(_v1, _arm) / _pool_dir(_v1, "baseline") - 1)
    _b = 100 * (_pool_dir(_RD, _arm) / _pool_dir(_RD, "baseline") - 1)
    _num_row_check(f"prompt set {_arm}", _SETT[_arm],
                   [abs(round(_a, 1)), abs(round(_b, 1))])
chk("prompt set: the row that is not M3's is marked",
    "`spec-draft-n8` \u2020" in _V4R_TEXT, True)
chk("prompt set: and the -75.1 % it used to carry was run L thinking ON",
    round(100 * (_pool_dir("v4_audit_2026_08_25/data/"
                           "matrix_L_thinkon_20260826_032652", "spec-draft-n8")
                 / _pool_dir("v4_audit_2026_08_25/data/"
                             "matrix_L_thinkon_20260826_032652", "baseline") - 1), 1),
    -75.1, 0.05)
chk("prompt set: the spread the sentence now quotes",
    sorted(abs(round(abs(100 * (_pool_dir(_v1, _a2) / _pool_dir(_v1, "baseline") - 1))
                     - abs(100 * (_pool_dir(_RD, _a2) / _pool_dir(_RD, "baseline") - 1)), 1))
           for _a2, _v1 in (("spec-dflash-n2", _M3D), ("spec-mtp-n2", _M3D),
                            ("spec-draft-n8", _LOFF))),
    [0.1, 1.8, 3.2])


print("\n=== A5's empty-content split, and two run tables in other files ===")
_a5_empty = defaultdict(int)
_a5_total = defaultdict(int)
for _f in sorted(glob.glob("results/*.json")):
    for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
        _a5_total[_r["tag"]] += 1
        _a5_empty[_r["tag"]] += not (_r.get("content_head") or "").strip()
for _f in sorted(glob.glob("results/verify/*.json")):
    for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
        _a5_total[_r["tag"]] += 1
        _a5_empty[_r["tag"]] += not (_r.get("content_head") or "").strip()
_A5T = _num_rows_seq(_ER_LINES, "| prompt | requests with empty `message.content` |")
chk("A5 empty-content table: four rows", len(_A5T), 4)
_a5_by = sorted(_a5_empty, key=lambda _t: (-_a5_empty[_t], _t))
_num_row_check("A5 reasoning", _A5T[0][1],
               [_a5_empty["reasoning"], _a5_total["reasoning"]])
_num_row_check("A5 code_small", _A5T[1][1],
               [_a5_empty["code_small"], _a5_total["code_small"]])
_a5_mid = ["short_q", "medium_chat", "medium_rec", "long_explain",
           "multi_turn_1", "multi_turn_2", "zh_cn"]
chk("A5: the seven prompts the third row groups all read the same",
    sorted({(_a5_empty[_t], _a5_total[_t]) for _t in _a5_mid}), [(15, 19)])
_num_row_check("A5 the seven at 15 of 19", _A5T[2][1],
               [_a5_empty["short_q"], _a5_total["short_q"]])
_num_row_check("A5 short_greet", _A5T[3][1],
               [_a5_empty["short_greet"], _a5_total["short_greet"]])
chk("A5: the totals the paragraph quotes",
    (sum(_a5_empty.values()), sum(_a5_total.values()),
     round(100 * sum(_a5_empty.values()) / sum(_a5_total.values()), 1)),
    (144, 190, 75.8))

_RT_J = _num_rows(_RT_LINES, "| arm | aggregate | vs no speculation |")
_j_base = st.mean(_aggregates(_JD, "baseline"))
chk("RETEST run J table: four rows", len(_RT_J), 4)
for _arm in ("spec-dflash-n4", "spec-dflash-n8", "spec-dflash-n16"):
    _v = st.mean(_aggregates(_JD, _arm))
    _num_row_check(f"RETEST run J {_arm}", _RT_J[_arm],
                   [round(_v, 1), abs(round(100 * (_v / _j_base - 1), 1))])
_num_row_check("RETEST run J no speculation", _RT_J["no speculation"],
               [round(_j_base, 1)])

_M3T = _num_rows(_V4R_LINES,
                 "| | thinking on (run M1, aggregate) | "
                 "thinking off (run M3, 5 repeats, pooled) |")
chk("run M think-off table: three arms", len(_M3T), 3)
for _arm in ("spec-mtp-n2", "spec-dflash-n2", "spec-mtp-n4"):
    # the left column is run M1's AGGREGATE, the metric its own table uses;
    # the right is run M3's pooled rate. The header says so now.
    _on = 100 * (st.mean(_aggregates(_M1D2, _arm))
                 / st.mean(_aggregates(_M1D2, "baseline")) - 1)
    _off = 100 * (_pool_dir(_M3D, _arm) / _pool_dir(_M3D, "baseline") - 1)
    _dn = _da = 0
    for _f in sorted(glob.glob(os.path.join(_M3D, f"{_arm}__rep*.json"))):
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            _dn += _r["draft_n"]
            _da += _r["draft_n_accepted"]
    _num_row_check(f"run M think-off {_arm}", _M3T[_arm],
                   [abs(round(_on, 1)), abs(round(_off, 1)),
                    round(100 * _da / _dn, 1)])
chk("run M: the table names the metric in each column",
    "thinking on (run M1, aggregate)" in _V4R_TEXT
    and "thinking off (run M3, 5 repeats, pooled)" in _V4R_TEXT, True)
chk("run M: and the flip holds on pooled rates in both columns, as it says",
    (round(100 * (_pool_dir(_M1D2, "spec-dflash-n2")
                  / _pool_dir(_M1D2, "baseline") - 1), 1),
     round(100 * (_pool_dir(_M1D2, "spec-mtp-n2")
                  / _pool_dir(_M1D2, "baseline") - 1), 1)), (26.7, 22.1))
chk("run M: the ranking really flips between the two workloads",
    (_pool_dir(_M1D2, "spec-dflash-n2") / _pool_dir(_M1D2, "baseline")
     > _pool_dir(_M1D2, "spec-mtp-n2") / _pool_dir(_M1D2, "baseline"),
     _pool_dir(_M3D, "spec-mtp-n2") / _pool_dir(_M3D, "baseline")
     > _pool_dir(_M3D, "spec-dflash-n2") / _pool_dir(_M3D, "baseline")),
    (True, True))


print("\n=== A13's split table and B7's fp16-KV control ===")
_A13T = _num_rows(_ER_LINES,
                  "| | arm-runs | largest / smallest gap between the two counters |")
chk("A13 split table: two groups", len(_A13T), 2)
_num_row_check("A13 split, no checkpoint",
               _A13T["speculative checkpoints never taken"],
               [len(_A13_NOCK),
                round(max(abs(_r["server_pct"] - _r["drafter_pct"])
                          for _r in _A13_NOCK), 2)])
_num_row_check("A13 split, checkpointing",
               _A13T["speculative checkpoints taken"],
               [len(_A13_CK),
                round(min(abs(_r["server_pct"] - _r["drafter_pct"])
                          for _r in _A13_CK), 2)])
chk("A13: the gap between the groups, as the sentence under it states",
    round(min(abs(_r["server_pct"] - _r["drafter_pct"]) for _r in _A13_CK)
          - max(abs(_r["server_pct"] - _r["drafter_pct"])
                for _r in _A13_NOCK), 2), 0.20)

_B7_ARMS = {"baseline (q8_0 KV)": "baseline",
            "baseline-kvfp16": "baseline-kvfp16",
            "ngram-cache (q8_0 KV)": "ngram-cache",
            "ngram-cache-kvfp16": "ngram-cache-kvfp16"}
_B7D = "v4_audit_2026_08_25/data/C_master_matrix_think_on"
_b7_base = _pool_dir(_B7D, "baseline")
_B7T = _num_rows(_ER_LINES, "| arm | pooled tok/s | vs `baseline` |")
chk("B7 fp16-KV table: four arms", sorted(_B7T), sorted(_B7_ARMS))
for _label, _arm in sorted(_B7_ARMS.items()):
    _v = _pool_dir(_B7D, _arm)
    _want = [round(_v, 1)] + ([] if _arm == "baseline"
                              else [abs(round(100 * (_v / _b7_base - 1), 1))])
    # the two KV-fp16 rows carry a 16 in their arm name, which is a value
    _num_row_check(f"B7 {_label}", _B7T[_label], _want)
chk("B7: fp16 KV is faster with no speculation and not with it",
    (_pool_dir(_B7D, "baseline-kvfp16") > _b7_base,
     _pool_dir(_B7D, "ngram-cache-kvfp16") < _pool_dir(_B7D, "ngram-cache")),
    (True, True))


print("\n=== B8's row census over every committed request ===")
# The claim is about which formula `predicted_per_second` follows, over every
# committed row of every committed arm-run file. Counting it is the check.
_b8_n1 = _b8_n = _b8_neither = _b8_rows = 0
_b8_files = 0
for _f in sorted(glob.glob("v4_audit_2026_08_25/data/*/*__rep*.json")):
    _b = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
    _b8_files += 1
    for _r in _b["rows"]:
        _t = _r["timings"]
        # not `_n`/`_ms`: `_ms` is the mean-and-SD helper by this point in the
        # file, and rebinding it turned the next section into a TypeError
        _bn, _bms, _got = (_t["predicted_n"], _t["predicted_ms"],
                           _t["predicted_per_second"])
        _b8_rows += 1
        if abs(1000 * (_bn - 1) / _bms - _got) < 1e-6:
            _b8_n1 += 1
        elif abs(1000 * _bn / _bms - _got) < 1e-6:
            _b8_n += 1
        else:
            _b8_neither += 1
_B8R = _num_rows(_ER_LINES, "| what the server reported | rows |")
chk("B8 row census: three outcomes", len(_B8R), 3)
def _grouped(n):
    """The digits of `n` the way this repository writes them.

    Four figures run together (`6265`); five and above take a thin space
    (`12 780`), which anything reading the cell sees as two numbers. Grouping
    unconditionally splits `6265` into 6 and 265, which is how this check
    failed the first time.
    """
    return [n] if n < 10000 else [int(_g) for _g in f"{n:,}".split(",")]


for _k, _want in (("1000 × (n − 1) / predicted_ms", [1000, 1] + _grouped(_b8_n1)),
                  ("1000 × n / predicted_ms", [1000] + _grouped(_b8_n)),
                  ("neither", _grouped(_b8_neither))):
    _num_row_check(f"B8 rows following {_k}", _B8R[_k], _want)
chk("B8: the totals the paragraph above it quotes",
    (_b8_rows, _b8_files), (30344, 3005))
# and the paragraph is read, not assumed. The pair above compares the tree with
# two literals in this file and never opens ERRATA, so the lead-in sat at
# "18 344 rows of 1805 files" through two runs with nothing to say otherwise.
_b8_lead = [_l for _l in _ER_LINES
            if _l.startswith("Across all **") and "committed arm-run files" in _l]
chk("B8: exactly one paragraph states them", len(_b8_lead), 1)
chk("B8: and it states the totals this run measured",
    [float(_x) for _x in _NUM_RE.findall(_b8_lead[0])],
    [float(_g) for _g in _grouped(_b8_rows) + _grouped(_b8_files)])


print("\n=== run W2: the same square at the power to answer it ===")
# Every figure below is recomputed from the committed arm-runs through the same
# functions the tool publishes from. The three tables W2 added are parsed cell
# by cell here because the census counts a table nobody parses as a hole, and
# three new holes is what it reported the moment they were written.
_W2 = sorted(_DATA.glob("matrix_W2_s*_20260830_220554"))
chk("W2: twelve sessions committed", len(_W2), 12)
chk("W2: a hundred arm-runs in each",
    sorted({len(_co.arm_runs(str(_d))) for _d in _W2}), [100])
chk("W2: every arm preceded by every other exactly once, read from t_start",
    sorted({_co.is_balanced(_co.arm_runs(str(_d)))[0] for _d in _W2}), [True])
chk("W2: it is a separate invocation, so nothing pools it with W",
    sorted({_d.name.split("_")[1] for _d in _W2 + _W}), ["W", "W2"])

_w2rep = _co.report([str(_d) for _d in _W2], True)
_wrep = _co.report([str(_d) for _d in _W], True)


def _iv(rep, key, arm, places=2):
    """(mean, lo, hi) of one arm, rounded the way the tables print it."""
    _v = rep[key][arm]
    return tuple(round(_v[_k], places)
                 for _k in ("mean_delta_pct" if "shift" not in key
                            else "mean_delta_pp", "lo", "hi"))


chk("W2: the pre-registered matched contrast for spec-dflash-n2",
    _iv(_w2rep, "across_sessions_matched", "spec-dflash-n2"),
    (-0.14, -0.68, 0.41))
chk("W: the same estimand, for the row above it in the table",
    _iv(_wrep, "across_sessions_matched", "spec-dflash-n2"),
    (-1.05, -2.97, 0.86))
chk("W2: and the grouped contrast the plan names as a sensitivity",
    _iv(_w2rep, "across_sessions", "spec-dflash-n2"), (-0.21, -0.78, 0.35))
chk("W2: the session SD of the primary estimand",
    round(st.stdev(
        _w2rep["across_sessions_matched"]["spec-dflash-n2"]["per_session"]), 3),
    0.858)

# the boundary-inclusive sensitivity is a different population, so it is a
# second invocation of the tool rather than a field of the first
_w2bnd = _co.report([str(_d) for _d in _W2], True, True)
chk("W2: the boundary-inclusive sensitivity, and it says which it is",
    (_iv(_w2bnd, "across_sessions_matched", "spec-dflash-n2"),
     _w2bnd["row_boundaries_included"]), ((-0.19, -0.75, 0.38), True))

# the published quantity, which is what the 2.4 pp is quoted in
chk("W2: the predecessor's effect on shift_pp, and W's beside it",
    (_iv(_w2rep, "across_sessions_mode_shift", "spec-dflash-n2"),
     _iv(_wrep, "across_sessions_mode_shift", "spec-dflash-n2")),
    ((0.49, -0.80, 1.77), (1.52, -0.98, 4.02)))
chk("W2 excludes a 2.4 pp predecessor effect on it and W does not, "
    "which is the whole reason the run exists",
    (_w2rep["across_sessions_mode_shift"]["spec-dflash-n2"]["hi"] < 2.4,
     _wrep["across_sessions_mode_shift"]["spec-dflash-n2"]["hi"] < 2.4),
    (True, False))

# the outcome the plan named in advance and did not expect
_w2zero = sorted(a for a, v in _w2rep["across_sessions_matched"].items()
                 if v["lo"] > 0 or v["hi"] < 0)
chk("W2: which arms' matched intervals exclude zero",
    _w2zero, ["spec-draft-n8", "spec-mtp-n2-cap"])
chk("W2: and neither of them is the estimand the plan is about",
    "spec-dflash-n2" in _w2zero, False)

# --- the three tables, parsed cell by cell -------------------------------
_W2T = _num_rows(_ER_LINES, "| run | sessions | matched contrast |")
chk("A17's W-versus-W2 table: two rows", sorted(_W2T), ["W", "W2"])
for _r, _rep in (("W", _wrep), ("W2", _w2rep)):
    _m, _lo, _hi = _iv(_rep, "across_sessions_matched", "spec-dflash-n2")
    _n = 12 if _r == "W2" else 5
    _num_row_check(f"A17 matched contrast row {_r}", _W2T[_r],
                   [_n, abs(_m), abs(_lo), abs(_hi)])
def _rows_under(lines, header_startswith, n):
    """The n data rows of one table, raw. The extractor drops the sign and the
    yes/no cells have no number at all, so both are read off the row text; a
    literal that looks like a table header would be counted as an unread table
    by the census, which is why this takes a position rather than a needle."""
    i = next(i for i, _l in enumerate(lines)
             if _l.strip().startswith(header_startswith))
    return lines[i + 2:i + 2 + n]


# `_num_rows` returns magnitudes, so a flipped sign in either table would pass
# every check above it. The rows are read a second time with the minus sign
# normalised, which is what `_pr_table` does for the body's copy.
def _signed_row(raw):
    return [float(_x) for _x in
            _NUM_RE.findall(_norm_early(raw).split("|", 2)[2])]


for _hdr2, _key2, _pl in (("| run | sessions | matched contrast |",
                           "across_sessions_matched", 2),
                          ("| run | sessions | `shift_pp` after capped",
                           "across_sessions_mode_shift", 2)):
    _rows2 = _rows_under(_ER_LINES, _hdr2, 2)
    for _raw2, _rep2 in zip(_rows2, (_wrep, _w2rep)):
        _m2, _lo2, _hi2 = _iv(_rep2, _key2, "spec-dflash-n2")
        chk(f"A17 {_key2}: the row prints the signs too",
            _signed_row(_raw2)[1:], [_m2, _lo2, _hi2])

_W2S = _num_rows(_ER_LINES, "| run | sessions | `shift_pp` after capped")
chk("A17's shift_pp table: two rows", sorted(_W2S), ["W", "W2"])
for _r, _rep in (("W", _wrep), ("W2", _w2rep)):
    _m, _lo, _hi = _iv(_rep, "across_sessions_mode_shift", "spec-dflash-n2")
    _n = 12 if _r == "W2" else 5
    _num_row_check(f"A17 shift_pp row {_r}", _W2S[_r],
                   [_n, abs(_m), abs(_lo), abs(_hi)])
chk("A17: the shift_pp table says which interval contains 2.4",
    [_r.split("|")[-2].strip().replace("*", "") for _r in
     _rows_under(_ER_LINES, "| run | sessions | `shift_pp` after capped", 2)],
    ["yes", "no"])

_PRW2 = {_r[0]: [float(_x) for _x in _NUM_RE.findall(" ".join(_r[1:]))]
         for _r in _pr_table("| quantity | W, 5 sessions | W2, 12 sessions |")}
chk("PR body: the W-versus-W2 table has five rows", len(_PRW2), 5)
# `_pr_table` normalises the minus sign, so these are SIGNED and a flip fails
_num_row_check("PR body matched contrast row",
               _PRW2["matched contrast, spec-dflash-n2"],
               [-1.05, -2.97, 0.86, -0.14, -0.68, 0.41])
_num_row_check("PR body grouped contrast row",
               _PRW2["grouped contrast, the same arm"],
               [-1.20, -2.61, 0.22, -0.21, -0.78, 0.35])
_num_row_check("PR body boundary row", _PRW2["including row boundaries"],
               [-0.19, -0.75, 0.38])
_num_row_check("PR body shift_pp row",
               _PRW2["shift_pp after capped minus after free"],
               [1.52, -0.98, 4.02, 0.49, -0.80, 1.77])
_num_row_check("PR body session SD row",
               _PRW2["session SD of the matched contrast"], [1.543, 0.858])


print("\n=== the -fit on control, and the CHANGELOG's run summary ===")
_FITT = _num_rows_seq(_V4R_LINES, "| control | aggregate |")
chk("J fit control table: three rows", len(_FITT), 3)
_fit_pinned = _aggregates("v4_audit_2026_08_25/data/matrix_I2_conc1_20260826_014750",
                   "baseline")
_fit_on = _aggregates("v4_audit_2026_08_25/data/matrix_J2_20260826_014750", "baseline")
_num_row_check("J fit control, pinned", _FITT[0][1],
               [999, 1] + _ms(_fit_pinned, 2))
_num_row_check("J fit control, -fit on", _FITT[1][1], _ms(_fit_on, 2))
_num_row_check("J fit control, difference", _FITT[2][1],
               [abs(round(100 * (st.mean(_fit_on) / st.mean(_fit_pinned) - 1), 2))])
chk("J fit control: the control is not handicapped",
    abs(100 * (st.mean(_fit_on) / st.mean(_fit_pinned) - 1)) < 0.1, True)

_CHR = _num_rows(_CH_LINES, "| run | question | answer |")
chk("CHANGELOG run summary: seven runs", sorted(_CHR),
    ["I", "J", "K", "L", "M", "N", "O"])
# every figure in it is quoted from a section that derives it, so the check is
# that the two agree rather than that the changelog is right on its own
_K1D = "v4_audit_2026_08_25/data/matrix_K1_sweep_20260826_025615"
_k1_base = _pool_dir(_K1D, "baseline")
_k1 = {_x: 100 * (_pool_dir(_K1D, f"spec-dflash-n{_x}") / _k1_base - 1)
       for _x in (1, 2, 3, 4, 6, 8)}
_k1_plat = sorted(_x for _x in _k1 if _k1[_x] >= max(_k1.values()) - 1)
chk("CHANGELOG run K: the plateau its row names, within a point of the best",
    _k1_plat, [2, 3, 4])
chk("CHANGELOG run K: and a cliff after it",
    [_k1[_x] < 0 for _x in (6, 8)], [True, True])
chk("CHANGELOG run M: its row quotes run M1's own aggregate, not run O's",
    round(100 * (st.mean(_aggregates(_M1D2, "spec-mtp-n2"))
                 / st.mean(_aggregates(_M1D2, "baseline")) - 1), 1), 18.6, 0.05)
chk("CHANGELOG run M: which really is its best MTP arm on that metric",
    max(("spec-mtp-n1", "spec-mtp-n2", "spec-mtp-n4", "spec-mtp-n8"),
        key=lambda _a: st.mean(_aggregates(_M1D2, _a))), "spec-mtp-n2")
chk("CHANGELOG run M: and +17.5 to +21.8 was run O's two metrics for that arm",
    (round(100 * (st.mean(_aggregates("v4_audit_2026_08_25/data/"
                                      "matrix_O_headtohead_20260826_081806",
                                      "spec-mtp-n2"))
                  / st.mean(_aggregates("v4_audit_2026_08_25/data/"
                                        "matrix_O_headtohead_20260826_081806",
                                        "baseline")) - 1), 1),
     round(100 * (_pool_dir("v4_audit_2026_08_25/data/"
                            "matrix_O_headtohead_20260826_081806", "spec-mtp-n2")
                  / _pool_dir("v4_audit_2026_08_25/data/"
                              "matrix_O_headtohead_20260826_081806",
                              "baseline") - 1), 1)), (17.5, 21.8))
for _run, _needle in (("I", "+64 %"), ("J", "+18.7 %"), ("L", "+14.1 %"),
                      ("N", "0.0 %"), ("O", "a factor of five")):
    chk(f"CHANGELOG run {_run}: its answer is quoted elsewhere too",
        " ".join(_norm(_needle).split()) in " ".join(_norm(_V4R_TEXT).split())
        or " ".join(_norm(_needle).split()) in " ".join(_norm(_ROOT_TEXT).split()),
        True)
_num_row_check("CHANGELOG run I", _CHR["I"], [64, 8])
_num_row_check("CHANGELOG run J", _CHR["J"], [18.7])
_num_row_check("CHANGELOG run K", _CHR["K"], [2, 4])
_num_row_check("CHANGELOG run L", _CHR["L"], [4, 4, 14.1])
_num_row_check("CHANGELOG run M", _CHR["M"], [18.6, 2])
_num_row_check("CHANGELOG run N", _CHR["N"], [0.0])
_num_row_check("CHANGELOG run O", _CHR["O"], [])


print("\n=== run I's concurrency check, eighteen arm-runs of it ===")
# `max_in_flight` is the client's own count from the request timestamps, and
# the paragraph is explicit that it is not the server's batch width. The table
# exists to rule out the negative case, so every cell of it matters.
_CONC_EXPECT = {}
for _c in (1, 4, 8):
    _d = f"v4_audit_2026_08_25/data/matrix_I2_conc{_c}_20260826_014750"
    _seen = [json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["max_in_flight"]
             for _f in sorted(glob.glob(os.path.join(_d, "*__rep*.json")))]
    _CONC_EXPECT[f"c={_c}"] = [_c, _c] + _seen
_CONCT = _num_rows(_V4R_LINES,
                   "| level | requested | observed client requests in flight |")
chk("run I concurrency table: one row per level",
    sorted(_CONCT), sorted(_CONC_EXPECT))
for _lvl, _want in sorted(_CONC_EXPECT.items()):
    _num_row_check(f"run I concurrency {_lvl}", _CONCT[_lvl], _want)
chk("run I: eighteen arm-runs across the three levels",
    sum(len(_v) - 2 for _v in _CONC_EXPECT.values()), 18)
chk("run I: and every one of them reached its configured level",
    sorted({tuple(sorted(set(_v[2:]))) == (_v[0],)
            for _v in _CONC_EXPECT.values()}), [True])


print("\n=== run V2's four tables, and the estimand its own docs disagreed on ===")
_V2D = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_V2_s*"))
_V2_FREE = sorted(_d for _d in _V2D if "freerun" in _d)
_V2_CAP = sorted(_d for _d in _V2D if "hardcap" in _d)
chk("V2: eight sessions in each mode", (len(_V2_FREE), len(_V2_CAP)), (8, 8))
_V2_ARMS = ["spec-dflash-n2", "spec-dflash-n4", "spec-mtp-n2", "spec-draft-n8"]
_v2_pp, _v2_log = defaultdict(list), defaultdict(list)
for _f, _c in zip(_V2_FREE, _V2_CAP):
    for _a in _V2_ARMS:
        _fr = _pool_dir(_f, _a) / _pool_dir(_f, "baseline")
        _ca = _pool_dir(_c, _a) / _pool_dir(_c, "baseline")
        _v2_pp[_a].append(100 * (_ca - _fr))
        # the log form: a ratio of ratios, reported as a percentage change.
        # `length_mode.py` documented this one and averaged the other, which
        # is what A17's table exists to show; they differ by 21 points on the
        # arm that sits furthest from its baseline.
        _v2_log[_a].append(100 * (math.exp(math.log(_ca) - math.log(_fr)) - 1))
_V2LT = _num_rows(_ER_LINES, "| arm | published, pp | log contrast |")
chk("A17 estimand table: four arms", sorted(_V2LT), sorted(_V2_ARMS))
for _a in sorted(_V2_ARMS):
    _num_row_check(f"A17 estimand {_a}", _V2LT[_a],
                   [round(st.mean(_v2_pp[_a]), 2), round(st.mean(_v2_log[_a]), 2)])
chk("A17: the two agree near the baseline and part where the arm is far from it",
    max(abs(st.mean(_v2_log[_a]) - st.mean(_v2_pp[_a])) for _a in _V2_ARMS) > 20,
    True)
_V2SD = _num_rows(_ER_LINES, "| arm | SD over 8 sessions | range |")
chk("A17 between-session table: four arms", sorted(_V2SD), sorted(_V2_ARMS))
for _a in sorted(_V2_ARMS):
    _num_row_check(f"A17 between-session {_a}", _V2SD[_a],
                   [round(st.stdev(_v2_pp[_a]), 2),
                    round(max(_v2_pp[_a]) - min(_v2_pp[_a]), 2)])


def _v2_counts(dirs, arm):
    """The one distinct (drafted, accepted) pair a session produced, summed
    over its five arm-runs. Every session produced the same pair, which is the
    claim the table is making."""
    _per = set()
    for _d in dirs:
        _dn = _da = 0
        for _f in sorted(glob.glob(os.path.join(_d, f"{arm}__rep*.json"))):
            _b = json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
            _dn += sum(_r["draft_n"] for _r in _b["rows"])
            _da += sum(_r["draft_n_accepted"] for _r in _b["rows"])
        _per.add((_dn, _da))
    return _per


_V2CT = _num_rows(_ER_LINES,
                  "| arm | freerun drafted / accepted | hard cap drafted / accepted |")
chk("A17 work table: four arms", sorted(_V2CT), sorted(_V2_ARMS))
for _a in sorted(_V2_ARMS):
    _fp, _cp = _v2_counts(_V2_FREE, _a), _v2_counts(_V2_CAP, _a)
    chk(f"A17 work {_a}: every session produced the same drafted/accepted pair",
        (len(_fp), len(_cp)), (1, 1))
    (_fd, _fa), (_cd, _ca) = next(iter(_fp)), next(iter(_cp))
    _num_row_check(f"A17 work {_a}", _V2CT[_a],
                   _grouped(_fd) + _grouped(_fa) + [round(100 * _fa / _fd, 1)]
                   + _grouped(_cd) + _grouped(_ca)
                   + [round(100 * _ca / _cd, 1)])
chk("A17: the request rows the paragraph counts across V2",
    sum(len(json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"])
        for _d in _V2D for _f in glob.glob(os.path.join(_d, "*__rep*.json"))),
    4000)
chk("A17: over that many arm-runs",
    sum(len(glob.glob(os.path.join(_d, "*__rep*.json"))) for _d in _V2D), 400)


print("\n=== run W's carryover contrast, arm by arm ===")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import carryover as _co_w                                          # noqa: E402
_W_DIRS = sorted(glob.glob("v4_audit_2026_08_25/data/matrix_W_s*"))
_w_rep = _co_w.report(_W_DIRS, True)["across_sessions"]
_WCT = _num_rows(_ER_LINES, "| arm | rate after capped vs free | 95 % t over 5 sessions |")
chk("W carryover table: three named arms and one catch-all row",
    sorted(_WCT), ["every other arm", "spec-dflash-n2", "spec-dflash-n2-cap",
                   "spec-dflash-n4-cap"])
for _a in ("spec-dflash-n2", "spec-dflash-n2-cap", "spec-dflash-n4-cap"):
    _v = _w_rep[_a]
    _num_row_check(f"W carryover {_a}", _WCT[_a],
                   [abs(round(_v["mean_delta_pct"], 2)),
                    abs(round(_v["lo"], 2)), abs(round(_v["hi"], 2))])
_w_rest = {_a: _v for _a, _v in _w_rep.items()
           if _a not in ("spec-dflash-n2", "spec-dflash-n2-cap",
                         "spec-dflash-n4-cap")}
_w_rest_max = max(abs(_v["mean_delta_pct"]) for _v in _w_rest.values())
chk("W carryover: the largest arm the table does not name",
    sorted(_a for _a, _v in _w_rest.items()
           if abs(_v["mean_delta_pct"]) == _w_rest_max), ["spec-dflash-n4"])
_num_row_check("W carryover every other arm", _WCT["every other arm"],
               [math.ceil(_w_rest_max * 100) / 100])
chk("W carryover: no arm's interval excludes zero",
    sorted({_v["lo"] <= 0 <= _v["hi"] for _v in _w_rep.values()}), [True])
chk("W carryover: and the largest is that arm, by six times",
    round(abs(_w_rep["spec-dflash-n2"]["mean_delta_pct"])
          / max(abs(_v["mean_delta_pct"]) for _a, _v in _w_rep.items()
                if _a != "spec-dflash-n2")), 6)


print("\n=== the instrumented build against stock, and T against T3 ===")
_O2D = "v4_audit_2026_08_25/data/matrix_O2_latin_20260826_153711"
_TD = "v4_audit_2026_08_25/data/matrix_T_timers_20260826_182639"
_T3D = "v4_audit_2026_08_25/data/matrix_T3_timers_20260826_203251"
_T_ARMS = {"no speculation": "baseline", "spec-draft-n8": "spec-draft-n8",
           "spec-dflash-n2": "spec-dflash-n2"}
_OTT = _num_rows(_ER_LINES,
                 "| arm | O2, stock, 9 blocks | T, instrumented, 4 blocks |")
chk("A12 instrumentation control: three arms", sorted(_OTT), sorted(_T_ARMS))
for _label, _arm in sorted(_T_ARMS.items()):
    _a, _b = _pool_dir(_O2D, _arm), _pool_dir(_TD, _arm)
    _num_row_check(f"A12 instrumentation {_label}", _OTT[_label],
                   [round(_a, 1), round(_b, 1), abs(round(100 * (_b / _a - 1), 2))])
chk("A12: the timers cost under a tenth of a percent on the arm they time",
    abs(round(100 * (_pool_dir(_TD, "spec-dflash-n2")
                     / _pool_dir(_O2D, "spec-dflash-n2") - 1), 2)) < 0.5, True)
_T3T = _num_rows(_ER_LINES, "| arm | run T | run T3 | change |")
chk("A16 T-against-T3 table: three arms", sorted(_T3T), sorted(_T_ARMS))
for _label, _arm in sorted(_T_ARMS.items()):
    _a, _b = _pool_dir(_TD, _arm), _pool_dir(_T3D, _arm)
    _num_row_check(f"A16 T against T3 {_label}", _T3T[_label],
                   [round(_a, 2), round(_b, 2), abs(round(100 * (_b / _a - 1), 2))])
chk("A16: the shortfall is on the DFlash arm and only on it",
    [_l for _l, _a in sorted(_T_ARMS.items())
     if 100 * (_pool_dir(_T3D, _a) / _pool_dir(_TD, _a) - 1) < -1],
    ["spec-dflash-n2"])


print("\n=== the upstream issue table: whose number is in which column ===")
# The left column quotes each report's own title, so the two numbers in it are
# theirs and nothing here can re-derive them; what can be checked is that they
# stay inside the quotation. Every figure in the right column is this
# repository's own.
_ISS = _num_rows_seq(_RM_LINES, "| Issue | Why it matters here |")
chk("issue table: six reports", len(_ISS), 6)
chk("issue table: each row opens with a link to the issue it names",
    sorted({bool(re.match(r"\[#\d+\]\(https://github\.com/ggml-org/llama\.cpp/"
                          r"issues/\d+\)", _c[0])) for _c, _n in _ISS}), [True])
chk("issue table: the link and the number in it agree",
    [re.match(r"\[#(\d+)\]\(\S+/issues/(\d+)\)", _c[0]).groups() for _c, _n in _ISS],
    [(_x, _x) for _x in ("24055", "25004", "24670", "25117", "27572", "27569")])
# scoped to this table's own rows: README carries two other tables whose first
# cell is an issue link, and scanning every line that starts `| [#` picked a
# number out of one of them
_iss_i = next(_i for _i, _l in enumerate(_RM_LINES)
              if _l.startswith("| Issue | Why it matters here |"))
_iss_lines = []
for _l in _RM_LINES[_iss_i + 2:]:
    if not _l.startswith("|"):
        break
    _iss_lines.append(_l)
chk("issue table: the numbers in the left column sit inside the quoted titles",
    sorted(_x[2] for _l in _iss_lines
           for _x in _tcovn._numbers_in(_l, _tcovn._pipe_spans(_l)[0])),
    ["0.0", "2"])
chk("issue table: and the document says that column is a quotation",
    "The left column quotes each report's own title" in " ".join(_ROOT_TEXT.split()),
    True)
# the right column, row by row
_iss_own = {re.match(r"\| \[#(\d+)\]", _l).group(1):
            [_x[2] for _x in _tcovn._numbers_in(_l, _tcovn._pipe_spans(_l)[1])]
            for _l in _iss_lines}
chk("issue table #24055: every figure is A12's, including the retracted sum",
    [float(_x) for _x in _iss_own["24055"]],
    [float(_ck["checkpoints_per_arm_run"][1]), round(_ck["checkpoint_mib"], 3),
     1.0, 300.0, round(_ck["checkpoints_per_request"], 1),
     round(_e["checkpoint_total_mib"] + _e["checkpoint_draft_component_mib"], 1),
     round(_e["checkpoint_total_mib"], 3),
     round(_e["checkpoint_draft_component_mib"], 3)])
chk("issue table #24055: the arm it counts is n_max 1 and its requests are ten",
    (1 in _ck["checkpoints_per_arm_run"], _ck["requests_per_arm_run"]), (True, 10))
_e_nmax = sorted(int(_a.rsplit("-n", 1)[1]) for _a in
                 {os.path.basename(_f).split("__rep")[0] for _f in
                  glob.glob("v4_audit_2026_08_25/data/E_past_threshold/"
                            "spec-draft-n*__rep*.json")})
chk("issue table #27569: the long-draft arm it points at exists",
    [float(_x) for _x in _iss_own["27569"]], [float(max(_e_nmax))])
for _k in ("25004", "24670"):
    chk(f"issue table #{_k}: its right column carries no figure of ours",
        _iss_own[_k], [])
for _k in ("25117", "27572"):
    chk(f"issue table #{_k}: the same, the figure being the report's own",
        _iss_own[_k], [])


print("\n=== A8's p_min defaults: what is measured and what is upstream's ===")
# The two older defaults are properties of llama.cpp source, not of anything
# here. Master's is measurable: run H ran the same arm with and without the
# flag, so the default's own draft volume can be placed against the sweep.
_HD = "v4_audit_2026_08_25/data/H_pmin_sweep"
_HM = json.loads((pathlib.Path(_HD) / "manifest.json").read_text(encoding="utf-8"))


def _h_drafted(arm):
    return sum(_r["draft_n"] for _f in
               sorted(glob.glob(os.path.join(_HD, f"{arm}__rep*.json")))
               for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"])


chk("A8: run H's default arm passes no p_min flag",
    "--spec-draft-p-min" in _HM["arms"]["spec-draft-n8"], False)
chk("A8: and the three that do, pass what their names say",
    [(_a, _HM["arms"][_a][_HM["arms"][_a].index("--spec-draft-p-min") + 1])
     for _a in ("spec-draft-n8-pmin50", "spec-draft-n8-pmin75",
                "spec-draft-n8-pmin90")],
    [("spec-draft-n8-pmin50", "0.5"), ("spec-draft-n8-pmin75", "0.75"),
     ("spec-draft-n8-pmin90", "0.9")])
_h_default = _h_drafted("spec-draft-n8")
chk("A8: the default drafts far more than 0.75 does, so it is not 0.75",
    (_h_default, _h_drafted("spec-draft-n8-pmin75")), (16641, 5535))
chk("A8: and more than 0.50 does, so it is below that",
    (_h_default > _h_drafted("spec-draft-n8-pmin50"),
     _h_drafted("spec-draft-n8-pmin50")), (True, 8424))
chk("A8: draft volume falls monotonically as p_min rises",
    [_h_drafted(_a) for _a in ("spec-draft-n8", "spec-draft-n8-pmin50",
                               "spec-draft-n8-pmin75", "spec-draft-n8-pmin90")],
    sorted([_h_drafted(_a) for _a in ("spec-draft-n8", "spec-draft-n8-pmin50",
                                      "spec-draft-n8-pmin75",
                                      "spec-draft-n8-pmin90")], reverse=True))
# who ran on which binary, which is the table's third column
_pm_bins = defaultdict(list)
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/")):
    _d = _d.rstrip("/")
    if not glob.glob(f"{_d}/*__rep*.json"):
        continue
    _pm_bins[json.loads((pathlib.Path(_d) / "manifest.json")
                        .read_text(encoding="utf-8"))["server_sha256"]].append(
        os.path.basename(_d))
chk("A8: the audit ran two binaries, and only run A the legacy one",
    sorted((len(_v), sorted(_v)[0][:1]) for _v in _pm_bins.values()),
    [(1, "A"), (76, "B")])
chk("A8: the row counts the directories on master",
    max(len(_v) for _v in _pm_bins.values()), 76)
chk("A8: and the table says so rather than naming four runs",
    ("every other audit run: 76 of the 77 directories" in _ER_LINES_TEXT
     and "the whole audit matrix: runs B, C, D, E" not in _ER_LINES_TEXT), True)
chk("A8: only run H sets the flag anywhere in the audit",
    sorted(os.path.basename(_d.rstrip("/")) for _d in
           glob.glob("v4_audit_2026_08_25/data/*/")
           if glob.glob(f"{_d}*__rep*.json")
           and any("--spec-draft-p-min" in _f for _f in
                   (json.loads((pathlib.Path(_d) / "manifest.json")
                               .read_text(encoding="utf-8")).get("arms")
                    or {}).values())),
    ["H_pmin_sweep"])
chk("A8: v1 ran on the commit the row names",
    sorted({_r["commit"] for _r in _V1CSV}), ["9789512"])
_PMT = _num_rows(_ER_LINES, "| build | `p_min` default | what ran on it |")
chk("A8 p_min table: three builds", len(_PMT), 3)
_num_row_check("A8 p_min, 9789512", _PMT["9789512"], [0.75])
_num_row_check("A8 p_min, bcb5eeb64", _PMT["bcb5eeb64"], [0.75, 2])
_num_row_check("A8 p_min, master 3737e4137", _PMT["master 3737e4137"],
               [0.00, max(len(_v) for _v in _pm_bins.values()),
                sum(len(_v) for _v in _pm_bins.values())])


print("\n=== A14's between-run groups, enumerated so they can be checked ===")
# The entry used to publish a four-row histogram over ten hand-picked pairs and
# said so: they were "not enumerated anywhere, which is why only the row that
# names its pair could be checked". This is the enumeration, from a definition
# the document states. A configuration is what the ARM ran with, so keying on
# every model path in the manifest is wrong: it splits runs that differ only in
# whether an unused drafter was recorded, which put run T and run T3 in
# different groups on the first attempt.
_A14_MODEL = (("spec-draft", "draft"), ("spec-mtp", "mtp"),
              ("spec-dflash", "dflash"))
_a14_groups = defaultdict(list)
for _d in sorted(glob.glob("v4_audit_2026_08_25/data/*/")):
    _d = _d.rstrip("/")
    _name = os.path.basename(_d)
    if _name.startswith("smoke") or not glob.glob(f"{_d}/*__rep*.json"):
        continue
    _m = json.loads((pathlib.Path(_d) / "manifest.json").read_text(encoding="utf-8"))
    _ca = _m.get("common_args", [])
    _policy = (_ca[_ca.index("-ngl") + 1] if "-ngl" in _ca else "unset",
               _ca[_ca.index("-c") + 1] if "-c" in _ca else None,
               _m.get("fit_target"),
               "off" if str(_m.get("think")).startswith("off") else "on",
               bool(_m.get("ignore_eos")), _m.get("concurrency") or 1)
    _b = _pool_dir(_d, "baseline") if glob.glob(f"{_d}/baseline__rep*.json") else None
    if not _b:
        continue
    _npr = len({_r["tag"] for _r in json.loads(
        pathlib.Path(sorted(glob.glob(f"{_d}/*__rep*.json"))[0])
        .read_text(encoding="utf-8"))["rows"]})
    _lbl = re.match(r"matrix_([A-Za-z]+[0-9]*)", _name)
    _lbl = _lbl.group(1) if _lbl else _name.split("_")[0]
    for _arm in sorted({os.path.basename(_f).split("__rep")[0]
                        for _f in glob.glob(f"{_d}/*__rep*.json")}):
        if _arm == "baseline" or _arm.endswith("-cap"):
            continue
        # an arm whose every repeat crashed contributes no rate; run A's two
        # speculative arms are the ones this skips
        if not any(not json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
                   .get("crashed")
                   for _f in glob.glob(f"{_d}/{_arm}__rep*.json")):
            continue
        _v = _pool_dir(_d, _arm)
        _mdl = next((_m.get(_fld) for _pre, _fld in _A14_MODEL
                     if _arm.startswith(_pre)), None)
        _a14_groups[(_arm,) + _policy + (_npr, _mdl)].append(
            (_lbl, 100 * (_v / _b - 1)))
_a14_pairs = {_k: _v for _k, _v in _a14_groups.items() if len(_v) in (2, 3)}
_a14_rows = sorted(((round(max(_x for _, _x in _v) - min(_x for _, _x in _v), 2),
                     _k[0], tuple(sorted(_n for _n, _ in _v)))
                    for _k, _v in _a14_pairs.items()), key=lambda _r: -_r[0])
_A14G = _num_rows_seq(_ER_LINES, "| arm | measured in | between-run spread |")
chk("A14 group table: one row per group measured two or three times",
    len(_A14G), len(_a14_rows))
chk("A14 group table: the document says how many there are",
    "Twelve (arm, configuration) groups" in _ER_LINES_TEXT
    and len(_a14_rows) == 12, True)
# compared as a set: two groups share a spread to the hundredth, so the row
# order is a presentation choice and sorting by it is not stable
_a14_pub = {(_c[0].replace("`", "").replace("*", "").split(",")[0].strip(),
             tuple(_x.strip() for _x in _c[1].split(", ")),
             float(_n[0])) for _c, _n in _A14G}
_a14_mine = {(_arm, _runs, _spread) for _spread, _arm, _runs in _a14_rows}
chk("A14 group table: every row is a group the data has, and every group a row",
    sorted(_a14_pub), sorted(_a14_mine))
chk("A14 group table: each row carries exactly one number, its spread",
    sorted({len(_n) for _c, _n in _A14G}), [1])
chk("A14: the median of the twelve, as the sentence quotes it",
    round(st.median([_r[0] for _r in _a14_rows]), 2), 0.55, 0.005)
chk("A14: six of them at or under 0.6 pp",
    sum(1 for _r in _a14_rows if _r[0] <= 0.6), 6)
chk("A14: and one an order of magnitude above the median",
    round(_a14_rows[0][0] / st.median([_r[0] for _r in _a14_rows])) >= 10, True)
chk("A14: the largest is the pair the entry chases",
    (_a14_rows[0][1], _a14_rows[0][2]), ("spec-mtp-n4", ("M1", "Q")))
chk("A14: the designed replications are excluded by the two-or-three rule",
    sorted({_k[0] for _k, _v in _a14_groups.items() if len(_v) > 3}),
    ["spec-dflash-n2", "spec-dflash-n4", "spec-draft-n8", "spec-mtp-n2"])


print("\n=== the open-gaps table, which is all cross-reference ===")
# Every figure in it is published and derived somewhere else in this
# repository. What the table has to do is agree with those, which is why one of
# its rows went stale the moment A14's histogram became an enumeration.
_GAPS = _num_rows(_V4R_LINES, "| gap | why it is still open |")
chk("open-gaps table: eleven rows", len(_GAPS), 11)
chk("open-gaps table: four of them are struck through as closed",
    sum(1 for _k in _GAPS if _k.startswith("~~")), 4)

# the two source line numbers, counted out of the committed patch's own hunk
_PATCH = pathlib.Path("v4_audit_2026_08_25/patches/checkpoint_timers.patch") \
    .read_text(encoding="utf-8").splitlines()
_hunk = next(_i for _i, _l in enumerate(_PATCH) if _l.startswith("@@"))
_old_ln = int(re.search(r"@@ -(\d+)", _PATCH[_hunk]).group(1))
_commented = []
for _l in _PATCH[_hunk + 1:]:
    if _l.startswith("@@"):
        break
    if _l.startswith("+"):
        continue
    if _l.startswith("-") and "//const int64_t" in _l:
        _commented.append(_old_ln)
    _old_ln += 1
chk("open gaps: the timer lines upstream left commented out, from the patch",
    _commented[:2], [2963, 2967])

_gap_ckpt = _GAPS["~~the wall-clock cost of checkpointing~~"]
_num_row_check("open gaps, checkpoint cost", _gap_ckpt,
               [_commented[0], _commented[1],
                round(_A12_CKPT_S, 2), round(_A12_EXCESS_S, 1),
                round(100 * _A12_CKPT_S / _A12_EXCESS_S, 1),
                round(_A16_T3_SHARE, 1)])

# `_f` is a string by here, so the A12 cost table is re-read rather than reused
_gap_unattr = _GAPS["the unattributed 21 % of the external drafter's excess decode time"]
_A12T2 = _num_rows(_ER_LINES, "| | seconds | share of the excess |")
_ck_share = 100 * _A12_CKPT_S / _A12_EXCESS_S
_gen_share = float(_A12T2["drafter generate()"][1])
_num_row_check("open gaps, the unattributed remainder", _gap_unattr,
               [round(100 - _ck_share - _gen_share), round(_ck_share, 1),
                _gen_share, round(100 - _gen_share)])

_num_row_check("open gaps, between-run reproducibility",
               _GAPS["between-run reproducibility"],
               [round(st.median([_r[0] for _r in _a14_rows]), 2),
                _a14_rows[0][0]])
chk("open gaps: that row moved with A14's enumeration",
    ("median 0.55 pp over twelve independently repeated groups" in _V4R_TEXT
     and "over ten independently repeated pairs" not in _V4R_TEXT), True)

# run P against run O, which is the pair the prompt-set row compares
_PD = "v4_audit_2026_08_25/data/matrix_P_extended_20260826_110747"
_OD2 = "v4_audit_2026_08_25/data/matrix_O_headtohead_20260826_081806"
_p_shift = max(
    abs(100 * (_pool_dir(_PD, _a) / _pool_dir(_PD, "baseline") - 1)
        - 100 * (_pool_dir(_OD2, _a) / _pool_dir(_OD2, "baseline") - 1))
    for _a in sorted({os.path.basename(_f).split("__rep")[0]
                      for _f in glob.glob(f"{_PD}/*__rep*.json")})
    if _a != "baseline" and glob.glob(f"{_OD2}/{_a}__rep*.json"))
_num_row_check("open gaps, the extended prompt set", _GAPS["~~ten prompts~~"],
               [round(_p_shift, 1)])

_num_row_check("open gaps, the Q4_K_M head",
               _GAPS["~~n_max 4 under a Q4_K_M MTP head~~"],
               [int("spec-mtp-n4".rsplit("-n", 1)[1])])
chk("open gaps: run Q measured that arm under both heads",
    sorted(_d.split("_")[2] for _d in
           (os.path.basename(_x.rstrip("/")) for _x in
            glob.glob("v4_audit_2026_08_25/data/matrix_Q_*/"))
           if glob.glob(f"v4_audit_2026_08_25/data/{_d}/spec-mtp-n4__rep*.json")),
    ["q4km", "q8"])

_num_row_check("open gaps, the two runs that differ",
               _GAPS["why two runs of the same configuration differ by 3.4 % on one arm"],
               [abs(round(100 * (_pool_dir(_T3D, "spec-dflash-n2")
                                 / _pool_dir(_TD, "spec-dflash-n2") - 1), 1))])

_num_row_check("open gaps, the thinking-off confound",
               _GAPS["every thinking-off comparison here"],
               [round(min(_v["shift_pp"] for _, _, _v in _lm_model), 1),
                round(max(_v["shift_pp"] for _, _, _v in _lm_model), 1)])
for _k in ("three repeats on most arms", "one host, one card, one quantisation",
           "expert routing"):
    chk(f"open gaps: {_k[:34]} carries no figure", _GAPS[_k], [])


print("\n=== the checkpoint volume table, the schema table, and the BOS trio ===")
# A12's two-run volume table: counts from the timer extract, GiB from the
# count times the logged per-checkpoint total. The J row is the earlier
# log-counted run and the T row the source-timed one, and the entry's point is
# that they differ by under three per cent.
_CKT = _num_rows_seq(_ER_LINES, "| | creates | restores | combined, GiB |")
chk("A12 volume table: two runs", len(_CKT), 2)
_ck_mib = _e["checkpoint_total_mib"]
_t_cr = sorted({_r["creates"] for _r in _ext})[0]
_t_rs = sorted({_r["restores"] for _r in _ext})[0]
_num_row_check("A12 volume, run T", _CKT[0][1],
               [_t_cr, round(_t_cr * _ck_mib / 1024, 2),
                _t_rs, round(_t_rs * _ck_mib / 1024, 2),
                round((_t_cr + _t_rs) * _ck_mib / 1024, 2)])
_num_row_check("A12 volume, run J", _CKT[1][1],
               [_e["checkpoints_created"], _e["nominal_state_written_gib"],
                _e["checkpoints_restored"], _e["nominal_state_read_back_gib"],
                _e["nominal_state_total_gib"]])
chk("A12: the two runs differ by the percentages the sentence quotes",
    (round(100 * (_t_cr / _e["checkpoints_created"] - 1), 1),
     round(100 * (_t_rs / _e["checkpoints_restored"] - 1), 1)), (1.7, 2.7))

# the telemetry schema table, counted from the committed traces themselves
_SCHEMA = defaultdict(list)
for _f in sorted(glob.glob("v4_audit_2026_08_25/data/gpu_telemetry_*.csv")):
    _rows = list(csv.reader(open(_f, encoding="utf-8")))
    _SCHEMA[len(_rows[0])].append(_f)


def _schema_interval(path):
    """Seconds between the first two samples, whichever timestamp form."""
    _rows = list(csv.reader(open(path, encoding="utf-8")))[1:4]
    _ts = []
    for _r in _rows:
        _m = re.search(r"(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?", _r[0])
        if _m:
            _ts.append(int(_m.group(1)) * 3600 + int(_m.group(2)) * 60
                       + int(_m.group(3)) + float("0." + (_m.group(4) or "0")))
    return round(_ts[1] - _ts[0]) if len(_ts) > 1 else None


_SCHT = _num_rows(_BE_LINES, "| schema | fields | interval | traces | what reads it |")
chk("telemetry schema table: three schemas", sorted(_SCHT),
    ["compact", "full", "raw"])
for _name, _fields in (("full", 19), ("compact", 9), ("raw", 10)):
    _files = _SCHEMA[_fields]
    _num_row_check(f"telemetry schema {_name}", _SCHT[_name],
                   [_fields, _schema_interval(_files[0]), len(_files)])
chk("telemetry: seventeen traces in total, and the file says so",
    (sum(len(_v) for _v in _SCHEMA.values()),
     "seventeen traces the repository carries" in " ".join(
         pathlib.Path("BENCHMARK_ENV.md").read_text(encoding="utf-8").split())),
    (17, True))
chk("telemetry: every trace of one schema samples at one interval",
    sorted({len({_schema_interval(_f) for _f in _v}) for _v in _SCHEMA.values()}),
    [1])

# the BOS gate, published in three places and derivable in none: the GGUF
# metadata is not committed. What is checkable is that the three agree and
# that the inequality the gate turns on really is one.
_BOS_TGT, _BOS_DFT, _EOS = 248044, 11, 248046
_BOS1 = _num_rows(_ER_LINES, "| | target `UD-Q4_K_XL` | draft `0.8B-Q4_K_M` |")
_num_row_check("A2 BOS key, target", _BOS1["tokenizer.ggml.bos_token_id"],
               [_BOS_TGT])
_num_row_check("A2 BOS as resolved", _BOS1["BOS as resolved by llama.cpp"],
               [_BOS_TGT, _BOS_DFT])
_BOS2 = _num_rows(_ER_LINES, "| condition | target | draft | |")
chk("A2 gate table: four conditions", len(_BOS2), 4)
_num_row_check("A2 gate, llama_vocab_bos", _BOS2["llama_vocab_bos"],
               [_BOS_TGT, _BOS_DFT])
_num_row_check("A2 gate, llama_vocab_eos", _BOS2["llama_vocab_eos"], [_EOS, _EOS])
chk("A2: exactly one of the four conditions fails, and it is the BOS one",
    [_k for _k, _v in _BOS2.items() if len(set(_v)) > 1], ["llama_vocab_bos"])
_BOS3 = _num_rows(_RT_LINES, "| | target `Q4_K_XL` | draft `0.8B-Q4_K_M` |")
_num_row_check("RETEST BOS key", _BOS3["tokenizer.ggml.bos_token_id"], [_BOS_TGT])
_num_row_check("RETEST BOS resolved", _BOS3["resolved BOS"], [_BOS_TGT, _BOS_DFT])
chk("the override the runs actually passed carries the target's own id",
    all(f"tokenizer.ggml.bos_token_id=int:{_BOS_TGT}" in " ".join(_v)
        for _v in json.loads((pathlib.Path(
            "v4_audit_2026_08_25/data/C_master_matrix_think_on/manifest.json"))
            .read_text(encoding="utf-8"))["arms"].values()
        if "--override-kv" in _v), True)


print("\n=== run O's headline table names the drafter, and that is a value ===")
# `spec-draft-n8 - external 0.8 B drafter`: the 0.8 is the drafter's parameter
# count, sits in the label column, and was the last number in that table that
# nothing read. It is in the manifest, in the file the run actually loaded.
_OD = "v4_audit_2026_08_25/data/matrix_O_headtohead_20260826_081806"
_o_draft = json.loads((pathlib.Path(_OD) / "manifest.json")
                      .read_text(encoding="utf-8"))["draft"]
_o_size = re.search(r"-([\d.]+)B-", os.path.basename(_o_draft)).group(1)
chk("run O: the drafter the manifest names", os.path.basename(_o_draft),
    "Qwen3.5-0.8B-Q4_K_M.gguf")
# by position, not containment: the same phrase appears in the scorecard, so
# `in _V4R_TEXT` is satisfied by the other occurrence and perturbing this one
# passed. Both rows are addressed as rows now.
_OHT = _num_rows_seq(_V4R_LINES, "| arm | pooled tok/s | \u0394 pooled |")
_o_row = next(_c for _c, _n in _OHT if _c[0].startswith("`spec-draft-n8`"))
chk("run O: the headline table's external-drafter row labels it with its size",
    _o_row[0].endswith(f"external {_o_size} B drafter"), True)
_o_sc = next(_c for _c, _n in
             _num_rows_seq(_V4R_LINES, "| family | sign predicted correctly |")
             if "external" in _c[0])
chk("run O: and so does the scorecard's row",
    _o_sc[0].strip("* ") == f"external {_o_size} B drafter", True)
chk("run O: every run that loads a drafter loads that one",
    sorted({os.path.basename(json.loads((pathlib.Path(_d) / "manifest.json")
                                        .read_text(encoding="utf-8"))["draft"])
            for _d in glob.glob("v4_audit_2026_08_25/data/*/")
            if glob.glob(f"{_d}spec-draft-n*__rep*.json")}),
    ["Qwen3.5-0.8B-Q4_K_M.gguf"])


print("\n=== the audit README's file map, and the size the logs really are ===")
# addressed by row rather than by key: the key strips `*`, which is what makes
# a bold cell readable and what makes a glob unrecognisable
_V4FM = _num_rows_seq(_V4R_LINES, "| Path | Contents |")
chk("v4 file map: fifteen rows", len(_V4FM), 15)
_v4fm_by = {_c[0]: _n for _c, _n in _V4FM}
_o2o3 = next(_n for _c, _n in _V4FM if "matrix_O2_latin" in _c[0])
chk("v4 file map: the O2/O3 row counts the request-pairs both runs hold",
    [float(_x) for _x in _o2o3],
    [float(sum(len(json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"])
               for _f in glob.glob("v4_audit_2026_08_25/data/"
                                   "matrix_O2_latin_*/*__rep*.json")))] * 2)
# that row spells its two counts as words, so it offers nothing to perturb;
# the words still have to be right
_tim_c, _tim = next((_c, _n) for _c, _n in _V4FM
                    if "checkpoint_timers_20260826" in _c[0])
chk("v4 file map: the timer row carries no digit to check", _tim, [])
_tim_n = len(json.loads(pathlib.Path(
    "v4_audit_2026_08_25/data/checkpoint_timers_20260826.json")
    .read_text(encoding="utf-8")))
chk("v4 file map: and its words are run T's record and repeat counts",
    (_tim_n, sorted({_r["repeat"] for _r in json.loads(pathlib.Path(
        "v4_audit_2026_08_25/data/checkpoint_timers_20260826.json")
        .read_text(encoding="utf-8"))})),
    (12, [0, 1, 2, 3]))
chk("v4 file map: which is what the row says",
    "twelve timer records, four repeats per arm" in _tim_c[1], True)
# the size of the uncommitted logs is stated in four places and one disagreed
_CES = pathlib.Path("bench/collect_evidence.sh").read_text(encoding="utf-8")
chk("the archive script states the log volume three times",
    _CES.count("3 GB"), 3)
chk("the audit README states the same", "~3 GB of server logs" in _V4R_TEXT, True)
chk("and so does the body, which said 7 GB until 2026-08-29",
    ("~3 GB of llama-server logs are not committed" in _PR
     and "7 GB of llama-server logs" not in _PR), True)


print("\n=== the README's three file maps, which share a header ===")
# `| Path | Contents |` appears three times in README.md, and the census marks
# all three parsed the moment the header reaches a reader. Reading only the
# first would make the coverage number describe two tables nothing looked at.
_DM = [_num_rows_seq(_RM_LINES, "| Path | Contents |", _k) for _k in range(3)]
chk("README file maps: three tables under one header",
    [len(_t) for _t in _DM], [11, 11, 8])
chk("README data map: the v1 label count, the v2 logs and the Exp 2 label",
    [float(_x) for _c, _n in _DM[0] for _x in _n],
    [float(len({_r["config"] for _r in _V1CSV})),
     float(len(_v2_logs) - len(_v2_verbose)), 2.0])
chk("README v4 map: run B's requests an arm, run A's short arms, the "
    "concurrency levels and the telemetry interval",
    [float(_x) for _c, _n in _DM[1] for _x in _n],
    [30.0, 12.0, 1.0, 4.0, 8.0, 5.0])
chk("README v4 map: the smoke row it ends on names the start-up checks",
    "the gate runs that decide a matrix is safe to start" in _ROOT_TEXT, True)
chk("README harness map: it carries no measurement at all",
    [_x for _c, _n in _DM[2] for _x in _n], [])
chk("README v4 map: the telemetry interval it names is the schemas' own",
    sorted({_schema_interval(_f) for _v in _SCHEMA.values() for _f in _v}), [1, 5])


print("\n=== the honest headline, and the A2 rate table in the TODO ===")
_HEAD = _num_rows_seq(_RM_LINES, "| | |")
chk("README honest-headline table: three rows", len(_HEAD), 3)
_num_row_check("README headline, across invocations", _HEAD[0][1],
               [int(min(_tw_v)), round(max(_tw_v))])
_num_row_check("README headline, O2 point", _HEAD[1][1],
               [round([_v for _t, _n, _v, _b in _TWELVE if "O2" in _n][0], 1)])
_num_row_check("README headline, O2 interval", _HEAD[2][1],
               [round(_pb2["spec-dflash-n2"][0], 1),
                round(_pb2["spec-dflash-n2"][1], 1)])

_A2R = _num_rows(_RT_LINES,
                 "| arm | `long_explain` | counted draft tokens | `code_small` |")
chk("RETEST A2 table: three arms", len(_A2R), 3)
# every rate in it is run A's own `long_explain` decode rate, per repeat. The
# baseline row quoted `~125-129 tok/s (quiet host)` until 2026-08-29, which is
# neither repeat of the control measured beside them.
_AD = "v4_audit_2026_08_25/data/A_bcb5eeb64_legacy"


def _a2_rates(arm):
    _out = []
    for _f in sorted(glob.glob(os.path.join(_AD, f"{arm}__rep*.json"))):
        for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]:
            if _r["tag"] == "long_explain":
                _out.append(round(1000 * _r["timings"]["predicted_n"]
                                  / _r["timings"]["predicted_ms"], 1))
    return _out


_A2_ROWS = {"translation fallback": "draft-max8-translate",
            "matched vocabulary": "draft-max8-matched",
            "baseline, measured beside them": "baseline"}
chk("RETEST A2: the rows it names", sorted(_A2R), sorted(_A2_ROWS))
for _label, _arm in sorted(_A2_ROWS.items()):
    _want = _a2_rates(_arm)
    if _arm != "baseline":
        _want = _want + [97, 97]
    _num_row_check(f"RETEST A2 {_label}", _A2R[_label], _want)
chk("RETEST A2: both speculative arms drafted and accepted the same tokens",
    sorted({(sum(_r["draft_n"] for _f in
                 sorted(glob.glob(os.path.join(_AD, f"{_a}__rep*.json")))
                 for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]),
             sum(_r["draft_n_accepted"] for _f in
                 sorted(glob.glob(os.path.join(_AD, f"{_a}__rep*.json")))
                 for _r in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]))
            for _a in ("draft-max8-translate", "draft-max8-matched")}),
    [(194, 194)])
_a2_drafted = sum(_r["draft_n"] for _f in
                  sorted(glob.glob(os.path.join(_AD,
                                                "draft-max8-matched__rep*.json")))
                  for _r in json.loads(pathlib.Path(_f)
                                       .read_text(encoding="utf-8"))["rows"])
chk("RETEST A2: which is what one repeat drafted, as the table prints it",
    _a2_drafted // json.loads(pathlib.Path(_AD).joinpath("manifest.json")
                              .read_text(encoding="utf-8"))["repeats"],
    int(_A2R["matched vocabulary"][2]))


print("\n=== the run registry: every number in it, against the manifests ===")
# Two tables nothing read. The first was wrong in four places at once: run C
# published as 3 repeats when its manifest says 5, run D as thirteen arms when
# it has five, run A as 30 requests an arm when its speculative arms abort at
# 12, and run E's sweep missing its lowest length. RETEST_TODO had said 5
# repeats and 900 requests for the same run all along.
_DATA = pathlib.Path(__file__).resolve().parents[1] / "v4_audit_2026_08_25" / "data"


def _rg_mf(name):
    return json.loads((_DATA / name / "manifest.json").read_text(encoding="utf-8"))


def _rg_runs(name):
    return sorted(glob.glob(str(_DATA / name / "*__rep*.json")))


def _rg_arms(name):
    return sorted({os.path.basename(f).split("__rep")[0]
                   for f in _rg_runs(name)})


def _rg_requests(name, arm=None):
    _n = 0
    for _f in _rg_runs(name):
        if arm and os.path.basename(_f).split("__rep")[0] != arm:
            continue
        _n += len(json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"])
    return _n


def _rg_prompts(name):
    _f = _rg_runs(name)[0]
    return len({_r["tag"] for _r
                in json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))["rows"]})


def _rg_nmax(name, prefix):
    """The draft lengths a run's arm names carry, in numeric order."""
    # [-1], not [1]: run C's `spec-draft-v1cfg` starts with the prefix and has
    # no `-n` at all, so [1] is an IndexError rather than a filtered arm.
    return sorted(int(_a.rsplit("-n", 1)[-1]) for _a in _rg_arms(name)
                  if _a.startswith(prefix) and _a.rsplit("-n", 1)[-1].isdigit())


_A, _B = "A_bcb5eeb64_legacy", "B_master_3737e4137"
_Cd, _Dd = "C_master_matrix_think_on", "D_master_matrix_think_off"
_Ed, _Hd = "E_past_threshold", "H_pmin_sweep"
_Kd = "matrix_K1_sweep_20260826_025615"
_Od2, _Od3 = "matrix_O2_latin_20260826_153711", "matrix_O3_latin_20260826_203251"
_Wd = sorted(str(_x.name) for _x in _DATA.glob("matrix_W_s*"))
_W2d = sorted(str(_x.name) for _x in _DATA.glob("matrix_W2_s*"))

_REG_EXPECT = {
    # A and B share a design; A's two speculative arms abort, so only its
    # baseline reaches the full ten prompts a repeat.
    "A / B": [len(_rg_arms(_A)), _rg_prompts(_A), _rg_mf(_A)["repeats"],
              _rg_mf(_B)["repeats"], _rg_requests(_B, "baseline")],
    "C / D": [len(_rg_arms(_Cd)), _rg_prompts(_Cd), _rg_mf(_Cd)["repeats"],
              len(_rg_arms(_Dd)), _rg_mf(_Dd)["repeats"]],
    # the 95 is E1's ceilinged coverage threshold, derived above
    "E": [math.ceil(math.log(0.05) / math.log(1 - rho))]
         + _rg_nmax(_Ed, "spec-draft") + [_rg_mf(_Ed)["repeats"]],
    "H": [0, 0.50, 0.75, 0.90, 8, 32, 128, 0.75, _rg_mf(_Hd)["repeats"]],
    "I": [1, 4, 8],
    "J": [len(_rg_arms("matrix_J2_20260826_014750")),
          _rg_mf("matrix_J2_20260826_014750")["repeats"]],
    "K": _rg_nmax(_Kd, "spec-dflash") + [_rg_mf(_Kd)["repeats"], 4, 8],
    "L": [len(_rg_arms("matrix_L_thinkon_20260826_032652")),
          _rg_mf("matrix_L_thinkon_20260826_032652")["repeats"]],
    "M": [len(_rg_arms("matrix_M1_20260826_075816")),
          _rg_mf("matrix_M1_20260826_075816")["repeats"]],
    "N": [len(_rg_arms("matrix_N_ngrammap_20260826_081806")),
          _rg_mf("matrix_N_ngrammap_20260826_081806")["repeats"]],
    "O": [len(_rg_arms("matrix_O_headtohead_20260826_081806")),
          _rg_mf("matrix_O_headtohead_20260826_081806")["repeats"]],
    "O2 / O3": [len(_rg_arms(_Od2)), _rg_mf(_Od2)["repeats"],
                _rg_requests(_Od2), _rg_requests(_Od2)],
    "P / R": [],
    "Q": [len(_rg_arms("matrix_Q_q8_20260826_110747")),
          _rg_mf("matrix_Q_q8_20260826_110747")["repeats"]],
    "T / T3 / T4": [],
    "U": [],
    "V / V2 / V3": [],
    "W": [len(_rg_arms(_Wd[0])), _rg_mf(_Wd[0])["repeats"],
          sum(len(_rg_runs(_d)) for _d in _Wd)],
    "W2": [len(_rg_arms(_W2d[0])), _rg_mf(_W2d[0])["repeats"],
           sum(len(_rg_runs(_d)) for _d in _W2d)],
}


_REG = _num_rows(_RM_LINES, "| run | question | design |")
chk("README run registry: one row per entry", sorted(_REG), sorted(_REG_EXPECT))
for _run, _want in sorted(_REG_EXPECT.items()):
    chk(f"README registry {_run}: every number in the row",
        [float(_x) for _x in _REG[_run]], [float(_x) for _x in _want])
chk("README registry: the rows with no number at all are the four prose ones",
    sorted(_k for _k, _v in _REG_EXPECT.items() if not _v),
    ["P / R", "T / T3 / T4", "U", "V / V2 / V3"])
# and the run A abort the corrected row now states, which is why it is not 30
chk("run A's speculative arms stop short of the baseline's request count",
    sorted({_rg_requests(_A, _a) for _a in _rg_arms(_A)
            if _a != "baseline"}), [12])
chk("run A's baseline does not", _rg_requests(_A, "baseline"), 20)
chk("and the short arms are the ones the manifest marks crashed",
    sorted({os.path.basename(_f).split("__rep")[0] for _f in _rg_runs(_A)
            if json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
            .get("crashed")}),
    ["draft-max8-matched", "draft-max8-translate"])
chk("run D's five arms are five of run C's thirteen",
    set(_rg_arms(_Dd)) <= set(_rg_arms(_Cd)), True)
chk("and run D is the thinking-off one",
    (_rg_mf(_Cd)["think"], _rg_mf(_Dd)["think"]), ("think_on", "off"))

# --- and the tier table above it, whose numbers describe the archives -------
# Its v4 row was orphaned from the header by a blank line, so GitHub rendered
# the controlled tier as literal pipes and no reader here could see it either.
_v1_labels = {_r["config"] for _r in _V1CSV}
_v1_drafted = {_r["config"] for _r in _V1CSV if int(_r["draft_n"]) > 0}
_V1RAW = json.loads((pathlib.Path(__file__).resolve().parents[1] / "results"
                     / "baseline.json").read_text(encoding="utf-8"))
_V2RAW = json.loads((pathlib.Path(__file__).resolve().parents[1]
                     / "v2_3090_followup" / "results_v2.json")
                    .read_text(encoding="utf-8"))
_X2RAW = json.loads((pathlib.Path(__file__).resolve().parents[1]
                     / "v2_3090_followup" / "n3_results_20260426.json")
                    .read_text(encoding="utf-8"))
_V3OUT = pathlib.Path(__file__).resolve().parents[1] / "v3_dflash_2026_05_07" \
    / "data" / "out_20260507_183341"


def _card_number(text):
    """The 3090 in `NVIDIA GeForce RTX 3090`, from the archive that recorded it."""
    return int(re.search(r"RTX (\d+)", text).group(1))


_v2_flags = _V2RAW["common_flags"]
_v3_dirs = sorted(_d.name for _d in _V3OUT.iterdir() if _d.is_dir())
_v3_dflash = [_d for _d in _v3_dirs if "dflash" in _d]
_v4_repeats = sorted({_rg_mf(os.path.basename(_d.rstrip("/")))["repeats"]
                      for _d in _REG_ALL if _d not in _REG_SMOKE})
_TIER_EXPECT = {
    "v1 primary matrix": [
        _card_number(_V1RAW["gpu_info"]),
        len(_v1_labels), len(_v1_drafted), len(_v1_labels - _v1_drafted),
        len({_r["prompt"] for _r in _V1CSV}),
        int(float(re.search(r'"temperature": ([0-9.]+)', (
            pathlib.Path(__file__).resolve().parents[1] / "bench_runner.py")
            .read_text(encoding="utf-8")).group(1))),
        Counter(int(_r["max_tokens"]) for _r in _V1CSV).most_common(1)[0][0]],
    "v2 follow-up": [
        _card_number(_V2RAW["hardware"]["gpu"]), len(_V2RAW["prompts"]),
        float(_v2_flags[_v2_flags.index("--temp") + 1]),
        int(_v2_flags[_v2_flags.index("-n") + 1])],
    "Exp 2 code/JSON": [
        2,
        len(_X2RAW["configs"]["01_baseline"]["per_trial"][0]["results"]),
        _X2RAW["n_trials"], len(_X2RAW["configs"])],
    "v3 DFlash": [
        len(list((_V3OUT / "01_baseline").glob("p*.log"))), 1, len(_v3_dflash)],
    "v4 audit": [
        _card_number(_V1RAW["gpu_info"]), _card_number(_V1RAW["gpu_info"]),
        _REG_ARMRUNS, len(_REG_ALL) - len(_REG_SMOKE),
        min(_v4_repeats), max(_v4_repeats)],
}
_TIER = _num_rows(_RM_LINES,
                  "| ID | Date | Host / runner | Design | Evidence level |")
chk("README tier table: one row per tier", sorted(_TIER), sorted(_TIER_EXPECT))
for _tier, _want in sorted(_TIER_EXPECT.items()):
    chk(f"README tier {_tier}: every number in the row",
        [float(_x) for _x in _TIER[_tier]], [float(_x) for _x in _want])
chk("the v1 host really had two of those cards",
    _V1RAW["gpu_info"].strip().count("RTX 3090"), 2)
chk("and the v1 commit the row names is the one the archive recorded",
    f"commit `{_V1RAW['llama_cpp_commit']}`" in _ROOT_TEXT, True)
chk("the v1 cap applies to all but the 1000-token label",
    sorted({int(_r["max_tokens"]) for _r in _V1CSV}), [300, 1000])
chk("Exp 2's own harness line says the same five prompts and cap",
    _X2RAW["harness"], "5 prompts x max_tokens=200 x temperature=0.5 x seed=42")
chk("every v3 draft-max directory is one of the three the row counts",
    sorted(_v3_dflash),
    ["05_dflash_max16", "06_dflash_max8", "07_dflash_max4"])


print("\n=== the status board, which is eighteen rows of pure cross-reference ===")
# Nothing had ever read this table. It is the repository's own index of what
# the audit did, so every figure in it is a result derived somewhere else --
# which is exactly how one of them went stale unnoticed: P0-1 still carried
# A2's superseded +0.3 %. The task ID in the first column contributes a digit
# of its own, so each row's expectation begins with the number its ID ends in.


def _sb_arm(pat):
    """`_arm` again, under a name nothing has rebound.

    A `for _arm, ... in (...)` loop at run V replaced the function with a
    string for the rest of the file, so calling it here would fail."""
    _v = [json.loads(pathlib.Path(_f).read_text(encoding="utf-8"))
          for _f in sorted(glob.glob(pat))]
    assert _v, f"no data for {pat}"
    _aggs = [_x["aggregate_tok_s"] for _x in _v]
    return (st.mean(_aggs), st.stdev(_aggs) if len(_aggs) > 1 else 0.0)


def _sb_delta(pat, arm, base="baseline"):
    return 100 * (_sb_arm(pat % arm)[0] / _sb_arm(pat % base)[0] - 1)


# P0-1. A2's vocabulary difference, pooled over both binaries and per cell.
# The A2 entry published +0.3 % overall and -1.2 % to +3.7 % per prompt; both
# were computed on one binary's six prompts.
_SB_CELL = {}
for _sbb in (_A, _B):
    for _sba in ("draft-max8-translate", "draft-max8-matched"):
        for _sbf in _rg_runs(_sbb):
            if os.path.basename(_sbf).split("__rep")[0] != _sba:
                continue
            for _sbr in json.loads(pathlib.Path(_sbf)
                                   .read_text(encoding="utf-8"))["rows"]:
                _sbc = _SB_CELL.setdefault((_sbb, _sba, _sbr["tag"]), [0, 0])
                _sbc[0] += _sbr["timings"]["predicted_n"]
                _sbc[1] += _sbr["timings"]["predicted_ms"]


def _sb_rate(key):
    return 1000 * _SB_CELL[key][0] / _SB_CELL[key][1]


_SB_TOT = {}
for (_sbb, _sba, _sbt), _sbc in _SB_CELL.items():
    _sbx = _SB_TOT.setdefault(_sba, [0, 0])
    _sbx[0] += _sbc[0]
    _sbx[1] += _sbc[1]
_A2_POOLED = 100 * ((_SB_TOT["draft-max8-matched"][0]
                     / _SB_TOT["draft-max8-matched"][1])
                    / (_SB_TOT["draft-max8-translate"][0]
                       / _SB_TOT["draft-max8-translate"][1]) - 1)
_A2_SPAN = sorted(100 * (_sb_rate((_sbb, "draft-max8-matched", _sbt))
                         / _sb_rate((_sbb, "draft-max8-translate", _sbt)) - 1)
                  for (_sbb, _sba, _sbt) in _SB_CELL
                  if _sba == "draft-max8-translate")
# not `_rg_prompts`: that reads one file, and on the legacy binary only the
# baseline reaches all ten prompts -- the two speculative arms abort at six,
# which is why the entry's original per-prompt span was over six cells.
_A2_TAGS = {_b: sorted({_t for (_bb, _aa, _t) in _SB_CELL
                        if _bb == _b and _aa == "draft-max8-translate"})
            for _b in (_A, _B)}
chk("A2: sixteen (binary, prompt) cells, not the legacy binary's six",
    (len(_A2_SPAN), len(_A2_TAGS[_A]), len(_A2_TAGS[_B])), (16, 6, 10))
chk("A2: the vocabulary difference pooled over both binaries",
    round(_A2_POOLED, 1), 0.2, 0.005)
chk("A2: and the span across those cells",
    (round(_A2_SPAN[0], 1), round(_A2_SPAN[-1], 1)), (-2.2, 3.7))
# and the diagnosis of where the superseded pair came from, which is a claim
# about arithmetic and so is checked like any other. `predicted_per_second` is
# the field B8 is about; the published +0.3 % is the ratio of two rounded
# copies of it, and the published span is the master binary's six v1 prompts.
_A2_RM = {}
for _sba in ("draft-max8-translate", "draft-max8-matched"):
    _A2_RM[_sba] = st.mean(
        [_r["timings"]["predicted_per_second"]
         for _f in _rg_runs(_B)
         if os.path.basename(_f).split("__rep")[0] == _sba
         for _r in json.loads(pathlib.Path(_f)
                              .read_text(encoding="utf-8"))["rows"]])
chk("A2: the superseded +0.3 % is the ratio of the table's two rounded means",
    (round(100 * (round(_A2_RM["draft-max8-matched"], 1)
                  / round(_A2_RM["draft-max8-translate"], 1) - 1), 1),
     round(100 * (_A2_RM["draft-max8-matched"]
                  / _A2_RM["draft-max8-translate"] - 1), 2)),
    (0.3, 0.19))
_A2_V1 = ["long_explain", "medium_chat", "medium_rec", "reasoning",
          "short_greet", "short_q"]
_A2_SIX = sorted(100 * (_sb_rate((_B, "draft-max8-matched", _t))
                        / _sb_rate((_B, "draft-max8-translate", _t)) - 1)
                 for _t in _A2_V1)
chk("A2: and the superseded span is that binary's six v1-tagged prompts",
    (len(_A2_SIX), round(_A2_SIX[0], 1), round(_A2_SIX[-1], 1)), (6, -1.2, 3.7))
chk("A2: the cell it drops is zh_hant, which is the widest of the sixteen",
    round(100 * (_sb_rate((_B, "draft-max8-matched", "zh_hant"))
                 / _sb_rate((_B, "draft-max8-translate", "zh_hant")) - 1), 1),
    round(_A2_SPAN[0], 1), 0.005)
_A2_SEC = _ER_LINES_TEXT.split("### A2.")[1].split("\n### ")[0]
# all three documents that carried the superseded pair, checked together: the
# entry, the audit README's answer 1, and the status board row
_A2_SAYS = " ".join(_norm("+0.2 % pooled over both binaries, and from −2.2 % "
                          "to +3.7 % across the sixteen (binary, prompt) "
                          "cells").split())
chk("ERRATA A2 states both, and no longer the pair they superseded",
    (_A2_SAYS in " ".join(_norm(_A2_SEC).split()), "+0.3 %" in _A2_SEC),
    (True, False))
chk("the audit README's answer 1 states the same pair, and not the old one",
    (_A2_SAYS in " ".join(_norm(_V4R_TEXT).split()),
     "+0.3 % overall" in _V4R_TEXT), (True, False))

# P0-2. What the thinking control's "50" counts. Four places printed 50/50
# against 0/50 with nothing to say the 50 was per arm; over the whole run it is
# 250 of 250 against 0 of 650.


def _sb_sup(run):
    _per = {}
    for _sbf in _rg_runs(run):
        _sbc = _per.setdefault(os.path.basename(_sbf).split("__rep")[0], [0, 0])
        for _sbr in json.loads(pathlib.Path(_sbf)
                               .read_text(encoding="utf-8"))["rows"]:
            _sbc[1] += 1
            _sbc[0] += bool(_sbr.get("thinking_suppressed"))
    return _per


_SUP_D, _SUP_C = _sb_sup(_Dd), _sb_sup(_Cd)
_PER_ARM = _rg_mf(_Dd)["repeats"] * _rg_prompts(_Dd)
chk("D: every arm suppressed thinking on every request it made",
    sorted({tuple(_v) for _v in _SUP_D.values()}), [(_PER_ARM, _PER_ARM)])
chk("C: and no arm did", sorted({tuple(_v) for _v in _SUP_C.values()}),
    [(0, _PER_ARM)])
chk("the per-arm count, and the two run totals it is not",
    (_PER_ARM, _PER_ARM * len(_SUP_D), _PER_ARM * len(_SUP_C)), (50, 250, 650))
for _sbdoc, _sbtxt, _sbneedle in (
        ("v4 README", _V4R_TEXT,
         "Per arm it is 50 of 50 in D and 0 of 50 in C, which is 250 of 250 "
         "over D against 0 of 650 over C."),
        ("ERRATA D3", _ER_LINES_TEXT,
         "per arm, 50 of 50 in the off run against 0 of 50 in the on run, "
         "which is 250 of 250 over D against 0 of 650 over C."),
        ("RETEST P0-2", "\n".join(_RT_LINES),
         "Measured, per arm: 50 of 50 requests suppressed with it and 0 of 50 "
         "without, which is 250 of 250 over run D against 0 of 650 over run C"),
):
    chk(f"{_sbdoc}: the thinking control says what its 50 counts",
        " ".join(_sbneedle.split()) in " ".join(_sbtxt.split()), True)

# P1-3. Run V's two block start times, the shift under length matching, and
# the mode contrast. The two are different quantities on the same run, which
# is what A17 was corrected for.
_SB_VFR = _rg_mf("matrix_V_freerun_20260826_210956")["created"]
_SB_VHC = _rg_mf("matrix_V_hardcap_20260826_210956")["created"]


def _sb_clock(stamp):
    return [int(stamp[11:13]), int(stamp[14:16]), int(stamp[17:19])]


chk("run V: the free-run block really did start before the capped one",
    _SB_VFR < _SB_VHC, True)
_MD_SHIFT = sorted(round(_v["length_matched_pct"] - _v["all_prompts_pct"], 1)
                   for _r in _LMR.values()
                   if _r["think"] == "off" and not _r["ignore_eos"]
                   for _a, _v in _r["arms"].items() if _a.startswith("spec-"))
chk("A17: length matching moves every model-drafting arm the same way",
    sorted({_x > 0 for _x in _MD_SHIFT}), [True])
_SB_VF_N4 = _LMR["matrix_V_freerun_20260826_210956"]["arms"][
    "spec-dflash-n4"]["all_prompts_pct"]
_SB_VH_N4 = _LMR["matrix_V_hardcap_20260826_210956"]["arms"][
    "spec-dflash-n4"]["all_prompts_pct"]
_SB_L_N4 = _LMR["matrix_L_thinkoff_20260826_032652"]["arms"][
    "spec-dflash-n4"]["length_matched_pct"]

# P4-2. The plateau is the arms within one baseline SD of the best in K1, and
# it is 2 to 4: n_max 1 is nearly ten tok/s short of it and n_max 6 collapses.
_SB_K1 = "v4_audit_2026_08_25/data/matrix_K1_sweep_*/%s__rep*.json"
_SB_K1P = {_n: _sb_arm(_SB_K1 % f"spec-dflash-n{_n}")[0]
           for _n in _rg_nmax(_Kd, "spec-dflash")}
_PLATEAU = sorted(_n for _n in _SB_K1P
                  if abs(_SB_K1P[_n] - max(_SB_K1P.values()))
                  < _sb_arm(_SB_K1 % "baseline")[1])
chk("K1: the plateau is the run of lengths inside one baseline SD of the best",
    _PLATEAU, [2, 3, 4])

# P3-1. Run E is run C's sweep continued past its top, so the board's second
# list is exactly what E adds.
_SB_C_SWEEP = _rg_nmax(_Cd, "spec-draft")
_SB_E_SWEEP = sorted(set(_rg_nmax(_Ed, "spec-draft")) - set(_SB_C_SWEEP))
chk("E continues C's sweep rather than repeating it",
    (min(_SB_C_SWEEP), max(_SB_C_SWEEP), _SB_E_SWEEP), (1, 32, [64, 96, 128]))

_SB_I = "v4_audit_2026_08_25/data/matrix_I2_conc%d_*/%s__rep*.json"
_SB_J = "v4_audit_2026_08_25/data/matrix_J2_*/%s__rep*.json"
_SB_KC = "v4_audit_2026_08_25/data/matrix_K_conc%d_*/%s__rep*.json"
_SB_CONC = sorted({_x["max_in_flight"] for _p in
                   ("matrix_I2_conc*_*", "matrix_K_conc*_*")
                   for _f in glob.glob(f"v4_audit_2026_08_25/data/{_p}/*__rep*.json")
                   for _x in [json.loads(pathlib.Path(_f)
                                         .read_text(encoding="utf-8"))]})
chk("runs I and K were driven at one, four and eight in flight",
    _SB_CONC, [1, 4, 8])
_SB_I_BASE = round(100 * (_sb_arm(_SB_I % (_SB_CONC[-1], "baseline"))[0]
                          / _sb_arm(_SB_I % (_SB_CONC[0], "baseline"))[0] - 1))
_SB_I_SPEC = round(100 * (_sb_arm(_SB_I % (_SB_CONC[-1], "spec-draft-n8"))[0]
                          / _sb_arm(_SB_I % (_SB_CONC[0], "spec-draft-n8"))[0] - 1))
_SB_J_N4 = round(_sb_delta(_SB_J, "spec-dflash-n4"), 1)
_SB_K_C4 = round(100 * (_sb_arm(_SB_KC % (_SB_CONC[1], "spec-dflash-n4"))[0]
                        / _sb_arm(_SB_KC % (_SB_CONC[1], "baseline"))[0] - 1), 1)
_SB_K_C8 = round(100 * (_sb_arm(_SB_KC % (_SB_CONC[-1], "spec-dflash-n4"))[0]
                        / _sb_arm(_SB_KC % (_SB_CONC[-1], "baseline"))[0] - 1), 1)
chk("run I and run K disagree about batching, which is the board's point",
    (_SB_I_BASE > 0, _SB_I_SPEC < 0, _SB_K_C4 > 0, _SB_K_C8 < 0),
    (True, True, True, True))

# The `n_max` each arm name carries, so the board's bare 4 is the arm's and
# not a repeat count that happens to match.
_SB_NMAX4 = _rg_nmax("matrix_J2_20260826_014750", "spec-dflash")[0]
chk("run J's DFlash arms start at n_max 4", _SB_NMAX4, 4)

_SB = _num_rows(_RT_LINES, "| # | task | state |")
_SB_ROW = {_l.strip("|").split("|")[0].strip(): _l
           for _l in _RT_LINES
           if _l.strip().startswith("| P") and _l.count("|") >= 4}
_SB_ROW = {_k: _v for _k, _v in _SB_ROW.items() if _k in _SB}
chk("status board: eighteen tasks, each row's own text in reach",
    (len(_SB), len(_SB_ROW)), (18, 18))
chk("status board: and the text found is the row that was parsed",
    {_k: [_x[2] for _sp in _tcovn._pipe_spans(_v)
          for _x in _tcovn._numbers_in(_v, _sp)]
     for _k, _v in _SB_ROW.items()}, _SB)
chk("status board: run V's two block times are printed as clock times",
    (_SB_VFR[11:19] in _SB_ROW["P1-3"], _SB_VHC[11:19] in _SB_ROW["P1-3"]),
    (True, True))

_SB_WANT = {
    "P0-1": [round(_A2_POOLED, 1)],
    "P0-2": [_PER_ARM, _PER_ARM],
    "P0-3": [],
    "P1-1": [_rg_mf(_Cd)["repeats"], len(_rg_arms(_Cd)), _rg_mf(_Cd)["repeats"],
             _rg_requests(_Cd) + _rg_requests(_Dd)],
    "P1-2": [],
    "P1-3": _sb_clock(_SB_VFR) + _sb_clock(_SB_VHC)
            + [_MD_SHIFT[0], _MD_SHIFT[-1], _v_mode[0], _v_mode[-1],
               abs(_SB_VF_N4), _SB_VH_N4],
    "P1-4": [],
    "P1-5": [len(_C4B_ALL)],
    "P2-1": [],
    "P2-2": [_SB_J_N4, _SB_NMAX4],
    "P2-3": [],
    "P4-1": [_SB_I_BASE, _SB_CONC[-1], abs(_SB_I_SPEC)],
    "P4-2": [_PLATEAU[0], _PLATEAU[-1], _SB_K_C4, _SB_CONC[1],
             abs(_SB_K_C8), _SB_CONC[-1]],
    "P4-3": [_SB_NMAX4, _SB_L_N4, _SB_VH_N4],
    "P3-1": [min(_SB_C_SWEEP), max(_SB_C_SWEEP)] + _SB_E_SWEEP,
    "P3-2": [],
    "P3-3": [],
    "P3-4": [_CARD_MIB // 1024],
}
chk("status board: the rows it has", sorted(_SB), sorted(_SB_WANT))
for _sbid in sorted(_SB_WANT):
    _num_row_check(f"status board {_sbid}", _SB[_sbid],
                   [int(_sbid.split("-")[1])] + _SB_WANT[_sbid])


print("\n=== the host probe: a dated reading, and which of it the archive holds ===")
# Fifteen numbers read off two machines on 2026-08-25. Four are in the archive
# already -- every run manifest carries its own `nvidia-smi` line, and it is
# the same card and the same idle reading the probe recorded. Three more are
# the model sizes this repository lists in BENCHMARK_ENV, and three are
# upstream identifiers it also lists. The remaining five are host state that
# nothing here can reproduce, and each of those now carries a dagger. One of
# the five had been quietly wrong: the row said 29 GiB was "too small for the
# 22 GiB target" when this repository's own `ls` twice says the target is 21G.
_HP_SMI = sorted({tuple(_x.strip() for _x in _m["nvidia_smi"].split(",")[:4])
                  for _f in glob.glob(str(_DATA / "*" / "manifest.json"))
                  for _m in [json.loads(pathlib.Path(_f)
                                        .read_text(encoding="utf-8"))]
                  if "nvidia_smi" in _m})
chk("every run manifest reports one card, and the same idle reading",
    _HP_SMI, [("0", "NVIDIA GeForce RTX 3090", "82 MiB", "0 %")])
_HP_CARDS = sorted({len([_l for _l in _m["nvidia_smi"].splitlines() if _l.strip()])
                    for _f in glob.glob(str(_DATA / "*" / "manifest.json"))
                    for _m in [json.loads(pathlib.Path(_f)
                                          .read_text(encoding="utf-8"))]
                    if "nvidia_smi" in _m})
chk("and exactly one of them, which is the `1 x` in that row", _HP_CARDS, [1])
_HP_CARD = int(_HP_SMI[0][1].rsplit(" ", 1)[1])
_HP_MIB = int(_HP_SMI[0][2].split()[0])
_HP_UTIL = int(_HP_SMI[0][3].split()[0])

_BE_FLAT = " ".join(_BE_LINES)


def _hp_size(fname):
    """The size BENCHMARK_ENV's own directory listings give a model file."""
    return sorted({int(_m.group(1)) for _m in
                   re.finditer(r"(\d+)[GM]\s+(?:\S*/)?" + re.escape(fname),
                               _BE_FLAT)})


_HP_TARGET = _hp_size("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
_HP_DRAFT = _hp_size("Qwen3.5-0.8B-Q4_K_M.gguf")
_HP_DFLASH = _hp_size("qwen36-dflash.gguf")
chk("BENCHMARK_ENV lists one size for each model, and lists them more than once",
    (_HP_TARGET, _HP_DRAFT, _HP_DFLASH), ([21], [508], [905]))
chk("the three model paths are the ones the manifests actually loaded",
    sorted({(os.path.basename(_m["target"]), os.path.basename(_m["draft"]))
            for _f in glob.glob(str(_DATA / "A_*" / "manifest.json"))
            for _m in [json.loads(pathlib.Path(_f)
                                  .read_text(encoding="utf-8"))]}),
    [("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "Qwen3.5-0.8B-Q4_K_M.gguf")])
# all three library builds are accounted for: two are the binaries
# BENCHMARK_ENV describes, and the third is v2's, which three other files name.
for _hpn, _hpwhat in ((22105, "the DFlash pull request"),
                      (8889, "the build the legacy binary reports"),
                      (8942, "the build the DFlash binary reports")):
    chk(f"BENCHMARK_ENV also carries {_hpwhat}", str(_hpn) in _BE_FLAT, True)
_HP_V2 = json.loads((pathlib.Path(__file__).resolve().parents[1]
                     / "analysis" / "verbose_accounting.json")
                    .read_text(encoding="utf-8"))[0]["build"]
chk("the third build is v2's, which is why it is not in BENCHMARK_ENV",
    (_HP_V2.split("-")[0], "8863" in _BE_FLAT), ("b8863", False))
chk("and ERRATA names it as the build the v2 controls ran on",
    "`v2_controls/` are `" + _HP_V2 + "`." in _ER_LINES_TEXT, True)

# read off the two hosts on 2026-08-25 and pinned here rather than derived:
# nothing in this archive can reproduce a free-disk figure, another machine's
# GPU occupancy, or a CUDA version. The assertion is that the document has not
# lost them, which is what the dagger beside each one says.
_HP_IDS = (22105, 8863, 8889, 8942)     # a PR and three library builds
_HP_READ = {"bench host free GiB": 262, "this box free GiB": 29,
            "this box cards": 1, "this box card": 3090,
            "this box GPU GiB in use": 20.2, "this box CUDA": 13.3}
_HP = _num_rows(_RT_LINES, "| | `3090` (100.112.135.98)")
chk("host probe: the rows it has",
    sorted(_HP), ["DFlash drafter", "GPU", "disk free", "draft model",
                  "driver", "gguf tooling", "llama-completion", "llama.cpp",
                  "target model", "toolchain"])
_HP_ROW = {_l.strip("|").split("|")[0].strip().replace("`", ""): _l
           for _l in _RT_LINES if _l.strip().startswith("|") and _l.count("|") == 4}
_HP_ROW = {_k: _v for _k, _v in _HP_ROW.items() if _k in _HP}
chk("host probe: the text found is the row that was parsed",
    {_k: [_x[2] for _sp in _tcovn._pipe_spans(_v)
          for _x in _tcovn._numbers_in(_v, _sp)]
     for _k, _v in _HP_ROW.items()}, _HP)
# the four figures nothing here can reproduce, and the two driver strings
# beside them. Each is pinned to the checker rather than derived, which is
# what the dagger says; the assertion is that the document has not lost it.
for _hprow, _hpfig, _hpname in (
        ("GPU", "1 × RTX 3090 †", "this box's card"),
        ("GPU", "**20.2 GiB used** †", "this box's GPU in use"),
        ("driver", "580.173.02 †", "the bench host's driver"),
        ("driver", "610.43.02 †", "this box's driver"),
        ("disk free", "**262 GiB** †", "the bench host's free disk"),
        ("disk free", "29 GiB †", "this box's free disk"),
        ("toolchain", "CUDA 13.3 †", "this box's CUDA")):
    chk(f"host probe: {_hpname} is marked as read off the host",
        _hpfig in _HP_ROW[_hprow], True)
chk("host probe: and the note says what the dagger means",
    ("Every figure in the `thc1006-debian13` column is one, because nothing "
     "here was measured on that machine"
     in " ".join("\n".join(_RT_LINES).split())), True)
chk("host probe: the superseded target size is gone",
    "too small for the 22 GiB target" in "\n".join(_RT_LINES), False)

_HP_WANT = {
    # the second column's card is pinned, not derived: the bench host's
    # manifests say nothing about the machine this box is
    "GPU": [_HP_CARDS[0], _HP_CARD, _HP_MIB, _HP_UTIL,
            _HP_READ["this box cards"], _HP_READ["this box card"],
            _HP_READ["this box GPU GiB in use"]],
    "driver": [],
    "disk free": [_HP_READ["bench host free GiB"],
                  _HP_READ["this box free GiB"],
                  _HP_TARGET[0], _HP_DRAFT[0], _HP_DFLASH[0]],
    "target model": [],
    "draft model": [],
    "DFlash drafter": [],
    "llama.cpp": list(_HP_IDS),
    "toolchain": [_HP_READ["this box CUDA"]],
    "llama-completion": [],
    "gguf tooling": [],
}
chk("host probe: every row is accounted for", sorted(_HP), sorted(_HP_WANT))
# and the accounting the entry publishes for them, which is the whole claim
# that a table of host readings can be checked at all
_HP_N = sum(len(_v) for _v in _HP_WANT.values())
_HP_SIZES = _HP_TARGET + _HP_DRAFT + _HP_DFLASH
chk("host probe: seventeen figures, and what each of them rests on",
    (_HP_N, len(_HP_READ), len(_HP_SIZES), len(_HP_IDS),
     _HP_N - len(_HP_READ) - len(_HP_SIZES) - len(_HP_IDS)),
    (17, 6, 3, 4, 4))
chk("host probe: and both documents account for them the same way",
    (("of its seventeen figures, four turn out to be in the archive"
      in " ".join(_ER_LINES_TEXT.split())),
     ("Of its seventeen figures, four turn out to be in the archive"
      in " ".join(_PR.split()))), (True, True))
for _hpid in sorted(_HP_WANT):
    _num_row_check(f"host probe {_hpid}", _HP[_hpid], _HP_WANT[_hpid])


print("\n=== how much of what is published is checked ===")
# Six times a figure has been computed here, compared against a literal, and
# printed into a table that nothing read; planting a wrong number in the table
# passed every check. Each was found by accident. `analysis/table_coverage.py`
# counts them instead, and these assertions stop the count getting worse: a new
# table has to be either parsed or accounted for, because `carrying_values` is
# exact and `parsed` may only rise.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import table_coverage as _tcov                                    # noqa: E402
_cov = _tcov.census()
chk("coverage: published tables", _cov["tables"], 139)
chk("coverage: those carrying measurements", _cov["carrying_values"], 127)
# EXACT, not "may only rise". `parsed >= 119` and `not_parsed <= 67` were a
# ratchet that permitted six unparsed measurement tables to stand indefinitely,
# and the census that fed them filed any table with fewer than three numeric
# cells as prose -- so a two-value result table could sit outside the population
# and outside the ratchet at the same time. Every table carrying a measurement
# is parsed now, and the gate says so rather than bounding it.
chk("coverage: every measurement table is parsed", _cov["parsed"], _cov["carrying_values"])
chk("coverage: none left unparsed", _cov["not_parsed"], 0)
chk("coverage: every table is one of the four",
    _cov["parsed"] + _cov["not_parsed"] + _cov["no_values"] + _cov["excluded_tables"],
    _cov["tables"])
# An exclusion is a claim about a table that exists. One that matches nothing is
# a claim about a table that does not, and it would quietly shrink the
# population it was written to keep honest.
chk("coverage: no stale table exclusion", _cov["excluded_stale"], [])
_repo = pathlib.Path(__file__).resolve().parents[1]
chk("coverage: no document is listed twice",
    sorted(_tcov.DOCS), sorted(set(_tcov.DOCS)))
chk("coverage: every censused document exists",
    [_d for _d in _tcov.DOCS if not (_repo / _d).exists()], [])
# fail closed: a markdown file in neither list would escape the count in
# silence, which is the shape of every defect this section is about
chk("coverage: every markdown file is either censused or excluded with a reason",
    sorted(str(_f.relative_to(_repo)) for _f in _repo.rglob("*.md")
           if ".git" not in _f.parts),
    sorted(set(_tcov.DOCS) | set(_tcov.EXCLUDED)))
chk("coverage: and no exclusion is left unexplained",
    [_k for _k, _v in _tcov.EXCLUDED.items() if not _v.strip()], [])
chk("coverage: parsed cell by cell, restated after the exclusions",
    _cov["parsed"] >= 57, True)

# A19 publishes the census, so the census has to be what A19 says. The probe
# figures beside them - 67 unguarded of 80, 44 of 44 parsed caught, 40 of 40
# prose - are NOT re-derived here: at about twenty seconds a table the probe is
# close to an hour and it needs a git worktree. The three commands that
# reproduce them are named in the entry, and this pins the half that is cheap.
_A19 = _ER_LINES_TEXT.split("### A19.")[1].split("\n### ")[0]
# A bolded run of digits, thousands space allowed. The `strip()` guard is not
# decoration: `**...tables.** **1 155**` puts one space between two bold marks,
# and a pattern that accepts spaces pairs them and captures the space instead
# of the number - which is how the first version of this line crashed rather
# than quietly matching nothing.
# the thousands separator here is a thin space, so a pattern that accepts only
# an ASCII one silently matches nothing and the check passes on an empty list
_A19N = [int(re.sub(r"[\s\u2009\u00a0]", "", _x)) for _x in
         re.findall(r"\*\*([\d][\d\s\u2009\u00a0]*)\*\*", _A19) if _x.strip()]
# the count of corrections is a number in a document too, so it is checked
# against the list it introduces rather than kept in step by hand
_A19_LIST = re.findall(r"^(\d+)\. ", _A19, re.M)
_WORDS = {20: "Twenty", 21: "Twenty-one", 22: "Twenty-two",
          23: "Twenty-three", 24: "Twenty-four", 25: "Twenty-five",
          26: "Twenty-six", 27: "Twenty-seven", 28: "Twenty-eight",
          29: "Twenty-nine", 30: "Thirty", 31: "Thirty-one",
          32: "Thirty-two", 33: "Thirty-three", 34: "Thirty-four",
          35: "Thirty-five", 36: "Thirty-six", 37: "Thirty-seven",
          38: "Thirty-eight", 39: "Thirty-nine", 40: "Forty",
          41: "Forty-one", 42: "Forty-two", 43: "Forty-three",
          44: "Forty-four", 45: "Forty-five", 46: "Forty-six",
          47: "Forty-seven", 48: "Forty-eight", 49: "Forty-nine",
          50: "Fifty"}
chk("ERRATA A19: the corrections are numbered from one, without a gap",
    [int(_x) for _x in _A19_LIST], list(range(1, len(_A19_LIST) + 1)))
chk("ERRATA A19: and the count it states is the length of that list",
    f"**{_WORDS[len(_A19_LIST)]} published statements were wrong" in _A19, True)


# the changelog publishes the same census and had no check on it, which is
# how it kept saying 119 while ERRATA moved
_CL_FLAT = " ".join("\n".join(_CH_LINES).split())
chk("CHANGELOG: the census it publishes is the census",
    (f"Of {_cov['tables']} tables, {_cov['carrying_values']} carry measurements "
     f"and all {_cov['parsed']} are parsed cell by cell." in _CL_FLAT,
     f"all {80 - _cov['not_parsed']} of those are parsed now" in _CL_FLAT),
    (True, True))
# and the arithmetic beside it: 44 tables were parsed when the entry was first
# written, eight more turned out to have been parsed all along, and the rest
# are readers written since. The 119 in this paragraph went stale unnoticed
# because nothing subtracted anything.
chk("CHANGELOG: eighty more are parsed, eight of them census corrections",
    (_cov["parsed"] - 44, _cov["parsed"] - 44 - 8), (83, 75))
chk("CHANGELOG: and that is the split it publishes",
    "Eighty-three tables are parsed that were not, eight of them census "
    "corrections and seventy-five new readers" in _CL_FLAT, True)
chk("ERRATA A19: the census it publishes is the census",
    (_cov["tables"], _cov["no_values"], _cov["excluded_tables"],
     _cov["carrying_values"], _cov["parsed"], _cov["not_parsed"]),
    (139, 7, 5, 127, 127, 0))
for _want, _what in ((_cov["tables"], "tables"),
                     (_cov["carrying_values"], "carrying measurements"),
                     (_cov["parsed"], "parsed"),
                     (_cov["not_parsed"], "not parsed"),
                     (_cov["no_values"], "no number at all"),
                     (_cov["excluded_tables"], "named exclusions")):
    chk(f"ERRATA A19 prints the {_what} count", _want in _A19N, True)
chk("ERRATA A19: it names the three commands that reproduce the probe figures",
    ("--probe`" in _A19 and "--probe --covered`" in _A19
     and "--prose --probe`" in _A19), True)
chk("ERRATA A19: and says the probe is not run in CI",
    "not run in CI" in _A19, True)

# the prose half, which is the larger one. Cheap to census, so it is pinned the
# same way; the sampled probe behind it is not, for the reason above.
# Prose census locations must be unique. They were {doc, line, value}, and
# `prose_probe` rewrote the first occurrence of `value` on that line -- so two
# identical decimals on one line gave two records that both perturbed the first,
# and the second number was counted as probed and never touched. 28 records are
# in that shape.
_PN = _tcov.prose_numbers()
chk("prose probe: every census record has an exact span",
    [n for n in _PN if n.get("start") is None or n.get("end") is None], [])
chk("prose probe: and the spans are unique",
    len({(n["doc"], n["line"], n["start"], n["end"]) for n in _PN}), len(_PN))
chk("prose probe: records whose value repeats on its own line",
    sum(1 for n in _PN
        if sum(1 for m in _PN if m["doc"] == n["doc"] and m["line"] == n["line"]
               and m["value"] == n["value"]) > 1), 30)

_pcov = _tcov.prose_census()
chk("coverage: decimal numbers in prose, outside every table",
    _pcov["prose_numbers"], 1312)
chk("coverage: those that are not a literal in this file",
    _pcov["not_a_literal"], 677)
# tested on a supplied source, not by searching this file for a phrase: the
# first version of this check searched for a label's own words, and the search
# string was itself a literal in the argument position, so it always found it.
_lbl_src = 'chk("A LABEL", "A VALUE", 1)'
chk("coverage: a label is not counted as a literal and its arguments are",
    ("A LABEL" in _tcov._checker_literals(_lbl_src),
     "A VALUE" in _tcov._checker_literals(_lbl_src)), (False, True))
# DERIVED, not literals. These were `(1226, ...), (647, ...)` typed in by
# hand, and the check asked only whether that number appears in the entry --
# so when the census moved to 1239 and the `_pcov` pin above was updated
# without the document, A19 kept publishing 1226 and nothing failed. The two
# copies of one number have to be the same copy.
for _want, _what in ((_pcov["prose_numbers"], "prose count"),
                     (_pcov["not_a_literal"], "count that are not literals"),
                     (_tcov.PROSE_SAMPLE, "sample size")):
    chk(f"ERRATA A19 prints the {_what}", _want in _A19N, True)
# The interval beside that sample was stated and never derived. Wilson, not
# the normal approximation: at 40 of 40 the normal one has zero width and says
# the population is 100 % unguarded, which is why the entry uses this one.


def _wilson_low(_k, _n, _z=1.959964):
    _ph = _k / _n
    _c = (_ph + _z * _z / (2 * _n)) / (1 + _z * _z / _n)
    _h = ((_z / (1 + _z * _z / _n))
          * math.sqrt(_ph * (1 - _ph) / _n + _z * _z / (4 * _n * _n)))
    return _c - _h


# The size of the probe's population is a published number too, and nothing
# read it: A19, the changelog and the pull-request body all print 2 373 and
# 125, and until now each was a literal typed in by hand after reading the
# tool's output. `cell_population` is what the probe itself would perturb.
_A19_POP = _tcov.cell_population(_cov["covered"])
chk("A19: the probe population it publishes is the one the tool would perturb",
    (_A19_POP, len(_cov["covered"])), (2415, 127))
# not `_grouped`, which returns the digit groups a table cell splits into:
# this is prose, and the question is whether the sentence contains the number.
# `str.split()` treats the thin space as whitespace, so normalising both sides
# makes the needle match whichever separator the document used.
_A19_PRINTED = (f"{_A19_POP // 1000} {_A19_POP % 1000:03d}"
                if _A19_POP >= 1000 else str(_A19_POP))
for _a19doc, _a19txt in (("ERRATA A19", _A19),
                         ("CHANGELOG", _CL_FLAT),
                         ("PR body", _PR)):
    # bold markers removed and the whole padded, so the count is matched as a
    # token: `**125**` is the same figure as `125`, and a bare `in` test on
    # "125" would also be satisfied by the 125 inside some other number
    _a19flat = " " + " ".join(_a19txt.replace("*", " ").split()) + " "
    chk(f"{_a19doc}: prints that population and the table count with it",
        (f" {_A19_PRINTED} " in _a19flat
         and f" {len(_cov['covered'])} " in _a19flat), True)
chk("ERRATA A19: the interval it publishes is Wilson's, at the count measured",
    math.floor(100 * _wilson_low(36, _tcov.PROSE_SAMPLE)), 76)
chk("ERRATA A19: and the entry states that pair",
    ("**36 of 40 accepted a wrong number**" in _A19
     and "**76 % or\nabove**" in _A19), True)
# the split of what this pass parsed, which the entry and the changelog give
# separately and which went stale in the entry at seventy-five
chk("ERRATA A19: the newly parsed count is the census's difference too",
    ("**What this pass changed.** Eighty-three tables count as parsed that "
     "did not: eight are the census correction above and seventy-five are "
     "new readers." in " ".join(_A19.split())), True)
chk("ERRATA A19: and it says the measurement was wrong four times, once",
    (_A19.count("The measurement was wrong"),
     "wrong four times before it measured anything" in _A19), (1, True))
chk("ERRATA A19: the sample size it names is the one the tool draws",
    _tcov.PROSE_SAMPLE, 40)
chk("ERRATA A19: and the seed it names is the one the tool uses",
    (_tcov.PROSE_SEED, str(_tcov.PROSE_SEED) in _A19), (20260828, True))

# A helper this file defines and later rebinds is a defect that hides. `agg`
# was rebound to a float by a loop target 230 lines after its definition, and
# the v1 table block written below it failed with "'float' object is not
# callable" - the lucky outcome. A rebinding to something still callable would
# have checked the wrong numbers in silence.
#
# The condition is exact rather than blunt: a call is a defect only if a
# rebinding sits between the definition that call would resolve to and the call
# itself. Ten names in this file are both a def and a loop target somewhere,
# and none of those ten is a defect - every one of them is reused after the
# last call, or rebound before the def. Asserting on the blunt version would
# have meant ten renames that changed nothing.
_defs = {}
for _n in _tree.body:
    if isinstance(_n, _ast.FunctionDef):
        _defs.setdefault(_n.name, []).append(_n.lineno)
_rebound = {}
for _n in _ast.walk(_tree):
    _tg = ([_n.target] if isinstance(_n, (_ast.For, _ast.comprehension))
           else _n.targets if isinstance(_n, _ast.Assign) else [])
    for _t in _tg:
        for _x in _ast.walk(_t):
            if isinstance(_x, _ast.Name) and _x.id in _defs:
                _rebound.setdefault(_x.id, []).append(_x.lineno)
_stale = set()
for _n in _ast.walk(_tree):
    if not (isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name)
            and _n.func.id in _defs):
        continue
    _before = [_l for _l in _defs[_n.func.id] if _l < _n.lineno]
    if _before and any(max(_before) < _l < _n.lineno
                       for _l in _rebound.get(_n.func.id, [])):
        _stale.add(f"{_n.func.id}:{_n.lineno}")
chk("checker: no call reaches a helper that was rebound after its def",
    sorted(_stale), [])
chk("checker: and the audit is not vacuous - names at risk are found",
    len(set(_rebound)) >= 8, True)

chk("checker: no chk() compares literals with literals",
    (len(_literal_only), _literal_only[:3]), (0, []))
chk("checker: and none compares an expression with itself",
    (len(_self_compare), _self_compare[:3]), (0, []))
chk("checker: number of assertions", len([1 for _n in _ast.walk(_tree)
     if isinstance(_n, _ast.Call) and isinstance(_n.func, _ast.Name) and _n.func.id == "chk"]) > 150, True)

# Last, because it needs every other assertion to have run: the PR body quotes
# how many there are, and a stale figure there is the same defect as a stale
# figure in any other published table.
#
# Eight of them are the git-provenance checks, which do not run where there is
# no `.git` - a mirror, or a shallow clone. So the count the body publishes is
# the one a full checkout produces, and this compares against that rather than
# pretending the two are the same number. `tests/data_mutate.py` runs the
# checker in exactly such a mirror, which is how the difference surfaced.
_GITLESS_SKIPPED = 13
_pr_total = len(RAN) + 1 + (0 if _HAS_GIT else _GITLESS_SKIPPED)
chk("PR body: the assertion count it quotes is a full checkout's",
    f"# {_pr_total} assertions" in _PR, True)


print(f"\n{'='*70}\n{'ALL CLAIMS VERIFIED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}\n{'='*70}")
sys.exit(1 if FAIL else 0)
