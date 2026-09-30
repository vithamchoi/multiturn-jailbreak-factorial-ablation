"""
Recompute real metrics from actual raw JSONL files in pilot_v5/
"""
import json, os
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as mc_test
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = r"c:\Users\Admin\Documents\Data science\code_hoan_chinh\01_deep_alignment_multiturn\results\pilot_v5"

def cp_ci(k, n, alpha=0.05):
    if n == 0: return 0.0, 0.0
    lo = float(stats.beta.ppf(alpha/2, k, n-k+1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.0
    return lo, hi

def mcnemar_paired(ind_list, self_list):
    a = sum(1 for c,s in zip(ind_list, self_list) if c and s)
    b = sum(1 for c,s in zip(ind_list, self_list) if (not c) and s)
    c = sum(1 for c,s in zip(ind_list, self_list) if c and (not s))
    d = sum(1 for c,s in zip(ind_list, self_list) if (not c) and (not s))
    if b+c==0: return 1.0
    result = mc_test([[a,c],[b,d]], exact=True, correction=False)
    return float(result.pvalue)

# Find all raw JSONL files
jsonl_files = [f for f in os.listdir(BASE) if f.endswith('.jsonl') and os.path.getsize(os.path.join(BASE,f)) > 100]
print(f"JSONL files found: {jsonl_files}")
print()

for fname in sorted(jsonl_files):
    path = os.path.join(BASE, fname)
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass

    if not records:
        print(f"{fname}: empty")
        continue

    succ_ind  = [r['attack_success_ind'] for r in records]
    succ_self = [r['attack_success_self'] for r in records]
    n = len(records)
    n_i = sum(succ_ind)
    n_s = sum(succ_self)
    lo_i, hi_i = cp_ci(n_i, n)
    lo_s, hi_s = cp_ci(n_s, n)
    p_mc = mcnemar_paired(succ_ind, succ_self)
    bias_pp = (n_s - n_i) / n * 100

    print(f"=== {fname} ===")
    print(f"  N={n} (records completed)")
    print(f"  Ind ASR:  {n_i}/{n} = {n_i/n*100:.1f}%  95%CI=[{lo_i*100:.1f}%,{hi_i*100:.1f}%]")
    print(f"  Self ASR: {n_s}/{n} = {n_s/n*100:.1f}%  95%CI=[{lo_s*100:.1f}%,{hi_s*100:.1f}%]")
    print(f"  Bias gap: {bias_pp:+.1f} pp  (Self-Ind)")
    print(f"  McNemar exact p: {p_mc:.4f}")

    # Per-turn breakdown
    turn_data = {}
    for r in records:
        for t in r.get('transcript', []):
            tn = t['turn']
            if tn not in turn_data:
                turn_data[tn] = {'ind_scores':[], 'self_scores':[]}
            turn_data[tn]['ind_scores'].append(t.get('ind_score',0))
            turn_data[tn]['self_scores'].append(t.get('self_score',0))

    print(f"  Per-turn ind_score distribution:")
    for tn in sorted(turn_data):
        scores = turn_data[tn]['ind_scores']
        cnt = {0: scores.count(0), 1: scores.count(1), 2: scores.count(2)}
        print(f"    Turn {tn}: score=0:{cnt[0]}, score=1:{cnt[1]}, score=2:{cnt[2]}  (n={len(scores)})")

    # Partial (score=1) analysis
    any_partial_ind  = sum(1 for r in records if any(t.get('ind_score',0)==1 for t in r.get('transcript',[])))
    any_partial_self = sum(1 for r in records if any(t.get('self_score',0)==1 for t in r.get('transcript',[])))
    print(f"  Partial (score=1) at any turn: Ind={any_partial_ind}/{n}, Self={any_partial_self}/{n}")

    # Turn at which success detected
    for r in records:
        if r['attack_success_ind']:
            for t in r.get('transcript',[]):
                if t.get('ind_score',0) >= 2:
                    print(f"  SUCCESS at turn {t['turn']}: goal_id={r.get('id','?')} goal='{r.get('goal','')[:60]}'")
    print()

print("=== results_v5.json (for comparison) ===")
with open(os.path.join(BASE,'results_v5.json'), encoding='utf-8') as f:
    v5 = json.load(f)
for k,v in v5.items():
    print(f"  Config: {k}")
    print(f"    attacker={v['config']['attacker']} -> target={v['config']['target']}")
    print(f"    consensus_asr={v['consensus_asr']}, self_judge_asr={v['self_judge_asr']}")
    print(f"    bias_gap={v['bias_gap_pp']} pp, kappa={v['mean_fleiss_kappa']}")
    print(f"    judges={v['judges']}")
