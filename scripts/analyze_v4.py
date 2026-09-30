import json, numpy as np
from scipy import stats

with open('results/pilot_v4/results_v4_raw.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f'Records loaded: {len(records)}')
n_cons = sum(1 for r in records if r.get('attack_success_consensus'))
n_self = sum(1 for r in records if r.get('attack_success_self'))
n = len(records)

for r in records:
    c = r.get('attack_success_consensus')
    s = r.get('attack_success_self')
    t = r.get('turns_used')
    print(f"  id={r['id']} consensus={c} self={s} turns={t}")

print(f"\nSummary: N={n}, consensus_successes={n_cons}, self_successes={n_self}")
