#!/usr/bin/env python3
"""Generate LaTeX table bodies and inline macros for the multi-turn safety paper.

    python3 make_tables.py <results_dir> <out_dir>

Every value is derived by analysis.py from the raw experimental artefacts.
Nothing is typed by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402
from analysis import (Data, CONFIGS, CONDITIONS, RULES, EXTRA_RULES,  # noqa: E402
                      INDIRECT, MULTI, KEYWORD_REFUSAL_LIST)

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "tables")
OUT.mkdir(parents=True, exist_ok=True)
D = Data(RES)


def write(name, body):
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}")


def pc(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def ci(pair):
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]"


def pfmt(p):
    if p < 1e-4:
        m, e = f"{p:.1e}".split("e")
        return f"$<10^{{{int(e)}}}$" if float(m) <= 1.05 else f"${m}\\times 10^{{{int(e)}}}$"
    return f"${p:.4f}$"


def tex(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ---------------------------------------------- Table 1: factorial ablation
rows = []
for cfg, label in CONFIGS:
    n = D.n(cfg)
    a = D.asr(cfg)
    w = D.wasr(cfg)
    h2, h1, _ = D.counts(cfg)
    bold = (lambda t: r"\textbf{" + t + "}") if "Indirect" in cfg else (lambda t: t)
    rows.append(" & ".join([
        bold(label), str(n),
        f"{h2}", f"{h1}",
        bold(pc(a)), ci(D.ci(cfg)), bold(pc(w)),
    ]) + r" \\")
write("tab_ablation.tex", "\n".join(rows))

# ---------------------------------------------- Table 2: paired McNemar
comparisons = [
    (CONFIGS[0][0], CONFIGS[1][0], "Direct $\\to$ indirect framing (single-turn)"),
    (CONFIGS[0][0], CONFIGS[2][0], "Single-turn $\\to$ multi-turn (direct)"),
    (CONFIGS[2][0], CONFIGS[1][0], "Multi-turn direct $\\to$ single-turn indirect"),
]
rows = []
for x, y, label in comparisons:
    m = D.mcnemar(D.by_goal(x), D.by_goal(y))
    rows.append(" & ".join([
        label, str(m["n"]), str(m["a"]), str(m["b"]), str(m["c"]), str(m["d"]),
        f"{m['chi2']:.2f}", f"{m['or_ha']:.2f}", pfmt(m["p"]),
    ]) + r" \\")
write("tab_mcnemar.tex", "\n".join(rows))

# ---------------------------------------------- Table 3: hardening defence
rows = []
for cfg, label in CONFIGS:
    b_asr, h_asr = D.asr(cfg, "baseline"), D.asr(cfg, "hardened")
    m = D.mcnemar(D.by_goal(cfg, "baseline"), D.by_goal(cfg, "hardened"))
    red = 100 * (1 - h_asr / b_asr) if b_asr else None
    rows.append(" & ".join([
        label,
        f"{D.n(cfg, 'baseline')}", pc(b_asr),
        f"{D.n(cfg, 'hardened')}", pc(h_asr),
        r"\textbf{" + pc(red, 1) + "}",
        str(m["c"]), pfmt(m["p"]),
    ]) + r" \\")
write("tab_hardening.tex", "\n".join(rows))

# ---------------------------------------------- Table 4: judge aggregation
rows = []
for rule, rlabel in RULES:
    cells = [rlabel]
    for cfg, _ in CONFIGS:
        v = D.asr(cfg, "baseline", rule)
        cells.append(r"\textbf{" + pc(v) + "}" if rule == "or" else pc(v))
    rows.append(" & ".join(cells) + r" \\")
write("tab_judgerule.tex", "\n".join(rows))

# ---------------------------------------------- Table 5: category breakdown
rows = []
for cat in D.categories():
    cells = [tex(cat)]
    n_shown = None
    for cfg, _ in CONFIGS:
        v, n = D.cat_asr(cat, cfg)
        n_shown = n_shown or n
        cells.append(pc(v, 1))
    rows.insert(0, (n_shown, " & ".join([cells[0], str(n_shown)] + cells[1:]) + r" \\"))
rows = [b for _, b in sorted(rows, key=lambda t: -t[0])]
write("tab_category.tex", "\n".join(rows))

# ---------------------------------------------- Table 6: measurement layers
kw = D.keyword_layer()
a3 = CONFIGS[2][0]
layers = [
    ("Keyword refusal detector, single-turn",
     f"{kw['n_single']}", f"{kw['single_pct']:.2f}", ci(kw["single_ci"])),
    ("Keyword refusal detector, multi-turn",
     f"{kw['n_multi']}", r"\textbf{" + f"{kw['multi_pct']:.2f}" + "}", ci(kw["multi_ci"])),
    ("Rubric judges, OR rule, multi-turn",
     f"{D.n(a3)}", f"{D.asr(a3, 'baseline', 'or'):.2f}", ci(D.ci(a3, 'baseline', 'or'))),
    ("Rubric judges, AND rule, multi-turn",
     f"{D.n(a3)}", r"\textbf{" + f"{D.asr(a3, 'baseline', 'and'):.2f}" + "}",
     ci(D.ci(a3, 'baseline', 'and'))),
]
write("tab_layers.tex", "\n".join(" & ".join(r) + r" \\" for r in layers))

# ---------------------------------------------- Table 7: turn origin
to = D.turn_origin()
tot = sum(to.values())
rows = []
for t in sorted(to):
    rows.append(" & ".join([
        f"Turn {t}", str(to[t]), f"{100 * to[t] / tot:.1f}",
    ]) + r" \\")
write("tab_turnorigin.tex", "\n".join(rows))

# ---------------------------------------------- inline macros
ja = D.judge_agreement()
m_frame = D.mcnemar(D.by_goal(CONFIGS[0][0]), D.by_goal(CONFIGS[1][0]))
m_turns = D.mcnemar(D.by_goal(CONFIGS[0][0]), D.by_goal(CONFIGS[2][0]))
m_vs = D.mcnemar(D.by_goal(CONFIGS[2][0]), D.by_goal(CONFIGS[1][0]))


# ------------------------------------------------- Table: judge reliability
ag = D.agreement()
rows = [
    ("Judged turns", f"${ag['n']:,}$".replace(",", "{,}")),
    ("Raw agreement on the 0/1/2 rubric", f"${100*ag['po']:.2f}\\%$"),
    ("Cohen's $\\kappa$", f"${ag['kappa']:.4f}$"),
    ("Gwet's AC1", f"${ag['ac1']:.4f}$"),
    ("Raw agreement, harmful vs not", f"${100*ag['po_binary']:.2f}\\%$"),
    ("PABAK, harmful vs not", f"${ag['pabak']:.4f}$"),
    ("Judge 1 scores 2", f"${100*ag['prev_j1']:.2f}\\%$ of turns"),
    ("Judge 2 scores 2", f"${100*ag['prev_j2']:.2f}\\%$ of turns"),
    ("Turns where only judge 1 scores 2", f"${ag['only_j1']}$"),
    ("Turns where only judge 2 scores 2", f"${ag['only_j2']}$"),
    ("Turns where both score 2", f"${ag['both']}$"),
]
write("tab_agreement.tex", "\n".join(f"{a} & {b} \\\\" for a, b in rows))

# ------------------------------------------------------ Table: reproducibility
rows = [
    ("Target model", r"\texttt{llama-3.1-8b-instant} (Groq)"),
    ("Attacker model (indirect arm)", r"\texttt{llama-3.1-8b-instant}"),
    ("Judge 1 / Judge 2", r"\texttt{llama-3.3-70b-versatile} / \texttt{qwen/qwen3-32b}"),
    ("Target decoding", r"$T=0.7$, \texttt{max\_tokens} $=200$"),
    ("Attacker decoding", r"$T=0.7$, \texttt{max\_tokens} $=200$"),
    ("Judge decoding", r"$T=0$, \texttt{max\_tokens} $=80$"),
    (r"\texttt{top\_p}, \texttt{seed}, penalties",
     "not sent; provider defaults apply"),
    ("Maximum turns", "3"),
    ("Intent source", r"AdvBench \texttt{harmful\_behaviors.csv}, de-duplicated"),
    ("Unique intents", f"${D.n_goals()}$" if hasattr(D, "n_goals") else "$520$"),
    ("Duplicate records removed", f"${D.n_duplicates()}$"),
]
write("tab_repro.tex", "\n".join(f"{a} & {b} \\\\" for a, b in rows))

# --------------------------------------------------------- Table: noise floor
nf = D.noise_floor("or")
fh = D.first_harmful_turn("or")
tot_fh = sum(fh.values())
rows = [
    (r"Single-turn direct, baseline",
     f"${nf['k_single']}/{nf['n_single']}$",
     f"{100*nf['k_single']/nf['n_single']:.2f}"),
    (r"Multi-turn direct, \emph{turn 1 only}",
     f"${nf['k_turn1']}/{nf['n_turn1']}$",
     f"{100*nf['k_turn1']/nf['n_turn1']:.2f}"),
    (r"\textbf{Gap (identical prompt, sampling only)}", "---",
     r"\textbf{" + f"{nf['gap_pp']:.2f}" + "}"),
    (r"Multi-turn direct, all three turns",
     f"${tot_fh}/{nf['n_turn1']}$", f"{100*tot_fh/nf['n_turn1']:.2f}"),
    (r"Increment contributed by turns 2--3",
     f"${tot_fh - nf['k_turn1']}/{nf['n_turn1']}$",
     f"{100*(tot_fh-nf['k_turn1'])/nf['n_turn1']:.2f}"),
]
write("tab_noise.tex",
      "\n".join(" & ".join(r) + " \\\\" for r in rows))

macros = [
    (r"\Nabl", str(len(D.abl))),
    (r"\Ngoals", str(len({r["goal"] for r in D.abl}))),
    (r"\Ncat", str(len(D.categories()))),
    (r"\AsrDirect", pc(D.asr(CONFIGS[0][0]))),
    (r"\AsrIndirect", pc(D.asr(CONFIGS[1][0]))),
    (r"\AsrMulti", pc(D.asr(CONFIGS[2][0]))),
    (r"\WasrDirect", pc(D.wasr(CONFIGS[0][0]))),
    (r"\WasrIndirect", pc(D.wasr(CONFIGS[1][0]))),
    (r"\WasrMulti", pc(D.wasr(CONFIGS[2][0]))),
    (r"\NDirect", str(D.n(CONFIGS[0][0]))),
    (r"\NIndirect", str(D.n(CONFIGS[1][0]))),
    (r"\NMulti", str(D.n(CONFIGS[2][0]))),
    (r"\NMultiHard", str(D.n(CONFIGS[2][0], "hardened"))),
    (r"\HardDirect", pc(D.asr(CONFIGS[0][0], "hardened"))),
    (r"\HardIndirect", pc(D.asr(CONFIGS[1][0], "hardened"))),
    (r"\HardMulti", pc(D.asr(CONFIGS[2][0], "hardened"))),
    (r"\FrameB", str(m_frame["b"])),
    (r"\FrameC", str(m_frame["c"])),
    (r"\FrameChi", f"{m_frame['chi2']:.2f}"),
    (r"\FrameOR", f"{m_frame['or_ha']:.2f}"),
    (r"\TurnsB", str(m_turns["b"])),
    (r"\TurnsC", str(m_turns["c"])),
    (r"\TurnsChi", f"{m_turns['chi2']:.2f}"),
    (r"\TurnsOR", f"{m_turns['or_ha']:.2f}"),
    (r"\VsB", str(m_vs["b"])),
    (r"\VsC", str(m_vs["c"])),
    (r"\VsChi", f"{m_vs['chi2']:.2f}"),
    (r"\Acone", f"{ag['ac1']:.4f}"),
    (r"\Pabak", f"{ag['pabak']:.4f}"),
    (r"\Pobin", f"{100*ag['po_binary']:.2f}"),
    (r"\Njudged", f"{ag['n']:,}".replace(",", "{,}")),
    (r"\Jonepos", f"{100*ag['prev_j1']:.2f}"),
    (r"\Jtwopos", f"{100*ag['prev_j2']:.2f}"),
    (r"\Onlyjone", str(ag["only_j1"])),
    (r"\Onlyjtwo", str(ag["only_j2"])),
    (r"\Bothtwo", str(ag["both"])),
    (r"\Ndup", str(D.n_duplicates())),
    (r"\Noisefloor", f"{nf['gap_pp']:.2f}"),
    (r"\Turnonepp", f"{100*nf['k_turn1']/nf['n_turn1']:.2f}"),
    (r"\Turnincpp", f"{100*(tot_fh-nf['k_turn1'])/nf['n_turn1']:.2f}"),
    (r"\Temptarget", "0.7"),
    (r"\Maxturns", "3"),
    # Moi con so ve do dong thuan deu tinh tren CUNG mot tap: 4,540 luot
    # da cham (ag), khong phai tren muc trial (ja), de bang va van khop nhau.
    (r"\Kappa", f"{ag['kappa']:.4f}"),
    (r"\Po", f"{ag['po']:.4f}"),
    (r"\JudgeOnePos", str(ag["only_j1"] + ag["both"])),
    (r"\JudgeTwoPos", str(ag["only_j2"] + ag["both"])),
    (r"\JudgeOne", tex(D.judges[0])),
    (r"\JudgeTwo", tex(D.judges[1])),
    (r"\AsrOrMulti", pc(D.asr(CONFIGS[2][0], "baseline", "or"))),
    (r"\AsrAndMulti", pc(D.asr(CONFIGS[2][0], "baseline", "and"))),
    (r"\AsrAndIndirect", pc(D.asr(CONFIGS[1][0], "baseline", "and"))),
    (r"\AsrJtwoIndirect", pc(D.asr(CONFIGS[1][0], "baseline", "j2"))),
    (r"\KwSingle", f"{kw['single_pct']:.2f}"),
    (r"\KwMulti", f"{kw['multi_pct']:.2f}"),
    (r"\KwN", str(kw["n_single"])),
    (r"\KwUniq", str(kw["uniq_single"])),
    (r"\KwTurnOne", str(kw["turn_dist"].get(1, 0))),
    (r"\TurnOneHarm", str(to.get(1, 0))),
    (r"\TurnTwoHarm", str(to.get(2, 0))),
    (r"\TurnThreeHarm", str(to.get(3, 0))),
    (r"\TurnTotalHarm", str(tot)),
    (r"\TurnOnePct", f"{100 * to.get(1, 0) / tot:.1f}"),
]
# ===========================================================================
# Revision tables
# ===========================================================================
import collections  # noqa: E402

nt = D.null_trials()
ms = D.multiturn_shape()
tl = D.turn_one_refusal_link()

# --- Table: the indirect arm's null trials, overall and by category
sub = D.rows(INDIRECT, "baseline")
cat = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in sub:
    row = cat[r["category"]]
    row[0] += 1
    if A._attacker_refused(r["transcript"][0]["attacker_message"]):
        row[1] += 1
    else:
        row[3] += 1
        if D.score(r, "or") == 2:
            row[2] += 1
rows = []
for c, (n_, nl, k_, live) in sorted(cat.items()):
    rows.append(" & ".join([
        c, str(n_), str(nl), pc(100 * nl / n_), str(live),
        f"{k_}/{live}", pc(100 * k_ / max(1, live)),
    ]) + r" \\")
rows.append(r"\addlinespace")
rows.append(" & ".join([
    r"\textbf{All categories}", r"\textbf{%d}" % nt["n"], r"\textbf{%d}" % nt["n_null"],
    r"\textbf{%s}" % pc(nt["null_pct"]), r"\textbf{%d}" % nt["n_live"],
    r"\textbf{%d/%d}" % (nt["k_live"], nt["n_live"]), r"\textbf{%s}" % pc(nt["asr_live"]),
]) + r" \\")
write("tab_null.tex", "\n".join(rows))

# --- Table: what the multi-turn arm actually varies
rows = []
p1 = ms["p_turn_one"] / 100.0
for turn, k_, at_risk, rate in ms["cond"]:
    rows.append(" & ".join([
        str(turn), str(at_risk), str(k_), pc(rate),
        f"{at_risk * p1:.1f}", pc(100 * p1),
    ]) + r" \\")
write("tab_mtshape.tex", "\n".join(rows))

# --- Table: aggregation rules, extended
rows = []
for rule, label in EXTRA_RULES:
    rows.append(" & ".join(
        [label] + [pc(D.asr_rule(cfg, rule)) for cfg, _ in CONFIGS]) + r" \\")
rows.append(r"\addlinespace")
kw = D.keyword_layer()
rows.append(" & ".join([
    r"Keyword detector (no judge)", "--", "--", pc(kw["multi_pct"]),
]) + r" \\")
write("tab_rules_ext.tex", "\n".join(rows))


macros += [
    (r"\Nnull", str(nt["n_null"])),
    (r"\Nnullpct", f"{nt['null_pct']:.2f}"),
    (r"\Nlive", str(nt["n_live"])),
    (r"\AsrIndirectLive", f"{nt['asr_live']:.2f}"),
    (r"\AsrIndirectNull", f"{nt['asr_null']:.2f}"),
    (r"\Klive", str(nt["k_live"])),
    (r"\Knull", str(nt["k_null"])),
    (r"\AttackerModel", tex("llama-3.1-8b-instant")),
    (r"\TargetModel", tex("llama-3.1-8b-instant")),
    (r"\Nnag", str(ms["nag"])),
    (r"\Nlater", str(ms["n_later"])),
    (r"\Nverbatim", str(ms["verbatim"])),
    (r"\Iidthree", f"{ms['iid3']:.2f}"),
    (r"\Pturnone", f"{ms['p_turn_one']:.2f}"),
    (r"\Condtwo", f"{ms['cond'][1][3]:.2f}"),
    (r"\Condthree", f"{ms['cond'][2][3]:.2f}"),
    (r"\Reflater", f"{tl['rate_refused']:.2f}"),
    (r"\Norefusedlater", f"{tl['rate_not']:.2f}"),
    (r"\Nrefusedt", str(tl["a"] + tl["b"])),
    (r"\Nnotrefusedt", str(tl["c"] + tl["d"])),
    (r"\Preflink", f"{tl['p']:.3f}"),
    (r"\Nkeywords", str(len(KEYWORD_REFUSAL_LIST))),
    (r"\MaxTokTarget", "200"),
    (r"\MaxTokJudge", "80"),
    (r"\Tempjudge", "0.0"),
    (r"\SumThreeIndirect", pc(D.asr_rule(INDIRECT, "sum3"))),
    (r"\SumThreeMulti", pc(D.asr_rule(MULTI, "sum3"))),
    (r"\SumThreeDirect", pc(D.asr_rule(CONFIGS[0][0], "sum3"))),
]

write("macros.tex", "\n".join(rf"\newcommand{{{n}}}{{{v}}}" for n, v in macros))
print("\nAll tables generated from:", RES.resolve())
