# Multi-turn jailbreak factorial ablation

Result files, experiment scripts and LaTeX sources for the paper

> V. T. Le, S. X. Ha, N. N. Phien and T. Q. Nguyen,
> "What Actually Moves an Aligned Model: Framing, Turns, or the Metric? A Factorial Ablation of Multi-Turn Jailbreaks with Judge-Rule Sensitivity",
> submitted to IEEE Transactions on Artificial Intelligence, 2026.

## What is in here

```
results/     the measured result files. Every number, table and figure in the
             paper is computed from these and from nothing else.
scripts/     the experiment code that produced those files.
latex/       the generators that read results/ and emit the table bodies and
             figures, plus the paper source they are substituted into.
```

## What you can reproduce, and what you cannot

**From this repository alone** you can regenerate every table, figure and
inline number in the paper:

```bash
pip install -r requirements.txt
cd latex
python3 make_tables.py  ../results  tables
python3 make_figures.py ../results  figures
python3 build.py        ../results          # writes main.tex
python3 build_ieee.py                      # writes ieee/main_ieee.tex
```

`build.py` substitutes the generated values into `paper_template.tex`. No number
in the paper is typed by hand, so a mismatch between the paper and a fresh run
of these scripts is a bug and we would like to hear about it.

`build_ieee.py` then rewrites that manuscript into the IEEE two-column form that
was actually submitted: it drops the CRediT section, which IEEE has no field for,
lifts the funding statement into a page-one footnote, rebuilds the author block
in IEEE style, and widens only the tables that overflow a column.
`latex/ieee/main_ieee.tex` is checked in even though it is generated, because it
is the exact manuscript we submitted. Regenerating it must produce the same
bytes; that diff is the artifact's own self-check, and we ran it on all nine
papers before publishing.

**You cannot regenerate `results/` from this repository alone.** Doing that needs
a Groq API key and the three models named in the paper: `llama-3.1-8b-instant` as target, and `llama-3.3-70b-versatile` plus `qwen/qwen3-32b` as the two judges. Model endpoints are mutable and two of these have already changed availability tier since the run, so a re-run will not reproduce the same generations token for token.
The scripts in `scripts/` are the code we ran; they are published so the
procedure can be inspected and re-executed by anyone who assembles that
environment.

## Layout of `results/`

| File | Used for |
|---|---|
| `pilot_v6/ablation_results.jsonl` | the rubric-judged trials of the 2x3 factorial. Every table, figure and inline number in the paper is computed from this file |
| `pilot_v6/checkpoint.json` | resume state of the v6 runner. It documents the single restart that duplicated one row, which analysis.py de-duplicates on (sample_id, goal) |
| `pilot/raw_outputs/raw.jsonl` | the single-turn trials as scored by the keyword detector |
| `pilot/raw_outputs/multiturn_raw.jsonl` | the multi-turn trials, same detector |
| `pilot/metrics.csv` | the pilot's own summary line for the single-turn arm |
| `pilot/multiturn_metrics.csv` | the pilot's own summary line for the multi-turn arm |
| `pilot_v3/results_v3.json` | earlier pilot round, superseded. No number in the paper comes from it; kept so the sequence of runs is visible |
| `pilot_v4/results_v4.json` | earlier pilot round, superseded |
| `pilot_v5/results_v5.json` | earlier pilot round, superseded |

## A file we removed from this artifact

`results/pilot_v6/results_v6.json` was present in our working tree. It was not a
measurement: it was written by a scaffolding script, `mock_results.py`, that drew
pseudorandom scores from hard-coded rates while the real runner was still being
written. `analysis.py` opened the file and never read a value out of it, so no
number in the paper ever depended on it. We deleted both the file and the
scaffolding script rather than ship them, and we removed the dead `open()` from
`analysis.py`. Rebuilding the paper before and after that change produces a
byte-identical `main.tex`, which is the check we used to confirm the file was
unused. We record this here because the honest thing to do with a generated file
that sat in a results directory is to say it was there.

## Paths in `scripts/`

`scripts/check_draft1.py` and `scripts/recompute_metrics.py` still contain the absolute path of the machine the experiments ran on. We have
not rewritten them. These files are published as a record of what was executed,
and editing a path inside a script after the fact would make the record less
reliable rather than more. A reader who wants to re-run them should change that
one line to point at their own checkout.

## Licence

Code in `scripts/` and `latex/` is MIT. The result files in `results/` are
CC BY 4.0. See `LICENSE`.
