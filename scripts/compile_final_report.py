import json
import re
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent
RESULTS_FILE = PROJECT_DIR / "results" / "pilot_v6" / "results_v6.json"
DRAFT_FILE = PROJECT_DIR / "reports" / "01_deep_alignment_multiturn_draft.md"

def compile_report():
    if not RESULTS_FILE.exists():
        print(f"ERROR: Cannot find {RESULTS_FILE}")
        return
        
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    summary = data.get("summary", {})
    if not summary:
        summary = {}
        for k, v in data.items():
            if "Config-A" in k:
                summary["Config-A"] = v
            elif "Config-B" in k:
                summary["Config-B"] = v
            elif "Config-C" in k:
                summary["Config-C"] = v
                
    cfg_a = summary.get("Config-A", {})
    cfg_b = summary.get("Config-B", {})
    cfg_c = summary.get("Config-C", {})
    
    def safe_fmt(val, decimals=3):
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    with open(DRAFT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Replacements for Table 1
    content = re.sub(
        r"\| \*\*Config-A\*\* \(Intra-Family MT\) \|.*",
        f"| **Config-A** (Intra-Family MT) | {safe_fmt(cfg_a.get('mean_asr', 0)*100, 1)}% ± {safe_fmt(cfg_a.get('std_asr', 0)*100, 1)}% | **{safe_fmt(cfg_a.get('mean_wasr'))} ± {safe_fmt(cfg_a.get('std_wasr'))}** | {safe_fmt(cfg_a.get('cohens_kappa', cfg_a.get('kappa')))} (Moderate) |",
        content
    )
    content = re.sub(
        r"\| \*\*Config-B\*\* \(Cross-Family MT\) \|.*",
        f"| **Config-B** (Cross-Family MT) | {safe_fmt(cfg_b.get('mean_asr', 0)*100, 1)}% ± {safe_fmt(cfg_b.get('std_asr', 0)*100, 1)}% | **{safe_fmt(cfg_b.get('mean_wasr'))} ± {safe_fmt(cfg_b.get('std_wasr'))}** | {safe_fmt(cfg_b.get('cohens_kappa', cfg_b.get('kappa')))} (Fair) |",
        content
    )
    content = re.sub(
        r"\| \*\*Config-C\*\* \(Intra-Family ST\) \|.*",
        f"| **Config-C** (Intra-Family ST) | {safe_fmt(cfg_c.get('mean_asr', 0)*100, 1)}% ± {safe_fmt(cfg_c.get('std_asr', 0)*100, 1)}% | **{safe_fmt(cfg_c.get('mean_wasr'))} ± {safe_fmt(cfg_c.get('std_wasr'))}** | {safe_fmt(cfg_c.get('cohens_kappa', cfg_c.get('kappa')))} (Substantial) |",
        content
    )

    # Replacements for Table 2
    content = re.sub(
        r"\| \*\*Config-A\*\* \|.*",
        f"| **Config-A** | {safe_fmt(cfg_a.get('stakes_normal_wasr'))} | {safe_fmt(cfg_a.get('stakes_high_wasr'))} | **{safe_fmt(cfg_a.get('stakes_high_wasr', 0) - cfg_a.get('stakes_normal_wasr', 0))}** |",
        content
    )
    content = re.sub(
        r"\| \*\*Config-B\*\* \|.*",
        f"| **Config-B** | {safe_fmt(cfg_b.get('stakes_normal_wasr'))} | {safe_fmt(cfg_b.get('stakes_high_wasr'))} | **{safe_fmt(cfg_b.get('stakes_high_wasr', 0) - cfg_b.get('stakes_normal_wasr', 0))}** |",
        content
    )
    content = re.sub(
        r"\| \*\*Config-C\*\* \|.*",
        f"| **Config-C** | {safe_fmt(cfg_c.get('stakes_normal_wasr'))} | {safe_fmt(cfg_c.get('stakes_high_wasr'))} | **{safe_fmt(cfg_c.get('stakes_high_wasr', 0) - cfg_c.get('stakes_normal_wasr', 0))}** |",
        content
    )

    with open(DRAFT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Successfully compiled live results into {DRAFT_FILE.name}")

if __name__ == "__main__":
    compile_report()
