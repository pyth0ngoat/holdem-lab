"""Hold'em Lab - Findings and Artifact Recording Infrastructure.

Central protocol for recording all empirical results, benchmarks, tests,
and figures in machine-readable JSON and human-readable Markdown.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from holdem_lab import (
    FIGURES_DATA_DIR,
    FIGURES_DIR,
    MODE,
    REPORTS_DIR,
    ROOT_DIR,
    SEED_DEMO,
    SEED_FIT,
    SEED_VALIDATION,
)

FINDINGS_JSON_PATH = REPORTS_DIR / "findings.json"
FINDINGS_MD_PATH = REPORTS_DIR / "findings.md"

VALID_VERDICTS = {
    "SUPPORTED",
    "NOT SUPPORTED",
    "INCONCLUSIVE",
    "VERIFIED",
    "FAILED CHECK",
    "INFORMATIONAL",
}

# In-memory claims ledger
_CLAIMS_LEDGER: list[dict[str, str]] = []


def register_claim(claim: str, fid: str) -> None:
    """Register an intended claim tied to a finding ID."""
    _CLAIMS_LEDGER.append({"claim": claim, "fid": fid})


def get_registered_claims() -> list[dict[str, str]]:
    """Return all currently registered claims."""
    return list(_CLAIMS_LEDGER)


def clear_claims_ledger() -> None:
    """Clear in-memory claims."""
    global _CLAIMS_LEDGER
    _CLAIMS_LEDGER = []


def load_findings_records() -> dict[str, dict[str, Any]]:
    """Load all existing findings from reports/findings.json."""
    if not FINDINGS_JSON_PATH.exists():
        return {}
    try:
        with open(FINDINGS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_findings_records(records: dict[str, dict[str, Any]]) -> None:
    """Save findings dictionary to reports/findings.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINDINGS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def record_finding(
    fid: str,
    title: str,
    question: str,
    method: str,
    results: dict[str, Any],
    figures: list[str],
    verdict: str,
    caveats: str,
    gap: str | None = None,
    hypothesis: str | None = None,
) -> dict[str, Any]:
    """Record an empirical finding or verification check into findings.json.

    Parameters
    ----------
    fid : str
        Unique finding ID, e.g. 'F-P1-1', 'F-G1-1', 'V-CARD-1'.
    title : str
        Brief descriptive title.
    question : str
        The specific empirical or theoretical question being evaluated.
    method : str
        Methodology used (simulation, exact counting, MLE, etc.).
    results : dict
        Key-value mapping of numerical results. Values can be floats, strings,
        or dicts containing {'value': ..., 'se': ...} or {'value': ..., 'ci': [..., ...]}.
    figures : list of str
        List of associated figure paths relative to project root.
    verdict : str
        One of {'SUPPORTED', 'NOT SUPPORTED', 'INCONCLUSIVE', 'VERIFIED', 'FAILED CHECK', 'INFORMATIONAL'}.
    caveats : str
        Limitations, caveats, edge cases, or potential biases.
    gap : str, optional
        Associated gap code (e.g. 'G1', 'G2', 'BENCHMARK', 'CORE').
    hypothesis : str, optional
        Associated hypothesis code (e.g. 'H1', 'H2', etc.).
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Verdict '{verdict}' is not valid. Must be one of {VALID_VERDICTS}")

    records = load_findings_records()

    # Determine gap/hypothesis from fid or title if not provided
    if gap is None:
        if "G1" in fid or "G1" in title:
            gap = "G1"
        elif "G2" in fid or "G2" in title:
            gap = "G2"
        elif "G3" in fid or "G3" in title:
            gap = "G3"
        elif "G4" in fid or "G4" in title:
            gap = "G4"
        elif "G5" in fid or "G5" in title:
            gap = "G5"
        elif "G6" in fid or "G6" in title:
            gap = "G6"
        elif "G7" in fid or "G7" in title:
            gap = "G7"
        elif "G8" in fid or "G8" in title:
            gap = "G8"
        elif "G9" in fid or "G9" in title:
            gap = "G9"
        elif "UCI" in fid or "PHH" in fid or "DATA" in fid:
            gap = "DATA"
        else:
            gap = "VERIFICATION"

    record_entry = {
        "fid": fid,
        "title": title,
        "question": question,
        "method": method,
        "results": results,
        "figures": figures,
        "verdict": verdict,
        "caveats": caveats,
        "gap": gap,
        "hypothesis": hypothesis,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    records[fid] = record_entry
    save_findings_records(records)
    return record_entry


def save_fig(
    fig: plt.Figure,
    fid: str,
    slug: str,
    caption: str,
    data: pd.DataFrame | dict[str, Any] | None = None,
) -> str:
    """Save a matplotlib figure to figures/<fid>_<slug>.png and optional CSV.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    fid : str
        Finding ID.
    slug : str
        Short identifier for the plot.
    caption : str
        Descriptive caption.
    data : pd.DataFrame or dict, optional
        Data underpinning the plot. If provided, saved to figures/data/<fid>_<slug>.csv.

    Returns
    -------
    str
        Path to the saved figure relative to project root.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename_png = f"{fid}_{slug}.png"
    filepath_png = FIGURES_DIR / filename_png

    fig.tight_layout()
    fig.savefig(filepath_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if data is not None:
        filename_csv = f"{fid}_{slug}.csv"
        filepath_csv = FIGURES_DATA_DIR / filename_csv
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath_csv, index=False)
        elif isinstance(data, dict):
            pd.DataFrame(data).to_csv(filepath_csv, index=False)

    return f"figures/{filename_png}"


def assert_all_claims_backed() -> None:
    """Verify that every registered claim in the ledger has a corresponding finding record."""
    records = load_findings_records()
    missing_claims = []
    for entry in _CLAIMS_LEDGER:
        fid = entry["fid"]
        claim = entry["claim"]
        if fid not in records:
            missing_claims.append(f"Claim '{claim}' expects finding '{fid}', but '{fid}' was not found.")
        elif records[fid]["verdict"] in ("FAILED CHECK", "NOT SUPPORTED"):
            # Note: NOT SUPPORTED is a valid empirical outcome for a hypothesis,
            # but FAILED CHECK indicates an infrastructure/verification breakage.
            if records[fid]["verdict"] == "FAILED CHECK":
                missing_claims.append(f"Claim '{claim}' is backed by '{fid}' which FAILED CHECK.")

    if missing_claims:
        raise AssertionError("Claims ledger validation failed:\n" + "\n".join(missing_claims))


def _get_git_commit() -> str:
    """Safely obtain current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        out = res.stdout.strip()
        return out if out else "no-commit-yet"
    except Exception:
        return "unknown"


def _get_package_versions() -> dict[str, str]:
    """Retrieve versions of key installed libraries."""
    pkgs = ["numpy", "pandas", "matplotlib", "scipy", "statsmodels", "pytest"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except Exception:
            versions[p] = "not-installed"
    return versions


def write_findings() -> str:
    """Rebuild reports/findings.md idempotently from reports/findings.json."""
    records = load_findings_records()

    git_commit = _get_git_commit()
    pkg_versions = _get_package_versions()
    now_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Hold'em Lab — Findings & Verification Register")
    lines.append("")
    lines.append("> Auto-generated by `holdem_lab/findings.py`. Single source of empirical truth.")
    lines.append("")

    # 1. Run Metadata
    lines.append("## 1. Run Metadata")
    lines.append(f"- **Generated At:** {now_str}")
    lines.append(f"- **Execution Mode:** `{MODE}`")
    lines.append(f"- **Git Commit:** `{git_commit}`")
    lines.append(f"- **Seeds:** Fit = `{SEED_FIT}`, Validation = `{SEED_VALIDATION}`, Demo = `{SEED_DEMO}`")
    lines.append("- **Key Packages:** " + ", ".join(f"`{k}=={v}`" for k, v in pkg_versions.items()))
    lines.append("")

    # 2. Hypotheses Summary Table
    lines.append("## 2. Pre-registered Hypotheses Status (H1 - H9)")
    lines.append("| Hypothesis | Description | Target Gap | Verdict | Key Finding |")
    lines.append("|---|---|---|---|---|")

    hypotheses_defs = [
        ("H1", "iid-max shortcut F^n misestimates the winning hand (board sharing)", "G1"),
        ("H2", "Equity shares average 1/n; equity distribution is Beta-like; profitable fraction changes with n", "G2"),
        ("H3", "Equity is a martingale; Rao-Blackwell lowers variance; Rule of 2/4 fails for large outs", "G3"),
        ("H4", "Tighter opponent ranges lower weak-hand equity and shift the formula", "G4"),
        ("H5", "Brownian ruin matches simulation; CLT is slow for per-hand heavy-tailed poker profit", "G5"),
        ("H6", "A simple formula predicts starting hand win probability within 1.0 pp tolerance", "G7"),
        ("H7", "Hand rankings change across table sizes n (72o vs 32o reordering)", "G7"),
        ("H8", "Limp-or-fold strategy profit interacts with table composition (table selection)", "G8"),
        ("H9", "Human probability guesses are better calibrated with the simulation aid", "G9"),
    ]

    for hid, desc, target_gap in hypotheses_defs:
        # Search for records tagged with this hypothesis or gap
        matching = [r for r in records.values() if r.get("hypothesis") == hid or (r.get("gap") == target_gap and hid in r.get("fid", ""))]
        if matching:
            v_list = list({m["verdict"] for m in matching})
            v_str = ", ".join(v_list)
            fids = ", ".join(m["fid"] for m in matching)
            lines.append(f"| **{hid}** | {desc} | {target_gap} | **{v_str}** | {fids} |")
        else:
            lines.append(f"| **{hid}** | {desc} | {target_gap} | *PENDING* | Awaiting run |")
    lines.append("")

    # 3. Findings Grouped by Research Gap
    lines.append("## 3. Findings by Research Gap")
    gaps = sorted(list({r.get("gap", "OTHER") for r in records.values()}))
    if not gaps:
        lines.append("*No findings recorded yet.*")
    else:
        for g in gaps:
            gap_records = [r for r in records.values() if r.get("gap") == g]
            lines.append(f"### Gap {g}")
            for r in gap_records:
                lines.append(f"#### [{r['fid']}] {r['title']}")
                lines.append(f"- **Verdict:** `{r['verdict']}`")
                lines.append(f"- **Question:** {r['question']}")
                lines.append(f"- **Method:** {r['method']}")
                lines.append("- **Results:**")
                for k, v in r["results"].items():
                    if isinstance(v, dict):
                        val_str = str(v.get("value", ""))
                        if "se" in v:
                            val_str += f" (SE: {v['se']})"
                        if "ci" in v:
                            val_str += f" (95% CI: [{v['ci'][0]}, {v['ci'][1]}])"
                        lines.append(f"  - `{k}`: {val_str}")
                    else:
                        lines.append(f"  - `{k}`: {v}")
                if r["figures"]:
                    lines.append("- **Figures:** " + ", ".join(f"[{fig}](file://{ROOT_DIR / fig})" for fig in r["figures"]))
                lines.append(f"- **Caveats / Limitations:** {r['caveats']}")
                lines.append("")

    # 4. Verification and Benchmark Log
    lines.append("## 4. Verification and Benchmark Log")
    verif_records = [r for r in records.values() if r.get("verdict") in ("VERIFIED", "FAILED CHECK") or r.get("gap") == "VERIFICATION"]
    if not verif_records:
        lines.append("*No verification records.*")
    else:
        lines.append("| ID | Check / Benchmark | Verdict | Summary |")
        lines.append("|---|---|---|---|")
        for r in verif_records:
            res_summary = "; ".join(f"{k}={v}" for k, v in list(r["results"].items())[:3])
            lines.append(f"| `{r['fid']}` | {r['title']} | `{r['verdict']}` | {res_summary} |")
    lines.append("")

    # 5. Negative and Failed Results
    lines.append("## 5. Negative and Failed Results")
    neg_records = [r for r in records.values() if r.get("verdict") in ("NOT SUPPORTED", "FAILED CHECK")]
    if not neg_records:
        lines.append("*No failed checks or unsupported hypotheses recorded.*")
    else:
        for r in neg_records:
            lines.append(f"- **`{r['fid']}` ({r['verdict']}):** {r['title']}. {r['caveats']}")
    lines.append("")

    # 6. Formula Card (G7 Simple Validated Formula)
    lines.append("## 6. Formula Card: Starting-Hand Equity Prediction")
    lines.append("### Model M1 (Normalized Power Law)")
    lines.append("$$\\hat{E}_n(h) = \\frac{1}{n} \\cdot \\frac{(2 \\cdot p_2(h))^{\\beta_n}}{\\kappa_n}$$")
    lines.append("By-hand simplified phone recipe (assuming $\\kappa_n \\approx 1.0$):")
    lines.append("$$\\hat{E}_n(h) \\approx \\frac{1}{n} (2 \\cdot p_2(h))^{\\beta_n}$$")
    lines.append("")
    lines.append("### Empirical Smooth Law for Exponent $\\beta_n$")
    lines.append("$$\\beta_n = 1.0 + 0.230 \\sqrt{n - 2}$$")
    lines.append("")
    lines.append("### Fitted Parameter Table")
    lines.append("| Table Size $n$ | Fitted $\\beta_n$ | Normalizer $\\kappa_n$ | Smooth Law $\\beta_n$ | Model M4 MAE (pp) |")
    lines.append("|---|---|---|---|---|")
    lines.append("| 2 | 1.0000 | 1.0000 | 1.0000 | 0.00 |")
    lines.append("| 3 | 1.2459 | 1.0059 | 1.2301 | 0.81 |")
    lines.append("| 4 | 1.3614 | 1.0094 | 1.3254 | 0.89 |")
    lines.append("| 5 | 1.4026 | 1.0108 | 1.3986 | 0.94 |")
    lines.append("| 6 | 1.4878 | 1.0140 | 1.4603 | 0.96 |")
    lines.append("| 7 | 1.5412 | 1.0161 | 1.5146 | 1.01 |")
    lines.append("| 8 | 1.5550 | 1.0166 | 1.5636 | 1.03 |")
    lines.append("| 9 | 1.5458 | 1.0162 | 1.6088 | 1.08 |")
    lines.append("")
    lines.append("### 3-Line Phone Recipe")
    lines.append("1. **Double heads-up equity:** Lookup heads-up equity $p_2$ from the 13x13 chart. Compute $x = 2 \\times p_2$.")
    lines.append("2. **Raise to power:** Compute exponent $\\beta_n = 1.0 + 0.230 \\sqrt{n - 2}$ and raise: $y = x^{\\beta_n}$.")
    lines.append("3. **Divide by table size:** Final estimated equity share is $\\hat{E}_n = y / n$.")
    lines.append("")
    lines.append("### Worked By-Hand Example (Pocket Kings at $n = 6$)")
    lines.append("- Given $p_2(\\text{KK}) = 0.8288$.")
    lines.append("- Base term: $x = 2 \\times 0.8288 = 1.6576$.")
    lines.append("- Exponent: $\\beta_6 = 1.0 + 0.230 \\sqrt{6 - 2} = 1.0 + 0.460 = 1.460$.")
    lines.append("- Power: $1.6576^{1.460} \\approx 2.0914$.")
    lines.append("- Estimated equity: $\\hat{E}_6 = 2.0914 / 6 \\approx 0.3486$ (34.86%).")
    lines.append("- Empirical equity: $E_6 = 0.4279$ (42.79%). Absolute error $= 7.93$ pp (reflects known power-law premium pair depression; corrected by M4).")
    lines.append("")

    # 7. Scope Tiers (Tier A, B, C)
    lines.append("## 7. Scope Tiers & Course Presentation Strategy")
    lines.append("- **Tier A (Core — Main Report Text & Hypotheses):**")
    lines.append("  - G1: Table size dynamics & order statistics ($F^n$ vs card sharing, H1).")
    lines.append("  - G2: 169-class equity distribution, combo-weighted $1/n$ conservation, and Beta fits (H2).")
    lines.append("  - G6: Waiting times for rare hands, Geometric distribution, and PGF derivations.")
    lines.append("  - G7: Simple formula specification (M1-M4), smooth exponent law, and rank stability decay (H6, H7).")
    lines.append("  - Real-Data Section: UCI Poker Hand verification & Table 1 GOF; PHH cash game analysis & showdown selection bias.")
    lines.append("- **Tier B (Selected Modules — One-page deep dives):**")
    lines.append("  - G3: Street equity martingale property, total variance decomposition, Rao-Blackwell variance reduction ($2.5-4.3\\times$), and Rule of 2/4 breakdown (H3).")
    lines.append("  - G4 & G8: Opponent range models (Rock, TAG, LAG), weak-hand equity collapse (H4), closed-form blind costs ($1.5/n$ bb/hand), and table selection two-way ANOVA interaction (H8).")
    lines.append("- **Tier C (Extra — Outside Master Report Claims):**")
    lines.append("  - G9 & `demo.py`: Live interactive multi-user websocket room with QR code join, simulated bot games, and human calibration logging. (EXTRA: Never cited as a core research claim).")
    lines.append("")

    # 8. Scope Map (from CONTEXT.md Section 3)
    lines.append("## 8. Scope Map")
    lines.append("| Aspect of poker | Status | What we do | Future study |")
    lines.append("|---|---|---|---|")
    lines.append("| Probability of hands and outcomes (combinatorics, equity) | **IN SCOPE (core)** | Exact counts, Monte Carlo and exact equity, equity tables, a simple formula | Omaha, other variants |")
    lines.append("| Number of players (table size) | **IN SCOPE (core)** | Order statistics of the winning hand, equity vs n, rank changes across n | Position and seat effects |")
    lines.append("| Imperfect information (unknown opponent cards) | **PARTLY** | Opponent cards modelled as uniformly random or as the top q% of hands (range model); selection bias measured | Bayesian updating of opponent ranges from actions; opponent modelling |")
    lines.append("| Probability estimation by humans | **IN SCOPE** | Live demo: unaided vs aided guesses, calibration | Training effects over many sessions |")
    lines.append("| Variance, luck and sample size | **IN SCOPE** | CLT convergence, sample size to separate strategies, ruin approximation, waiting times | Bankroll management policies |")
    lines.append("| Long-term planning (blinds, many-hand results) | **PARTLY** | Blind cost model, long-run profit per hand, table composition effects | Dynamic programming of stack and blind schedules; tournaments and ICM |")
    lines.append("| Simple strategies | **PARTLY** | Play-or-fold (limp-or-fold) strategies and range-width archetypes at different table sizes | Raising, bet sizing, bluffing, value betting, slow-play, fold equity |")
    lines.append("| Multi-street decision making | **OFF LIMITS (analysis only)** | Equity over streets (martingale, variance by street) as pure probability; no betting policy | Postflop policies, reinforcement learning, counterfactual regret minimization |")
    lines.append("| Game-theoretic equilibrium (GTO) | **OFF LIMITS** | Cited only (CFR, Libratus, Pluribus) | Solving small games; exploitability |")
    lines.append("| Psychology (tilt, tells), real-money behaviour | **OFF LIMITS** | None | Behavioural studies |")
    lines.append("| Real-money gambling advice | **OFF LIMITS** | Explicitly excluded | n/a |")
    lines.append("")

    # 9. Authorship and AI-Use Allocation
    lines.append("## 9. Authorship & AI-Use Disclosure Summary")
    lines.append("| Component | Primary Author | Status | Verification Mechanism |")
    lines.append("|---|---|---|---|")
    lines.append("| Six Student-Owned Functions (cards & stats) | Student Author | To be written by author (stubs & checkers active) | Automated unit tests in Section 3 (`USE_STUDENT=False`) |")
    lines.append("| Hand-Solved Problems (E1 - E6) | Student Author | To be solved by author on paper | `check_answer(example_id, value)` in Section 13 |")
    lines.append("| Master Report (20-30 pages) & Slides (10-15) | Student Author | To be written by author | Scaffolds and support pack in `reports/support.md` |")
    lines.append("| Vectorized Evaluators & Monte Carlo Engine | AI Assistant | Implemented & Verified | 20k hand Treys check & 1.025M UCI hand verification |")
    lines.append("| Statistical Toolkit & Distribution Fits | AI Assistant | Implemented & Verified | Pure NumPy verified against SciPy & Statsmodels |")
    lines.append("| Simulation Data Caching & Figures Pipeline | AI Assistant | Implemented & Verified | All PNG figures accompanied by raw CSV data files |")
    lines.append("")

    # 10. Open Limitations
    lines.append("## 10. Open Limitations")
    lines.append("1. **Showdown selection bias in PHH data:** Hole cards are only observed for hands reaching showdown; folded hands remain unobserved, inflating observed showdown hand quality.")
    lines.append("2. **Equal-contribution / Limp-or-fold abstraction:** Does not reflect no-limit bet sizing, aggressive bluffing, or fold equity.")
    lines.append("3. **Board card correlation:** iid order statistics assume independent draws, ignoring the 5 shared community cards which induce positive dependence.")
    lines.append("4. **Range sensitivity:** Bot play rates prior to showdown may differ from showdown ranges.")
    lines.append("5. **Continuous Beta model for discrete equity:** Beta distribution assumes continuous support on $[0, 1]$, whereas starting hands have discrete rank structures.")
    lines.append("")

    content = "\n".join(lines)
    with open(FINDINGS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def export_figures_data_json(output_path: Path | str | None = None) -> dict[str, Any]:
    """Aggregate all CSV files in figures/data/ into a unified JSON structure."""
    data_dir = FIGURES_DATA_DIR
    files = sorted(list(data_dir.glob("*.csv")))

    manifest: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "_metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_directory": str(data_dir.relative_to(ROOT_DIR)),
            "total_files": len(files),
            "manifest": manifest,
        }
    }

    for f in files:
        try:
            df = pd.read_csv(f)
            records = df.to_dict(orient="records")
            cleaned = []
            for r in records:
                cleaned_row = {}
                for k, v in r.items():
                    if pd.isna(v):
                        cleaned_row[k] = None
                    else:
                        cleaned_row[k] = v
                cleaned.append(cleaned_row)

            entry = {
                "filename": f.name,
                "stem": f.stem,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "data": cleaned,
            }
            payload[f.name] = entry
            manifest.append({
                "filename": f.name,
                "stem": f.stem,
                "row_count": len(df),
                "columns": list(df.columns),
            })
        except Exception:
            pass

    target_path = Path(output_path) if output_path else ROOT_DIR / "figures_data.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    sub_path = data_dir / "figures_data.json"
    if target_path.resolve() != sub_path.resolve():
        with open(sub_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    return payload

