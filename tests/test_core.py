"""Hold'em Lab - Core Infrastructure and Findings Tests.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from holdem_lab import (
    CACHE_DIR,
    CONFIG,
    FIGURES_DATA_DIR,
    FIGURES_DIR,
    MODE,
    REPORTS_DIR,
    ROOT_DIR,
)
from holdem_lab.findings import (
    assert_all_claims_backed,
    clear_claims_ledger,
    load_findings_records,
    record_finding,
    register_claim,
    save_fig,
    write_findings,
)


def test_repo_layout_directories():
    """Verify that expected core directories exist."""
    assert ROOT_DIR.exists()
    assert CACHE_DIR.exists()
    assert FIGURES_DIR.exists()
    assert FIGURES_DATA_DIR.exists()
    assert REPORTS_DIR.exists()


def test_config_structure():
    """Verify configuration parameters."""
    assert MODE in ("quick", "full")
    assert "seeds" in CONFIG
    assert CONFIG["seeds"]["fit"] == 42
    assert CONFIG["seeds"]["validation"] == 1337
    assert CONFIG["seeds"]["demo"] == 2026
    assert CONFIG["formula_tolerance"] == 0.01


def test_findings_record_and_save_fig(tmp_path):
    """Verify recording a finding, saving a figure, and CSV data generation."""
    test_fid = "T-CORE-1"
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    ax.plot(x, y, label="y=2x+1")
    ax.set_title("Test Figure")

    df = pd.DataFrame({"x": x, "y": y})
    fig_path = save_fig(fig, test_fid, "test_plot", "Test plot caption", data=df)

    assert Path(fig_path).exists()
    assert (FIGURES_DIR / f"{test_fid}_test_plot.png").exists()
    assert (FIGURES_DATA_DIR / f"{test_fid}_test_plot.csv").exists()

    rec = record_finding(
        fid=test_fid,
        title="Test Core Finding",
        question="Does findings helper record correctly?",
        method="Unit test verification",
        results={"slope": 2.0, "intercept": {"value": 1.0, "se": 0.01}},
        figures=[fig_path],
        verdict="VERIFIED",
        caveats="Synthetic unit test only",
        gap="VERIFICATION",
    )

    records = load_findings_records()
    assert test_fid in records
    assert records[test_fid]["verdict"] == "VERIFIED"
    assert records[test_fid]["results"]["slope"] == 2.0


def test_write_findings_idempotency():
    """Verify that write_findings generates reports/findings.md and is idempotent."""
    md1 = write_findings()
    findings_md_file = REPORTS_DIR / "findings.md"
    assert findings_md_file.exists()

    md2 = write_findings()
    assert md1 == md2


def test_claims_ledger_verification():
    """Verify that assert_all_claims_backed succeeds for valid claims and fails for missing ones."""
    clear_claims_ledger()

    # Backed claim
    record_finding(
        fid="T-BACKED-1",
        title="Backed Claim Finding",
        question="Is it backed?",
        method="Direct verification",
        results={"success": True},
        figures=[],
        verdict="VERIFIED",
        caveats="None",
    )
    register_claim("System has core verification", "T-BACKED-1")
    assert_all_claims_backed()

    # Missing claim
    register_claim("Non-existent claim", "T-MISSING-999")
    with pytest.raises(AssertionError) as exc_info:
        assert_all_claims_backed()
    assert "T-MISSING-999" in str(exc_info.value)

    clear_claims_ledger()
