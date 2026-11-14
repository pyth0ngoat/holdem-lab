# Hold'em Lab

Applied Probability Distributions (THM6242) Research Project on Texas Hold'em:
*How does the number of players at the table change which hands win, can a simple formula estimate a hand's chance of winning at an n-player table, and what do real hand histories say about it?*

---

## Authorship and AI-Use Disclosure Statement

> **Notice to Examiners:** This declaration complies with the course policy on academic integrity and artificial intelligence assistance. Blanks are reserved strictly for the student author.

```
Student Name:                   [ ______________________________ ]
Student ID:                     [ ______________________________ ]
Course / Module:                Applied Probability Distributions (THM6242)
Submission Date:                [ ______________________________ ]
Target Viva / Presentation:     [ ______________________________ ]

Student-Owned Components (Authored entirely by the student without AI code generation):
1. Five-card hand category classifier (inline NumPy implementation)
2. Beta distribution parameter estimation by Method of Moments and Maximum Likelihood (inline NumPy)
3. Chi-square goodness-of-fit test statistic and p-value calculation (inline NumPy)
4. Order-statistic maximum CDF for discrete random variables with ties (inline NumPy)
5. Geometric distribution waiting time moments via Probability Generating Functions (inline NumPy)
6. Weighted Least Squares regression for formula scaling exponent beta_n (inline NumPy)
7. All hand-solved mathematical problems in reports/hand_examples_sheet.md
8. Master written project report (20-30 pages) and presentation slide deck (10-15 slides)

AI Assistant Usage Statement:
The AI assistant (Antigravity / Gemini) was utilized as a lead software engineering tool to:
- Construct the scaffolding repository, automated build pipeline, and test harness
- Implement Monte Carlo simulation engines and benchmark validations against standard libraries
- Format and record empirical simulation findings to reports/findings.md and figures/data/
- Scaffold notebook templates and automated assertion checks

I confirm that all mathematical interpretations, written analytical prose in the report,
and the six student-owned functions identified above represent my own original intellectual work.

Student Signature:              [ ______________________________ ]
Date:                           [ ______________________________ ]
```

---

## Setup Guide

### System Requirements
- Python 3.11+
- Compatible with macOS (Apple Silicon / Intel), Linux (x86_64 / arm64), and Windows (WSL2 recommended or native PowerShell).

### Installation

1. **Clone and enter repository:**
   ```bash
   git clone <repo-url> holdem-lab
   cd holdem-lab
   ```

2. **Create virtual environment and install dependencies:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     venv/bin/pip install -r requirements-dev.txt
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\pip install -r requirements-dev.txt
     ```

3. **Verify installation:**
   ```bash
   venv/bin/pytest -v
   ```

---

## Data Download and Access Instructions

### 1. UCI Poker Hand Dataset (Required Open Dataset)
- **Source:** UCI Machine Learning Repository (Dataset ID: 158; DOI: `10.24432/C5KW38`).
- **Citation:** Cattral, R. and Oppacher, F. (2002), Poker Hand, UCI Machine Learning Repository.
- **Files required:**
  - `data/external/uci/poker-hand-training-true.data` (~0.6 MB)
  - `data/external/uci/poker-hand-testing.data` (~23.4 MB)
  - `data/external/uci/poker-hand.names`
- **Acquisition methods:**
  - Direct download from [UCI Repository](https://archive.ics.uci.edu/dataset/158/poker+hand) (unzip into `data/external/uci/`).
  - Or via `ucimlrepo`:
    ```bash
    venv/bin/pip install ucimlrepo
    # In python:
    from ucimlrepo import fetch_ucirepo
    poker = fetch_ucirepo(id=158)
    ```

### 2. PHH Hand Histories (Core Real-Play Dataset)
- **Source:** "A Dataset of Poker Hand Histories", Juho Kim, University of Toronto (Zenodo DOI: `10.5281/zenodo.10796885`).
- **Format:** [phh.readthedocs.io](https://phh.readthedocs.io); GitHub repository: [github.com/uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset).
- **Access Policy:**
  - Set decision flag in `data/external/phh/DECISION.txt`:
    - `have-files`: Use pre-extracted HandHQ cash game hands in `data/external/phh/phh-dataset/data/handhq/` (441 files, ~311 MB).
    - `download-full`: Full 20.3 GB zip from Zenodo (requires confirmation of bandwidth and disk space).
    - `fallback-small`: Condensed sets (Pluribus 10,000 hands, WSOP 83 hands).
    - `skip`: Skip real-play analysis.

---

## Hand-Solved Problems and Verification

The student problem sheet is located at `reports/hand_examples_sheet.md`.
It contains six one-page problems built from real project numbers, designed to be solved by hand on paper:
- **E1:** Discrete order statistic of the maximum (tie-aware) using exact 5-card category probabilities.
- **E2:** Exact combinatorial draw probability by counting (flush draw) and the $1/n$ exchangeability identity on a miniature deck.
- **E3:** Beta distribution shape parameters by Method of Moments from the 169-class equity mean and variance at $n = 6$.
- **E4:** Pearson chi-square goodness-of-fit test statistic and degrees of freedom for a 4-category sub-table from UCI data.
- **E5:** Geometric waiting time mean and variance derived via the Probability Generating Function (PGF) for Pocket Aces ($p = 1/221$).
- **E6:** Closed-form formula equity prediction (Model M1) for Pocket Kings (KK) at $n = 6$ vs empirical simulation table.

### How to Use the Sheet and Checker
1. Open `reports/hand_examples_sheet.md` and solve each problem by hand on paper.
2. Verify your computed values without revealing solutions using the automated checker in `holdem_lab.ipynb` Section "Hand Examples Checker" or via Python:
   ```python
   from holdem_lab.analysis import check_answer

   # Check your answer for Problem E1
   check_answer("E1", your_computed_number)
   ```
3. The checker reports `PASS` or `FAIL`. It reveals the reference final value only upon a `PASS` or after a second failed attempt.
4. **Academic Integrity:** The checker never prints worked steps or derivations. Handwritten solutions belong in your personal coursework folder and report appendix.

---

## Running the Project

### Running the Notebook

#### 1. Full Development Mode (with `holdem_lab` package)
Execute end-to-end in quick mode (~25s):
```bash
HOLDEM_MODE=quick venv/bin/jupyter nbconvert --to notebook --execute holdem_lab.ipynb --output holdem_lab_executed.ipynb
```
Or open interactively:
```bash
venv/bin/jupyter lab holdem_lab.ipynb
```

#### 2. Portable Submission Mode (Running with only the Submission Files)
A grader or examiner who receives only the submission directory can run the notebook without installing the development package:
```bash
# 1. Create and activate a clean virtual environment
python3 -m venv submission_env
source submission_env/bin/activate  # On Windows: submission_env\Scripts\activate

# 2. Install pinned dependencies (no demo server packages required)
pip install -r submission/requirements.txt

# 3. Open and run all cells in Jupyter
jupyter lab holdem_lab.ipynb
```
- **Expected Run Time:** ~3-5 seconds in portable mode (loads verified cached tables and runs pure-NumPy verifications inline); ~25 seconds in quick simulation mode.
- **Required Files Staged Alongside Notebook:**
  - `data/external/uci/`: `poker-hand-training-true.data` and `poker-hand-testing.data`
  - `data/formula_params.json`: Fitted power-law formula parameters
  - `figures/data/*.csv`: Pre-computed simulation outputs, GOF tables, and benchmark data
  - `reports/hand_examples_sheet.md`: Six hand-solved problem sheets
  - `submission/requirements.txt`: Minimal pinned submission dependencies
- **Zero-Traceback Guarantee:** Notebook automatically detects package absence (`USE_PACKAGE = False`) and executes seamlessly start-to-end.

### Running the Live Interactive Demo (`demo.py`, EXTRA)
Host a live 9-seat room with real-time probability displays and QR code join:
```bash
venv/bin/python demo.py --bots 6 --port 8000
```
- Open `http://localhost:8000` on projector / host machine.
- Mobile players scan the on-screen QR code or connect to `http://<LAN-IP>:8000/join?room=<code>`.

### Troubleshooting Demo & Networking
- **Port in use:** Specify a different port with `--port 8080`.
- **Firewall:** Ensure incoming TCP traffic on the selected port is allowed in your OS firewall settings.
- **Client isolation:** Institutional or campus Wi-Fi frequently isolates wireless clients. Use a personal phone or laptop hotspot to connect host and player phones on the same subnet.
