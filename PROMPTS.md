# PROMPTS.md — Hold'em Lab build prompts (v3)

Paste ONE prompt at a time, in order. After each: run `venv/bin/pytest`, open `reports/findings.md` and the new figures, check `git log --oneline -3`, then continue.
**Run order:** P0, P1, P2, P3, P9 (data), P4, P5, P8, P6, P7, P12, P11, P11b, then P10 (demo, an EXTRA) only if time allows.
**Fast path if time is short:** P0, P1, P2, P3, P9, P4, P5, P8, P12, P11. P6 and P7 (Tier B) and P10 (extra) are optional.
**Authorship:** six core functions, the hand-solved examples, the report and the slides are yours to write (CONTEXT.md Section 2A). The prompts give you stubs, checkers and numbers, not solutions.

## HANDS-OFF MODE: put the files in the folder, paste AUTO RUN once

Before pasting AUTO RUN, do this (all optional ones have safe defaults):
1. Folder contains `CONTEXT.md`, `AGENTS.md` (from the block below) and `PROMPTS.md`.
2. **Data:** put the UCI files in `data/external/uci/` (`poker-hand-training-true.data`, `poker-hand-testing.data`, `poker-hand.names`). For the PHH real-play data create `data/external/phh/DECISION.txt` containing exactly one word: `download-full` (you accept the 20.3 GB download), `partial` (try extracting only the HandHQ files), `fallback-small` (small Pluribus/WSOP sets only; this is also the default if the file is missing) or `skip`. If you already have extracted PHH files, put them in `data/external/phh/` and write `have-files`.
3. **Antigravity settings:** terminal command auto-execution set so the agent is not interrupted (allow list or Always Proceed with the deny list rm, sudo, git push, git reset, curl, wget; file access limited to this folder). Start a new agent conversation after changing settings.
4. Leave the disclosure blanks (README and notebook) empty; the agent never fills them.

**AUTO RUN prompt (paste once):**
```
AUTO RUN. Follow CONTEXT.md and AGENTS.md. For this run only, the rule "do ONLY the pasted prompt" is replaced by: execute every prompt in PROMPTS.md in the run order given at the top of PROMPTS.md (P0, P1, P2, P3, P9, P4, P5, P8, P6, P7, P12, P11, P11b, P10), one after another, without waiting for me, stopping only under the STOP conditions below.

Preflight (do not stop unless it fails): read CONTEXT.md, AGENTS.md and PROMPTS.md fully; check Python 3.11+ and report free disk space; check that data/external/uci contains poker-hand-training-true.data and poker-hand-testing.data (if not, try the documented download once; if that fails, STOP and tell me exactly which file to place where); read data/external/phh/DECISION.txt if present (download-full, partial, fallback-small, have-files, skip); if absent use fallback-small and log it.

Loop, for each prompt in the run order:
1. Re-read CONTEXT.md and the prompt text; skip any prompt already ticked in CONTEXT.md Section 19 (this makes the run resumable).
2. Do exactly what the prompt says and obey every rule in AGENTS.md, including: never implement the six student-owned functions, the hand-solved examples, the report or the slides; only stubs, checkers and problem sheets.
3. Run pytest and any command the prompt requires. If something fails, find the root cause and fix it; at most 3 attempts per distinct failure, then STOP and report.
4. Update CONTEXT.md Sections 18 and 19 and commit as the prompt says. Never git push.
5. Print a 5-line status (prompt, tests, findings recorded, anything skipped, next prompt) and continue immediately.
Rules: MODE=quick for every notebook run, never full mode. Do not download any single file over 1 GB unless DECISION.txt says download-full or partial. Never delete my data. No network access except URLs documented in CONTEXT.md Section 9 and package installs.

STOP only for: (a) UCI data missing and not downloadable; (b) a failure still unresolved after 3 attempts; (c) something that needs my decision or credentials; (d) a conflict between a prompt and CONTEXT.md you cannot resolve conservatively; (e) free disk space under 5 GB before a download; (f) any action that would break an AGENTS.md rule.
Do NOT stop for: missing PHH data (use the fallback and log it), optional steps that cannot be completed (record an INFORMATIONAL finding and continue), a tool permission popup (retry once), or style choices (choose the simplest and log it).

At the end run the whole test suite and the notebook in quick mode, then print a HANDOFF report: what is done; what failed or was skipped and why; the tier summary; and the exact list of things only I can do (fill the disclosure blanks; write the six student-owned functions and then set USE_STUDENT = True and re-run P11b; solve the hand-solved examples; write the report and slides; confirm dates and the AI policy; open the PHH GitHub folder if the fallback was used; test the QR room on my network), with the exact commands to open the notebook and start the demo.
```

**Resume prompt (if the session stops or the context runs out):**
```
Resume AUTO RUN. Read CONTEXT.md Section 19, find the first unticked prompt in the run order, check git status for unfinished work, finish it, and continue with the same rules and STOP conditions as the AUTO RUN prompt.
```

**After the student-owned functions are written:**
```
Re-run P11b: I have implemented the six student-owned functions in the notebook. Set USE_STUDENT = True only if every checker passes, run the notebook in quick mode in portable mode, and report any failure. Do not edit my functions.
```

## STEP 0 (you, not the agent): new folder and repo

```
mkdir holdem-lab
cd holdem-lab
git init
```
Put `CONTEXT.md` and `AGENTS.md` (below) in the folder, open it in Antigravity, and set the terminal allow list as before (make, venv/bin/python, venv/bin/pip, venv/bin/pytest, venv/bin/ruff, venv/bin/black, ls, cat, git status, git diff, git add, git commit). For the demo, plan to use a phone or laptop hotspot.

## AGENTS.md (create this file with exactly this content)

```
You are the lead engineer on the project in CONTEXT.md (Hold'em Lab). Rules:
- At the start of every task read CONTEXT.md fully. It is the single source of truth.
- Do ONLY the prompt I paste. Never start the next prompt. Exception: when I paste the AUTO RUN prompt, execute every prompt in PROMPTS.md in the run order given there, obeying that prompt's STOP conditions.
- Keep the file count low: only the files in CONTEXT.md Section 13. Do not add modules or scripts without logging a reason in Section 18.
- holdem_lab/cards.py and holdem_lab/stats.py must be numpy/stdlib only. scipy, statsmodels and poker libraries (phevaluator, treys) only in tests and notebook verification cells.
- Never fabricate data, numbers, citations or results. Verify downloads and URLs; if you cannot, say so and log it.
- Every Monte Carlo result carries a standard error or confidence interval and a fixed seed. Fitting seeds and validation seeds are separate.
- EVERYTHING tested or computed must be recorded in reports/findings.md through holdem_lab/findings.py (record_finding, save_fig, write_findings), including failures and negative results. Every figure goes to figures/ with its data in figures/data/. Nothing is claimed unless it has a finding record.
- Every analysis prompt must also add its section to holdem_lab.ipynb (cells that call package functions, then verification cells) and run that section in quick mode.
- Educational framing: play chips only, no gambling advice.
- Authorship: the six student-owned functions, the hand-solved examples, the report and the slides are written by the human (CONTEXT.md Section 2A). For those, write only stubs, checkers, problem statements with real numbers and, when asked, conceptual explanations or hints; never write the solution into the repository. Keep the README disclosure statement accurate to what you actually wrote.
- Data: never fabricate or substitute data. Large downloads (over 1 GB) need the human's confirmation of disk space and bandwidth first; state sizes before downloading. Verify every URL and file listing; if something is unavailable, say so and use the documented fallback.
- Do not write finished report or slide prose for submission; produce code, tables, figures, scaffolds and notes. The human writes in their own words.
- Terminal rules: run commands from the workspace root, one command at a time, no && chains, no cd, no `source venv/bin/activate`; call venv/bin/python, venv/bin/pip and venv/bin/pytest directly.
- Write tests as you go. A prompt is done only when pytest passes.
- Do not ask questions unless you need something only the human can provide.
- At the end of every prompt: update CONTEXT.md Section 19 (tick the box, add a dated 3-6 line summary with file paths and caveats), append decisions to Section 18, run git add -A, git commit -m "<what this prompt built>", then STOP and report: what changed, exact verification commands, what you could not verify.
```

---

## PROMPT 0 — Fresh repo, helpers, notebook skeleton

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P0). This is a brand-new repository.
1. Create the layout from CONTEXT.md Section 13 (empty modules with docstrings and TODOs, tests folder, figures/ and figures/data/, reports/, data/external/uci, data/external/phh, data/cache, data/demo_logs, .gitignore for venv, data/cache, data/external, __pycache__, .env). Create the virtual environment with python3 -m venv venv, then use venv/bin/pip.
2. requirements.txt: numpy, pandas, matplotlib, plotly, pyyaml, tqdm, rich, jupyter, nbformat, nbconvert, ipykernel, fastapi, uvicorn[standard], websockets, qrcode[pil], pytest, ruff, black. requirements-dev.txt: scipy, statsmodels, phevaluator, treys (test and notebook verification only). Minimal Makefile: setup, test, lint, notebook (executes holdem_lab.ipynb with nbconvert), demo. README.md with a start-to-end setup guide for macOS, Windows and Linux and troubleshooting (ports, firewall, hotspot).
3. holdem_lab/findings.py: record_finding(fid, title, question, method, results, figures, verdict, caveats) where results is a dict of named values (values may carry se or ci), verdict in {SUPPORTED, NOT SUPPORTED, INCONCLUSIVE, VERIFIED, FAILED CHECK, INFORMATIONAL}; save_fig(fig, fid, slug, caption, data=None) saving figures/<fid>_<slug>.png (dpi 200, tight layout, readable fonts, colourblind-safe palette) and, if a dataframe is given, figures/data/<fid>_<slug>.csv; write_findings() rebuilding reports/findings.md idempotently with: run metadata (date, mode, seeds, git commit, package versions), a hypotheses summary table (H1-H9 from CONTEXT.md, status from records), findings grouped by gap, a verification log, a negative and failed results section, the scope map from CONTEXT.md Section 3, and open limitations. Records are stored in reports/findings.json (machine-readable) so write_findings can rebuild from all prompts' records, not just the current run. Provide a claims ledger helper: register_claim(claim, fid) and assert_all_claims_backed().
4. config in holdem_lab/__init__.py (or a small config dict): MODE from environment variable HOLDEM_MODE (quick or full, default quick), seeds (fit, validation, demo), n range 2-9, Monte Carlo sizes per mode, formula tolerance 1.0 percentage points, paths.
5. holdem_lab.ipynb skeleton with the section headings from CONTEXT.md Section 11, a first markdown cell holding the authorship and AI-use disclosure statement from Section 2A (with blanks for the human; never fill the blanks yourself), a setup cell that sets MODE and imports the package, and a final cell that calls assert_all_claims_backed() and write_findings(). README.md must also contain the same disclosure statement and the data-download instructions from Section 9 (UCI via page download or ucimlrepo; PHH access plan).
6. tests/test_core.py: findings helper writes a record, a figure and a CSV; write_findings is idempotent; assert_all_claims_backed fails when a claim has no record.
Finish per AGENTS.md (commit: chore: fresh repo and helpers).
```

## PROMPT 1 — Cards, evaluator, exact counts, benchmark tests

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P1). Code goes in holdem_lab/cards.py.
1. Card encoding, deck, seeded shuffling with numpy Generator, and a vectorized evaluator in numpy only: arrays of 5-card and 7-card hands -> integer strength score (higher is better, ties equal) and category 0-9 (high card, pair, two pair, three of a kind, straight, flush, full house, four of a kind, straight flush, royal flush). Handle the wheel. Document the algorithm and scoring scheme. Report measured speed (hands per second) and note whether it is adequate for the Monte Carlo sizes in CONTEXT.md (target at least 1 million 7-card hands per second if achievable).
2. Exact counts: 5-card hands per category (sum 2,598,960) and 7-card hands per category (sum 133,784,560) by exact enumeration or exact combinatorial counting (verify any combinatorial method by sampling). Probabilities as fractions and decimals.
3. Benchmark and verification tests (tests/test_core.py; libraries allowed here): agreement with phevaluator or treys on at least 2 million random 7-card hands (ordering and category, including ties); exact counts equal textbook values; set on the flop with a pocket pair = 1 - C(48,3)/C(50,3) (about 1 in 8.51); flush by the river from four suited cards on the flop = 1 - C(38,2)/C(47,2) = 34.97% (rule of 4 gives 36%); seven-card royal flush = 4,324 of 133,784,560 (1 in 30,940).
4. Notebook section 4 (evaluator and exact counts): table of category counts and probabilities, bar chart on a log scale, verification cells; save with save_fig; record findings: evaluator agreement (VERIFIED or FAILED CHECK), exact counts (VERIFIED), the three benchmark probabilities (VERIFIED), speed (INFORMATIONAL).
5. Add a DRAFT worked example to the notebook's theory section (counting full houses with binomial coefficients) for the human to rewrite.
Finish per AGENTS.md (commit: feat: evaluator and exact counts).
```

## PROMPT 2 — Equity engine and hand classes

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P2). Code in holdem_lab/equity.py.
1. Definitions from CONTEXT.md Section 4: P_win, P_tie, equity share (ties split 1/k).
2. Monte Carlo: equity of a specified hand (two hole cards, optional partial board) against n-1 random opponent hands, vectorized, seeded, returning P_win, P_tie, equity share with standard errors and a stopping rule on precision. An all-players engine that deals full tables of n players and returns every player's outcome. Opponents may optionally be drawn from a range (top q% under a given ranking) for later use in G4.
3. Exact enumeration where feasible (turn and river enumeration, heads-up preflop for a specified hand vs specified hand); verify Monte Carlo against exact within confidence intervals.
4. Hand classes: map any two cards to one of 169 classes; combo weights 6, 4, 12 (sum 1,326); class representatives.
5. Identity check: average equity share per player equals 1/n for n = 2..9 within Monte Carlo error.
6. Benchmarks as tests (informal but reproducible; tolerances stated): AA about 85.3% heads-up and about 49% against five opponents; 72o about 34.5% heads-up and about 8.6% against five, with 72o below 32o from six players on. Record them as INFORMATIONAL/VERIFIED findings noting the source is an informal simulation video and our own computation is the evidence.
7. Notebook section 5: examples, the 1/n identity table, benchmark table, MC vs exact check, speed; figures through save_fig; findings recorded.
8. Tests: exact vs MC, class weights sum to 1,326, seed reproducibility, ties handled.
Finish per AGENTS.md (commit: feat: equity engine).
```

## PROMPT 3 — From-scratch statistics toolkit

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P3). Code in holdem_lab/stats.py (numpy/stdlib only; formulas in docstrings).
Implement: log-gamma (stdlib allowed), digamma, regularized incomplete beta and gamma, Normal/Student-t/Beta/Gamma/F/chi-square pdf, cdf and ppf as needed; Beta fit by method of moments and MLE (Newton), Gamma and Geometric fits (MoM, MLE), Poisson and Negative Binomial fits; KS statistic; chi-square goodness of fit with pooled sparse bins and the degrees-of-freedom arithmetic shown; OLS and weighted least squares with standard errors; t-interval; bootstrap (iid and block) CIs; permutation test helper; one-way and two-way ANOVA with interaction F, with a permutation-calibrated p-value; Spearman and Kendall rank correlation; order-statistic helpers (CDF of the max and min of n iid draws, tie-aware discrete version, k-th order statistic expectation by integration or simulation); PGF utilities (derive mean and variance numerically from a PGF); Brownian ruin probability exp(-2*mu*B/sigma^2).
Tests (tests/test_stats.py, scipy and statsmodels allowed): every function against scipy/statsmodels on simulated data with stated tolerances; ANOVA interaction F against statsmodels; permutation p-value approximately uniform under the null; tie-aware order-statistic CDF against brute-force enumeration on a toy discrete distribution.
Notebook: add a short "toolkit verification" subsection to section 3 showing a few from-scratch vs scipy comparisons; record a VERIFIED finding with the maximum discrepancies.
Student-owned stubs: in notebook section 3 create six clearly marked cells headed "YOUR CODE" with function STUBS ONLY (signature, docstring giving the exact formula and the expected inputs and outputs, and `raise NotImplementedError`) for the six functions in CONTEXT.md Section 2A: five-card category, Beta MoM and MLE, chi-square GOF statistic and p-value, tie-aware order-statistic CDF of the maximum, Geometric/PGF moments, WLS fit of beta_n. After each stub add a checker cell that tests the student's function against the toolkit, scipy or exact values and prints PASS or FAIL with the difference. Add a flag USE_STUDENT = False; while False, later cells call the toolkit versions; when the human sets it True and all checkers pass, later cells call the student's functions. Do NOT implement the six functions in the notebook.
Finish per AGENTS.md (commit: feat: statistics toolkit and student stubs).
```

## PROMPT 4 — G1 and G2: winning hand vs n, equity distribution, break-even range

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P4). Code in holdem_lab/analysis.py.
1. G1: for n = 2..9 simulate many full tables (seeded, sizes from MODE) and record the category and strength of the winning hand, the probability of a split pot, expected winning category. Compare with the tie-aware iid-max model F(x)^n using the exact single-player 7-card strength distribution; quantify the gap (total variation distance, difference in expected category, difference in split-pot probability) with CIs; report the SIGN of the discrepancy neutrally; compute the exact two-player case where feasible.
2. G2: compute equity share for all 169 classes at each n with the equity engine (quick mode: precision target about 0.3 percentage points; full mode: about 0.1; cache to data/cache and write figures/data/equity_table.csv). Verify the combo-weighted mean equals 1/n. Fit Beta by MoM and MLE to the distribution of equity over a random starting hand, with bootstrap CIs, a KS statistic and a chi-square goodness-of-fit table with equal-probability bins and the degrees of freedom shown (this is goodness-of-fit Table 2). Compute the fraction of hands with equity above 1/n by n, and the return per unit staked R_n = n*E_n - 1 for every class (heatmap for the 13x13 grid at n = 2, 6, 9; lines for AA, 72o, 32o, a suited connector).
3. Figures: winning-category stacked bars by n; F^n vs exact vs simulation; split-pot probability by n; equity densities with Beta fits; share of hands above 1/n by n; R_n curves and heatmaps. All via save_fig with data CSVs.
4. Findings: H1 (SUPPORTED/NOT SUPPORTED/INCONCLUSIVE with numbers), H2 (identity VERIFIED; Beta fit; fraction above 1/n trend), plus the R_n observation that aces gain return per unit staked as n grows while 72o loses (report the numbers; this is within the equal-contribution model only).
5. Notebook section 6 with a DRAFT worked order-statistics example for n = 2 (distribution of the maximum of two draws) for the human to rewrite.
6. Tests: tie-aware F^n vs brute-force on a toy case; weighted-mean identity; Beta MoM vs scipy.
Finish per AGENTS.md (commit: feat: table size and equity distribution).
```

## PROMPT 5 — G7: the simple formula and rank stability

```
Follow CONTEXT.md Sections 6 (G7) and 8 and AGENTS.md. Do ONLY this prompt (P5). Code in holdem_lab/formula.py.
1. Use the equity table from P4 (fit-seed table). Rank stability first: Spearman and Kendall correlations between the 169-class rankings at n = 2 and each n, the classes whose rank changes most, and a verification that 72o vs 32o reorders (from six players on); a 13x13 heatmap of rank change between n = 2 and n = 9. Record H7 evidence.
2. Fit and compare, for each n, the candidates M0, M1 (with and without kappa_n), M2, M3 and M4 from CONTEXT.md Section 8 using only fit-seed data; estimate beta_n with weighted least squares on the log scale and test for a smooth law in n; M4 adds a multiway-potential term (suited, connectedness, gap) with n-dependent coefficients. Leave-hand-classes-out cross-validation weighted by combo counts; report mean absolute error, maximum error in percentage points, calibration slope and AIC per model and per n.
3. Select the simplest formula within the tolerance; check the 1/n constraint, monotonicity in n, and that premium pairs fall steeply (for example aces below 50% at six players). If nothing meets tolerance, report the best with its actual error and the verdict INCONCLUSIVE or NOT SUPPORTED for H6.
4. Out-of-sample validation: simulate fresh full-table games using the validation seed; compare formula-predicted win frequency with realized, binned calibration plot and error table.
5. Write the formula card into findings (formula, parameters with SEs, error table, limits of validity, a worked by-hand example with real numbers, a plain three-line phone recipe) and export data/formula_params.json for the demo.
6. Figures: predicted vs actual, error heatmap by class and n, beta_n vs n, rank-change heatmap, calibration plot. Notebook section 7 with a DRAFT by-hand worked example for the human to rewrite.
7. Tests: formula reproduces held-out classes within tolerance; a planted power-law synthetic table recovers the known beta; cross-validation excludes the held-out class.
Finish per AGENTS.md (commit: feat: formula and rank stability).
```

## PROMPT 6 — G3: equity across streets

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P6). Code in holdem_lab/analysis.py.
1. For chosen hands (for example AA, 72o, 98s, AKo) and n = 2, 5, 9: equity at preflop, flop, turn and river conditional on information available, over many simulated deals. Verify the martingale property numerically; decompose variance by the law of total variance into the part resolved at each street; plot sample equity paths and the share of variance resolved by street.
2. Rao-Blackwell: compare the plain Monte Carlo win estimator with the estimator that replaces the final indicator by exact conditional equity given the flop; report the variance ratio by class and n with CIs and the equivalent sample-size gain.
3. Rule of 2 and 4: for flop and turn draws with k outs, compare 4k and 2k with exact hit probabilities; plot error vs k and state where it fails.
4. Findings: H3 with numbers. Figures via save_fig with data CSVs. Notebook section 8 with a DRAFT worked example of the tower property or outs for the human to rewrite.
5. Tests: martingale check on a toy problem; variance decomposition adds up; Rao-Blackwell estimator unbiased and lower variance on a test case; exact draw probabilities for a flush draw and open-ended straight draw.
Finish per AGENTS.md (commit: feat: streets analysis).
```

## PROMPT 7 — G4 and G8: opponents, blinds, table selection

```
Follow CONTEXT.md Sections 3, 4 (range model and limp-or-fold game) and AGENTS.md. Do ONLY this prompt (P7). Code in holdem_lab/strategy.py (and analysis.py).
1. G4: opponent range model. Rank the 169 classes by the heads-up ranking and by the multiway ranking (n = 6). For q in {8, 20, 33, 50, 100}% compute: the winning-hand distribution at n = 2..9 when all opponents play the top q%; the equity of selected classes against such opponents; how the formula parameters (beta_n) shift with q. Report how sensitive each result is to the choice of ranking, and state that a bot's preflop play rate is not the range it takes to showdown. If the PHH showdown data from P9 are available, compare the model's winning-hand distribution at the fitted q with the observed one and report the agreement; if not, record that G4 rests on simulation only.
2. Limp-or-fold showdown game with blinds (CONTEXT.md Section 4): implement it exactly as specified, with documented rules for the cases when nobody calls, only the big blind remains, and ties. Strategies: play the top q% under a ranking (the five archetypes), an equity-rule strategy (play if formula equity > 1/n + margin), and random. Return profit per hand in big blinds.
3. Blind-cost break-even: derive and verify numerically the closed form for expected profit per hand of a range of width q at an n-player table (blind cost per folded hand averaged over positions vs the expected gain from played hands); report the break-even q by n and show why very tight ranges bleed heads-up.
4. Table selection (G8): a focal-strategy x opponent-mix design (for example mixes: all-wild, mixed, all-tight, all-medium) with many replicates of thousands of hands each; two-way ANOVA on profit per hand with the interaction F-test, permutation-calibrated p-value, and an effect size (eta-squared); a profit heatmap strategy x mix with CIs. Report whether strategy rankings change across mixes.
5. Findings: H4 and H8 with numbers, including negative results; figures via save_fig with data CSVs; notebook section 9.
6. Tests: pot and blind accounting conserves chips; strategy with q=100% plays every hand; closed-form blind-cost vs simulation within CI; ANOVA on planted effects recovers the interaction.
Finish per AGENTS.md (commit: feat: opponents, blinds, table selection).
```

## PROMPT 8 — G5 and G6: luck, sample size, waiting times

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P8). Code in holdem_lab/analysis.py.
1. G5 using the limp-or-fold game from P7: skewness and excess kurtosis of per-hand profit and of N-hand sums for N from 1 to 10,000 (CLT convergence); hands needed to separate two strategies at 95% confidence (power calculation, checked by simulation); F-test and bootstrap comparison of outcome variance between strategies; ruin probability by simulation vs the Brownian approximation exp(-2*mu*B/sigma^2) for several bankrolls B (in big blinds).
2. G6: hands until pocket aces, any pair and a suited hand: Geometric fit and empirical waits with a chi-square goodness-of-fit table (pooled tail bins, degrees of freedom shown; goodness-of-fit Table 3); derive mean and variance from the PGF and verify numerically; Poisson vs Negative Binomial counts of rare hands per 1,000 hands; streak statistics in pure luck.
3. Findings: H5 and G6 results with numbers; figures via save_fig with data CSVs; notebook section 10 with a DRAFT worked example (geometric waiting time via PGF) for the human to rewrite.
4. Tests: PGF moments match closed forms; power calculation vs simulation; ruin approximation sanity on a random walk with known drift.
Finish per AGENTS.md (commit: feat: luck and waiting times).
```

## PROMPT 9 — Data: UCI and PHH, cleaning, goodness of fit, real-play comparison

```
Follow CONTEXT.md (Section 9) and AGENTS.md. Do ONLY this prompt (P9). Run it right after P3.
Part A: UCI Poker Hand (required dataset).
1. Get the data into data/external/uci/ using the page Download button instructions in README.md, or ucimlrepo fetch_ucirepo(id=158), or the OpenML mirror (id 1567); verify the URL and file sizes; if a download is blocked, write exact manual steps into README.md and stop for the human. Record source, URL, DOI (10.24432/C5KW38), license and access date in a data card in findings.md and in the notebook.
2. Cleaning and EDA before any model: column names, types, ranges (suits 1-4, ranks 1-13), missing values, duplicates, class frequencies per split, summary statistics (mean, variance, skewness, kurtosis) of the rank features, histograms.
3. Verification: run our five-card evaluator on every labelled hand (map UCI suit and rank coding to ours; Ace = rank 1, high or low); report the agreement rate and document every mismatch (for example royal flush labelling). Record VERIFIED or FAILED CHECK.
4. Goodness-of-fit Table 1: first check which split, if any, follows natural class frequencies (compare each split with the exact probabilities and report), then chi-square of observed class counts vs exact probabilities with pooled sparse classes and degrees of freedom shown, plus a Monte Carlo multinomial cross-check. Note that UCI hands are ordered; explain what that implies for class proportions. A rejection is reported as a finding with its cause, not hidden.
Part B: PHH real play (core real-data section).
5. Follow the access plan in CONTEXT.md Section 9 D2 in order. First tell the human exactly what you found (what the GitHub repository includes; the Zenodo file is one 20.3 GB zip) and ask for confirmation of disk space and bandwidth before any download over 1 GB; try partial extraction of only the HandHQ hands if the human agrees and the server supports range requests; otherwise use the small fallback sets (Pluribus 10,000 hands, WSOP 83 hands) and say clearly that they are a limited illustration. If nothing can be obtained, record an INFORMATIONAL finding "real-play data not obtained" and stop Part B.
6. Parser (pokerkit isolated in one optional function, installed only for this step): per hand extract seats dealt, players reaching showdown, revealed hole cards, board, winners; evaluate each showdown hand with our evaluator; cache parsed showdowns to data/cache and figures/data. Log the number of hands removed at each cleaning step and summarize the retained sample (by table size, stakes, site).
7. Analyses: distribution of table sizes; winning-hand category by number of players dealt and at showdown with CIs, compared with the G1 model; an estimate of effective tightness q by method of moments or maximum likelihood from the showdown data (state the selection bias plainly); goodness of fit (chi-square) of the observed winning-category distribution against the model at the fitted q. Report disagreement honestly.
8. Notebook sections 2 and 11 (data cards, cleaning logs, EDA, tables, figures via save_fig) and findings for H-related results. Tests with small fixture files for both loaders and the parser.
Finish per AGENTS.md (commit: feat: data, goodness of fit and real-play comparison).
```

## PROMPT 10 — demo.py (one file) — EXTRA, run last

```
Follow CONTEXT.md Section 12 and AGENTS.md. Do ONLY this prompt (P10). This is an EXTRA: nothing from the demo or from G9 is cited as a result in the report, slides or abstract; keep the demo's help panel clearly labelled as a simulation-based aid. Everything in the single file demo.py (server, game engine, bots, HTML for phones and host as strings or a small templates folder only if unavoidable), reusing holdem_lab modules, data/formula_params.json and figures/data/equity_table.csv.
1. Rooms: short code, max 9 seats, extra joiners become spectators, nicknames only (sanitized), reconnect token, host controls (start, pause, next round, bots on/off, speed). In-memory only, no personal data.
2. Game engine: rounds with an optional preflop play/fold decision; deck and RNG on the server with the demo seed and an audit log; hole cards sent only to the owning seat; streets revealed by the host; showdown with split pots; chip accounting; bots using strategies from strategy.py fill empty seats.
3. Per-seat payload: hand class, empirical table row for the current n (P_win, P_tie, equity with SE), formula estimate with the by-hand recipe, the 13x13 chart cell, and a guess slider. Unaided guess first, then the aid is revealed, so both guesses are logged. Recompute when n changes. Post-flop: outs counter, rule of 2 and 4 estimate and exact equity; host page shows live equity lines per seat over streets, pot, chips, leaderboard by chips and by calibration, crowd average vs truth, and a table-size panel showing how a chosen hand's equity moves as players fold (empirical table with formula overlay). Spectators get a shared poll question and no private information.
4. QR join: detect the LAN IP, print and render a QR code for http://<ip>:<port>/join?room=<code> on the host page; warn if only loopback is found; document hotspot use.
5. Phone UI: mobile-first, four-colour deck, large text and touch targets, light and dark modes, screen-reader labels, a one-line hand explainer, and a visible banner "educational probability demo, play chips only".
6. CLI: --bots N, --rounds N, --port, --seed, --no-browser, --selftest (pre-flight PASS/FAIL checklist: data files present, server starts, a simulated 9-seat bots-only session completes, QR generated, LAN IP detected), --record and --from-recording (replay of host events as a backup). Logs to data/demo_logs/*.jsonl (anonymized seat ids; guesses, true equity, decisions, outcomes).
7. Notebook section 12: function in analysis.py that reads demo logs and records G9 findings: unaided vs aided absolute error and Brier-style score, paired difference with a t-interval and a bootstrap CI clustered by seat, error vs hand strength and vs n, crowd average vs truth; clearly labelled as a small convenience sample; skip gracefully when no logs exist.
8. Security and robustness: input validation, message size limits, rate limiting, seat cap, graceful disconnects. Tests in tests/test_demo.py with an async client: seat cap 9 and spectator overflow; private cards never reach other sockets; n recomputation after a fold; chip conservation; bots-only run completes 20 rounds; recording replay identity.
9. README.md: demo day checklist (hotspot, two-phone test, selftest 30 minutes before, recording, 10-minute talk path, bots-only fallback).
Finish per AGENTS.md (commit: feat: demo).
```

## PROMPT 11 — Final notebook QA, claims ledger, support.md

```
Follow CONTEXT.md and AGENTS.md. Do ONLY this prompt (P11).
1. Make holdem_lab.ipynb run top to bottom on a clean kernel in quick mode (report the measured run time) and verify that full mode works (report an estimate of its run time rather than running it if it would take long). Cells are short and commented; heavy results are cached with a recompute flag; every from-scratch step is followed by a verification cell (scipy/statsmodels/library) with matching numbers.
2. Claims ledger: register every claim the project intends to make (including the formula's accuracy, H1-H9 verdicts, the benchmark checks, the scope map) with its finding ID; the final cell calls assert_all_claims_backed() and write_findings(). The run must fail if a claim lacks evidence.
3. Check reports/findings.md: it must contain every finding from P1-P10 (including failures and negative results), the hypotheses summary table, the verification log, the formula card, the scope map and the limitations; every figure listed exists in figures/ with its CSV in figures/data/; remove orphan files. List any section missing and fix it.
4. Notebook final sections: unit coverage table (where Units 1-4 are used), the scope map (in scope, partly, off limits, future study) and limitations written from the real results.
5. reports/support.md (one file): required report layout with, per section, the figures, tables and numbers to include (file paths and finding IDs); slide plan for 10-15 slides with the figure per slide and a 10-minute timing guide; 40 likely viva questions with concise answers using real numbers (our project: order statistics and independence, the 1/n identity, the formula and its error, rank reversal, Rao-Blackwell, the martingale, selection bias, blind cost, ANOVA interaction, limits; plus basics of Units 1-4: definitions, moments of standard distributions, CLT statement, order-statistic CDF, PGF facts, t/chi-square/F). No finished report prose.
6. Run pytest, ruff and the notebook; list anything unverified; update CONTEXT.md including remaining human to-dos.
7. Standalone check: add the final notebook cell that re-runs the inline from-scratch cells with the holdem_lab import blocked and confirms identical numbers; confirm the notebook opens and runs with only the .ipynb plus the CSVs and data files in a fresh folder; list anything that fails.
8. Disclosure check: confirm README.md and the notebook's first cell contain the disclosure statement (blanks untouched), that findings.md lists what was AI-written vs student-owned (the six functions, hand-solved examples, report and slides are marked "to be written by the author"), and that no file describes AI work inaccurately.
9. Tier check: findings.md and support.md present Tier A fully, list the chosen Tier B items, and mark the demo and G9 as EXTRA outside the report.
Finish per AGENTS.md (commit: docs: final notebook, findings and support pack).
```

## PROMPT 11b — Portable notebook: runs start to end on any machine

```
Follow CONTEXT.md (Section 11) and AGENTS.md. Do ONLY this prompt (P11b). Run it after P11 and again after the student-owned functions are implemented.
Goal: a reader who receives only the submission folder (notebook, CSVs, small data files, requirements) can run the notebook from the first cell to the last without errors.
1. Root and paths: one ROOT variable set in the first code cell (current working directory); every file is read and written relative to ROOT; no absolute paths anywhere.
2. Environment cell: print the Python version and the versions of numpy, pandas, matplotlib and any other import; if an import is missing, print the exact pip command and stop with a clear message. Create submission/requirements.txt with pinned versions of only what the notebook needs (no fastapi, uvicorn or demo packages) and note the tested Python version in the notebook header.
3. Data manifest cell: list every required file (UCI files, cached equity table, cached PHH showdown CSV or the documented fallback, figures/data CSVs) with the expected path and size; verify existence (and checksums for small files); if a file is missing print where to download it (Section 9 instructions) and either skip the dependent section with a visible "SKIPPED: data not present" message and a finding of INFORMATIONAL, or stop if the file is essential. The notebook must never crash with a bare traceback.
4. Portable mode: setup cell tries `import holdem_lab`; if it fails set USE_PACKAGE = False. Every section must work in both cases: with the package (recompute allowed) and without it (load cached results from data/cache and figures/data; the inline from-scratch cells use only numpy). Cells that need the package must check USE_PACKAGE and fall back to the cache. The package-based toolkit versions must not be needed when USE_STUDENT = True; if USE_STUDENT is False and the package is absent, print a clear message that the student-owned cells must be implemented.
5. Raw PHH data is too large to ship: the notebook loads the cached parsed showdown CSV; if absent it says so and skips the real-play comparison with an INFORMATIONAL finding rather than failing.
6. Determinism and time: all seeds from one config cell; quick mode is the default; print the elapsed time per section; target under the quick-mode time already measured. No network access during a normal run (downloads only in an explicit, off-by-default cell).
7. Fresh-environment test: create a temporary folder containing ONLY the files that would be submitted (notebook, requirements, CSVs, data files), create a new virtual environment there, install submission/requirements.txt, and execute the notebook with nbconvert from the first cell to the last; also run it once with the holdem_lab package unavailable. Report the result and fix every failure. Save the executed notebook output check as a finding (VERIFIED or FAILED CHECK).
8. README.md: how to run it (install, open, run all), expected run time, what files must sit next to the notebook.
Finish per AGENTS.md (commit: chore: portable notebook).
```

## PROMPT 12 — Hand-solved problem sheet and answer checker

```
Follow CONTEXT.md (Section 2A) and AGENTS.md. Do ONLY this prompt (P12). The human solves these by hand; you provide problems and checks, NOT solutions.
1. Create reports/hand_examples_sheet.md with six one-page hand-solved PROBLEMS built from real project numbers (read them from reports/findings.json or figures/data; never type results by hand). Each has: the setup in words, the given numbers, what to compute, and nothing else (no steps, no answers):
   E1 order statistics: distribution of the maximum of two draws from a small discrete distribution taken from the project (tie-aware);
   E2 an exact draw probability by counting (for example the flush-draw hit probability) and the 1/n identity for n = 2 on a tiny deck;
   E3 Beta distribution parameters by method of moments from the project's equity mean and variance at a chosen n;
   E4 a chi-square statistic and degrees of freedom for a small observed-versus-expected table from the project;
   E5 Geometric waiting time (mean and variance via the PGF) for pocket aces;
   E6 using the project's simple formula to estimate the win probability of a named hand at a named n, then comparing with the empirical table value.
2. Add a notebook section "Hand examples checker" with a function check_answer(example_id, value) that compares the human's number with the computed truth within a stated tolerance and prints PASS or FAIL; only after a PASS (or a second attempt) it may print the final value. It must never print worked steps or the derivation.
3. Update README.md with how to use the sheet. Do not write any solution text anywhere.
Finish per AGENTS.md (commit: docs: hand-solved problem sheet and checker).
```

## Appendix A — Resync prompt (when the agent drifts or a new chat starts)

```
Read CONTEXT.md in full. List which prompts are ticked in Section 19, inspect the repository tree and summarize in 10 lines what exists, point out mismatches between CONTEXT.md and the code, check that reports/findings.md and figures/ are consistent with what was claimed, and propose the exact next prompt. Write no code in this step. Follow AGENTS.md.
```

## Appendix B — Debug template

```
We are on PROMPT <n>; follow CONTEXT.md and AGENTS.md. Problem: <expected vs actual>. Command: `<command>`. Full error: `<paste>`. Find the root cause (reproduce and read the code first), fix it with the smallest change, add a regression test, run pytest, record any changed design decision in CONTEXT.md Section 18, and commit.
```
