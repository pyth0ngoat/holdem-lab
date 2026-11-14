# Support Pack: Report Layout, Slide Deck Plan, and Viva Preparation
## Hold'em Lab — Applied Probability Distributions (THM6242)

> **Notice to Author (CONTEXT.md Section 2):**
> This document provides scaffolds, outlines, file references, exact numbers, and 40 viva question-and-answer pairs.
> In accordance with course integrity rules, **you write the final report prose and slides in your own words**.
> This support pack organizes every empirical finding, figure path, and theoretical derivation for your submission.

---

## Part 1: Recommended Master Report Structure (20–30 Pages)

### Section 1: Introduction, Research Questions, and Scope Tiers (2–3 pages)
- **Primary Question:** How does table size $n \in [2, 9]$ alter the distribution of winning poker hands, can a simple closed-form formula estimate preflop equity, and how do real-play cash game histories reflect these dynamics?
- **Theoretical Scope:** Units 1, 2, 3, and 4 of THM6242.
- **Scope Tiers Table:**
  - *Tier A (Core — Main Report):* G1 (Order Statistics vs Card Sharing), G2 (169-Class Equity & Beta Fits), G6 (Waiting Times & PGFs), G7 (Formula Specification & Rank Stability), Real-Data Ingestion (UCI Table 1 & PHH Selection Bias).
  - *Tier B (Chosen Deep Dives — 1 page each):* G3 (Street Martingale & Rao-Blackwell), G4/G8 (Opponent Ranges, Blind Costs & Table Selection ANOVA).
  - *Tier C (Extra — Outside Master Report Claims):* G9 Live human calibration & `demo.py` websocket demo.
- **Key Benchmark Numbers to Cite:**
  - $E[\text{equity of random starting hand}] = 1/n$ exactly (Finding `V-IDENTITY-1N`).
  - AA heads-up equity: $85.01\% \pm 0.36\%$ (empirical), literature $85.20\%$ (Finding `V-EQUITY-BENCH`).
  - 72o heads-up equity: $35.03\% \pm 0.47\%$ (empirical), literature $34.59\%$ (Finding `V-EQUITY-BENCH`).

---

### Section 2: Combinatorial Foundations and Vectorized Hand Evaluator (3–4 pages)
- **Target Unit:** Unit 2 (Combinatorics and Discrete Probability).
- **Mathematical Derivation:**
  - 5-card combinations: $\binom{52}{5} = 2,598,960$.
  - 7-card combinations: $\binom{52}{7} = 133,784,560$.
  - Exact 7-card Royal Flush count: 4,324 ($P \approx 3.232 \times 10^{-5}$, 1 in 30,940).
- **Vectorized Architecture:**
  - Pure-NumPy base-15 kicker representation with rank-bitmask lookup.
  - Measured throughput: $\sim 3.95 \times 10^6$ hands/second (Target $> 1.0 \times 10^6$).
  - Full Treys benchmark validation: 20,000 hands tested with 0 category mismatches and 0 rank ordering errors (Finding `V-CARD-EVAL`).
- **Figures to Include:**
  - `figures/V-CARD-COUNTS_exact_probabilities.png` (Exact 5-card vs 7-card log probability distributions).
- **Data Tables to Include:**
  - `figures/data/V-CARD-COUNTS_exact_probabilities.csv`.

---

### Section 3: Open Data Ingestion, Cleaning, and Goodness-of-Fit Table 1 (3–4 pages)
- **Target Units:** Unit 3 (Chi-Square Goodness of Fit) & Unit 5 (Open Data Engineering).
- **Datasets Analyzed:**
  1. *UCI Poker Hand Dataset (DOI: 10.24432/C5KW38):* 1,025,010 ordered 5-card hands (25,010 train, 1,000,000 test). Evaluator achieved 100.0% agreement across all 1.025M hands (Finding `V-UCI-EVAL`).
  2. *PHH Real-Play Cash Games (Zenodo DOI: 10.5281/zenodo.10796885):* 441 files, 311 MB from HandHQ (July 2009 cash games, 25NL to 1000NL).
- **Goodness-of-Fit Table 1 (Compulsory Course Table):**
  - *Testing Split (1M hands):* $\chi^2 = 1.66$, $\text{df} = 8$, $p = 0.990$ (Natural multinomial class frequencies verified).
  - *Training Split (25,010 hands):* $\chi^2 = 16.12$, $\text{df} = 7$, $p = 0.024$ (Statistically significant rejection due to intentional over-sampling of rare hands in UCI benchmark design; Finding `F-UCI-GOF`).
- **Showdown Selection Bias Analysis (PHH):**
  - High Card hands drop from 17.41% in random deals to 3.82% at showdown ($-78.1\%$ relative drop).
  - Two Pair increases from 23.50% to 33.15% ($+41.1\%$), and Full House increases from 2.60% to 7.91% ($+204.6\%$).
  - Effective preflop tightness estimated at $q \approx 0.25 - 0.50$ (Finding `F-PHH-TIGHTNESS-Q`).
- **Figures to Include:**
  - `figures/V-UCI-EVAL_uci_frequencies.png`
  - `figures/F-PHH-TABLESIZES_table_distribution.png`
  - `figures/F-PHH-WINNING-CAT_winning_categories.png`
- **Data Tables to Include:**
  - `figures/data/V-UCI-EVAL_table1_gof.csv`
  - `figures/data/phh_showdowns.csv`

---

### Section 4: Table Size Dynamics and Breakdown of Independent Order Statistics (G1) (3–4 pages)
- **Target Units:** Unit 4 (Order Statistics) & Unit 2 (Discrete Distributions).
- **Core Hypothesis H1 (Supported):**
  - The naive independent order-statistic model $F_{\max}(x) = [F(x)]^n$ fails for Texas Hold'em because all players share the 5 community board cards.
  - Total Variation (TV) distance between empirical winning category distribution and independent $F^n$ model spans $0.051$ ($n=2$) to $0.137$ ($n=9$).
  - Split-pot probability is vastly overestimated by $F^n$ (by up to 20 percentage points) because independent draws ignore distinct kicker resolutions.
- **Figures to Include:**
  - `figures/F-G1-IID-VS-SIM_iid_vs_simulation.png`
  - `figures/F-G1-WINNING-CAT_winning_category_by_n.png`
- **Data Tables to Include:**
  - `figures/data/F-G1-IID-VS-SIM_iid_vs_simulation.csv`
  - `figures/data/F-G1-WINNING-CAT_winning_category_by_n.csv`

---

### Section 5: The 169 Starting-Hand Equities and Beta Distribution Fits (G2) (3–4 pages)
- **Target Units:** Unit 1 (Exchangeability) & Unit 2 (Continuous Distributions: Beta).
- **Core Hypothesis H2 (Supported):**
  - Evaluated full 169 canonical hand classes across $n = 2..9$ (1,352 table cells).
  - Combo-weighted equity identity strictly holds: $\sum_{h} w_h E_n(h) = 1/n$ with numerical error $< 10^{-15}$.
  - Percentage of profitable starting hands ($E_n > 1/n$) steadily declines as table size expands: $47.8\%$ ($n=2$) $\to 41.8\%$ ($n=6$) $\to 39.4\%$ ($n=9$).
  - Return per unit staked $R_n(h) = n E_n(h) - 1$ demonstrates the leverage of premium hands: AA return expands from $+70.0\%$ ($n=2$) to $+212.6\%$ ($n=9$).
- **Goodness-of-Fit Table 2 (Beta Equity Model):**
  - Method of Moments estimates: at $n=6$, $\hat{\alpha} = 6.61$, $\hat{\beta} = 33.05$ with ratio $\hat{\beta}/\hat{\alpha} = 5.00 = n-1$.
  - Kolmogorov-Smirnov test statistics remain small ($D_{\text{KS}} = 0.049$ to $0.081$).
- **Figures to Include:**
  - `figures/F-G2-EQUITY-DIST_equity_distributions_beta.png`
  - `figures/F-G2-RETURN-STAKED_return_per_unit_staked.png`
  - `figures/F-G2-HEATMAPS_equity_heatmaps.png`
- **Data Tables to Include:**
  - `figures/data/equity_table.csv`
  - `figures/data/F-G2-EQUITY-TABLE_table2_beta_gof.csv`

---

### Section 6: Starting-Hand Rank Stability and The Simple Formula (G7) (4–5 pages)
- **Target Units:** Unit 1 (Mathematical Modelling) & Unit 4 (Rank Correlation).
- **Core Hypotheses H6 & H7 (Supported):**
  - *Rank Stability Decay (H7):* Spearman rank correlation relative to heads-up decays from $\rho = 1.000$ ($n=2$) to $\rho = 0.829$ ($n=9$) (Kendall $\tau = 0.647$).
  - *Rank Inversion:* Suited connectors gain up to 67 rank places in full ring (65s rises from rank 133 to 66), while unsuited high cards drop by 61 places (K6o drops from rank 61 to 122). 72o beats 32o heads-up ($35.0\%$ vs $32.2\%$), but reverses at $n \ge 6$ ($9.1\%$ vs $9.4\%$).
  - *Formula Specification (H6):* Model M1 power-law $\hat{E}_n = \frac{1}{n} (2 p_2)^{\beta_n}$ with smooth law $\beta_n = 1.0 + 0.230 \sqrt{n - 2}$.
  - *Two-Feature Model M4:* Incorporates suitedness and gap bonuses; achieves cross-validated Mean Absolute Error (MAE) of $0.92$ pp ($< 1.0$ pp tolerance).
  - *Out-of-Sample Validation:* Verified on independent validation seed 1337 with calibration correlation $r = 0.9868$.
- **Figures to Include:**
  - `figures/F-G7-RANK-CHANGE_rank_change_heatmap.png`
  - `figures/F-G7-BETA-LAW_beta_vs_n.png`
  - `figures/F-G7-CALIBRATION_oos_calibration.png`
  - `figures/F-G7-PRED-VS-ACTUAL_predicted_vs_actual.png`
- **Data Tables to Include:**
  - `data/formula_params.json`
  - `figures/data/F-G7-CALIBRATION_oos_calibration.csv`

---

### Section 7: Rare Hand Waiting Times, PGF Moments, and Table 3 GOF (G6) (2–3 pages)
- **Target Units:** Unit 4 (Probability Generating Functions) & Unit 2 (Geometric Distribution).
- **Mathematical Derivations:**
  - Geometric distribution $P(X = k) = (1-p)^{k-1} p$ on $k \in \{1, 2, \dots\}$.
  - PGF: $G(s) = \frac{ps}{1 - (1-p)s}$.
  - Factorial moments: $G'(1) = 1/p$, $G''(1) = \frac{2(1-p)}{p^2}$.
  - Variance: $\text{Var}(X) = G''(1) + G'(1) - [G'(1)]^2 = \frac{1-p}{p^2}$.
  - For Pocket Aces ($p = 1/221$): $E[X] = 221.0$, $\text{Var}(X) = 48,620.0$, $\sigma = 220.50$. Numerical PGF differentiation matched theory to $10^{-5}$ error.
- **Goodness-of-Fit Table 3:**
  - Verified Geometric distribution fits on 30k–100k simulated deals for Pocket Aces ($\chi^2 = 3.82, \text{df}=5, p=0.575$), Any Pocket Pair ($\chi^2 = 4.19, \text{df}=6, p=0.651$), and Suited Hands ($\chi^2 = 2.45, \text{df}=5, p=0.784$). All $p > 0.05$ confirm goodness of fit.
- **Figures to Include:**
  - `figures/F-G6-WAITING-TIMES_waiting_times_geometric.png`
- **Data Tables to Include:**
  - `figures/data/F-G6-WAITING-TIMES_table3_geometric_gof.csv`

---

### Section 8: Selected Tier B Topics (2 pages total)
#### Sub-section 8A: Street Martingale and Rao-Blackwell Variance Reduction (G3)
- **Target Unit:** Unit 1 (Conditional Expectation & Variance Decomposition).
- **Core Hypothesis H3 (Supported):**
  - Equity process $(E_0, E_1, E_2, E_3)$ across preflop, flop, turn, river satisfies the martingale property: $E[E_{t+1} \mid \mathcal{F}_t] = E_t$ (expectation error $< 0.015$).
  - Law of total variance verified: $\text{Var}(E_3) = \sum_{t=0}^2 \text{Var}(\Delta E_t)$ with incremental covariances $\text{Cov}(\Delta E_s, \Delta E_t) \approx 0$.
  - Rao-Blackwell conditioning on flop reduces Monte Carlo showdown variance by $2.5\times$ to $4.3\times$ (mean $3.19\times$).
  - Rule of 2 and 4 breakdown: exact flop-to-river probability is quadratic $P = (93k - k^2)/2162$; Rule of 4 overestimates large draws ($+5.88$ pp error at 15 outs, $+14.06$ pp at 21 outs).
- **Figures:** `figures/F-G3-MARTINGALE_street_equity_martingale.png`, `figures/F-G3-RULE-2-4_rule_of_2_and_4_error.png`.

#### Sub-section 8B: Opponents, Blinds, and Table Selection ANOVA (G4 & G8)
- **Target Units:** Unit 3 (Two-Way ANOVA) & Unit 1 (Expected Return).
- **Core Hypotheses H4 & H8 (Supported):**
  - Against tight opponents (top 8% Rock), 72o equity drops from $35.0\%$ to $26.8\%$ heads-up and collapses by over $60\%$ relative multiway; AA retains $>93\%$ baseline equity (Finding `F-G4-OPPONENTS-RANGES`).
  - Closed-form blind drain is $1.5/n$ bb/hand. Tight ranges suffer insurmountable losses heads-up (Rock loses $-28.5$ bb/100 due to $75.0$ bb/100 blind cost), but turn profitable at 9-max ($+5.1$ bb/100 for TAG).
  - Two-way factorial ANOVA on Strategy $\times$ Opponent Mix reveals a highly significant interaction ($F = 12.18, p < 0.001, \eta^2 = 0.071$), confirming statistically that the optimal playing strategy depends on table lineup (Finding `F-G8-TABLE-SELECTION`).
- **Figures:** `figures/F-G4-OPPONENT-RANGES_opponent_tightness_equity.png`, `figures/F-G8-BLINDS-BREAKEVEN_blind_cost_breakeven.png`, `figures/F-G8-TABLE-SELECTION_strategy_mix_anova_heatmap.png`.

---

### Section 9: Scope Map, Limitations, and Future Directions (2 pages)
- Reproduce Scope Map from CONTEXT.md Section 3.
- Discuss open limitations: PHH showdown selection bias, limp-or-fold abstraction, board card correlation, and discrete kicker resolution.

---

## Part 2: Slide Deck Plan (10–15 Slides, 10-Minute Presentation)

| Slide | Title | Key Talking Points & Real Numbers | Recommended Figure / Table | Timing |
|---|---|---|---|---|
| **1** | Hold'em Lab: Table Dynamics & Probability | Research question; applied probability framing; Units 1–4; play chips only. | Title slide layout | 0:45 |
| **2** | Combinatorics & High-Throughput Engine | Exact 5-card ($2.6\times 10^6$) and 7-card ($1.34\times 10^8$) counts; 3.95M hands/s NumPy evaluator; Treys 20k validation (0 errors). | `figures/V-CARD-COUNTS_exact_probabilities.png` | 0:45 |
| **3** | Real Data: UCI & Goodness-of-Fit Table 1 | 1,025,010 hands; Test split matches theory ($p=0.990$); Training split oversamples rare hands ($p=0.024$). | `figures/V-UCI-EVAL_uci_frequencies.png` | 1:00 |
| **4** | Real Play: Selection Bias at Showdown | 2009 HandHQ cash games; High Card drops from 17.4% to 3.8% at showdown; Two Pair & Full House elevated; $q \approx 0.25-0.50$. | `figures/F-PHH-WINNING-CAT_winning_categories.png` | 1:00 |
| **5** | Table Size Dynamics: Card Sharing (H1) | Naive $F_{\max} = F^n$ fails due to shared board cards; TV distance $0.05-0.14$; split pots overestimated by 20 pp. | `figures/F-G1-IID-VS-SIM_iid_vs_simulation.png` | 1:00 |
| **6** | 169 Starting Hands & Beta Fits (H2) | Combo-weighted equity averages $1/n$ exactly; Beta MoM ($\hat{\beta}/\hat{\alpha} = n-1$); Profitable fraction shrinks ($47.8\% \to 39.4\%$). | `figures/F-G2-EQUITY-DIST_equity_distributions_beta.png` | 1:00 |
| **7** | Rank Stability Decay & Inversion (H7) | Spearman $\rho$ decays $1.000 \to 0.829$; Suited connectors rise 67 ranks; 72o vs 32o reverses at $n \ge 6$. | `figures/F-G7-RANK-CHANGE_rank_change_heatmap.png` | 1:00 |
| **8** | Simple Formula & Smooth Law (H6) | Power law $\hat{E}_n = \frac{1}{n} (2p_2)^{\beta_n}$; Smooth law $\beta_n = 1.0 + 0.230\sqrt{n-2}$; Out-of-sample $r=0.9868$; MAE $< 1.0$ pp. | `figures/F-G7-CALIBRATION_oos_calibration.png` | 1:00 |
| **9** | Waiting Times & PGF Derivations | Pocket Aces $p=1/221$; $E[X] = 221$, $\text{Var}(X) = 48,620$; Table 3 GOF ($p > 0.05$); PGF moments match theory to $10^{-5}$. | `figures/F-G6-WAITING-TIMES_waiting_times_geometric.png` | 0:45 |
| **10** | Street Martingale & Rao-Blackwell (H3) | Discrete martingale verified; Rao-Blackwell $3.2\times$ variance reduction; Rule of 4 quadratic error ($+5.9$ pp at 15 outs). | `figures/F-G3-MARTINGALE_street_equity_martingale.png` | 1:00 |
| **11** | Blind Costs & Table Selection ANOVA (H8) | Blind drain $1.5/n$ bb; Rock loses heads-up ($-28.5$ bb/100) but wins at full ring; ANOVA interaction $F=12.18, p<0.001$. | `figures/F-G8-TABLE-SELECTION_strategy_mix_anova_heatmap.png` | 1:00 |
| **12** | Conclusions, Academic Limits & Viva Q&A | Summary of hypotheses H1–H8 supported; selection bias & model abstractions; hand-solved problem sheet. | Scope Map & Summary Table | 0:45 |

---

## Part 3: 40 Viva Questions and Concise Answers with Real Numbers

### Category A: Core Probability Theory & Course Foundations (Units 1–4)

**Q1: What is the formal definition of equity in Texas Hold'em, and how does it relate to conditional expectation?**
> **Answer:** Equity of a starting hand class $h$ is defined as $E_n(h) = E[\mathbf{1}_{\text{win}} + \sum_{k=2}^n \frac{1}{k}\mathbf{1}_{\text{tie}, k} \mid H_1 \in h]$, which is the conditional expectation of pot share given the player's hole-card class, integrating over the uniform distribution of opponents' unseen cards and board cards.

**Q2: Prove why the combo-weighted average of equity across all 169 starting hand classes must equal $1/n$.**
> **Answer:** In a showdown of $n$ exchangeable players with cards dealt uniformly at random from a shuffled deck, pot shares strictly partition the pot ($\sum_{i=1}^n S_i = 1$). By linearity of expectation and symmetry under permutation, $E[\sum_{i=1}^n S_i] = \sum_{i=1}^n E[S_i] = n E[S_1] = 1$, yielding $E[S_1] = 1/n$ identically.

**Q3: State the tower property of conditional expectation and explain how it applies across betting streets.**
> **Answer:** For nested information $\sigma$-algebras $\mathcal{F}_s \subset \mathcal{F}_t$, the tower property states $E[E[X \mid \mathcal{F}_t] \mid \mathcal{F}_s] = E[X \mid \mathcal{F}_s]$. In poker, where $X = \mathbf{1}_{\text{win}}$ and $\mathcal{F}_t$ represents cards revealed up to street $t$, the street equity process $E_t = E[X \mid \mathcal{F}_t]$ forms a discrete martingale satisfying $E[E_{t+1} \mid \mathcal{F}_t] = E_t$.

**Q4: How does the law of total variance decompose the variance of equity across streets?**
> **Answer:** $\text{Var}(X) = E[\text{Var}(X \mid \mathcal{F}_t)] + \text{Var}(E[X \mid \mathcal{F}_t])$. Telescoping across preflop, flop, turn, and river decomposes total outcome variance into the sum of variances of orthogonal increments $\text{Var}(E_3) = \sum_{t=0}^2 \text{Var}(\Delta E_t)$ because increments have zero covariance ($E[\Delta E_s \Delta E_t] = 0$ for $s < t$).

**Q5: What is the theoretical principle behind Rao-Blackwell variance reduction in Monte Carlo equity estimation?**
> **Answer:** The Rao-Blackwell theorem states that if $T$ is an estimator and $\mathcal{F}$ is a conditioning sub-$\sigma$-algebra, then $E[\text{Var}(T \mid \mathcal{F})] \ge 0$, so $\text{Var}(E[T \mid \mathcal{F}]) \le \text{Var}(T)$. In our project, conditioning on the flop and evaluating exact conditional equity reduced Monte Carlo variance by $2.5\times$ to $4.3\times$ (mean $3.19\times$) compared to raw binary showdown indicators.

**Q6: What is a Probability Generating Function (PGF) and how are its derivatives used to compute moments?**
> **Answer:** For non-negative integer random variable $X$, $G(s) = E[s^X] = \sum_{k=0}^\infty P(X=k) s^k$. The first derivative evaluated at $s=1$ gives the mean $G'(1) = E[X]$, the second gives the factorial moment $G''(1) = E[X(X-1)]$, and variance is $\text{Var}(X) = G''(1) + G'(1) - [G'(1)]^2$.

**Q7: Derive the mean and variance of a Geometric distribution on $\{1, 2, \dots\}$ using its PGF.**
> **Answer:** $G(s) = \frac{ps}{1-(1-p)s}$. First derivative: $G'(s) = \frac{p}{(1-(1-p)s)^2}$, so $G'(1) = 1/p$. Second derivative: $G''(s) = \frac{2p(1-p)}{(1-(1-p)s)^3}$, so $G''(1) = \frac{2(1-p)}{p^2}$. Variance $= \frac{2(1-p)}{p^2} + \frac{1}{p} - \frac{1}{p^2} = \frac{1-p}{p^2}$.

**Q8: Explain the difference between Method of Moments (MoM) and Maximum Likelihood Estimation (MLE) for the Beta distribution.**
> **Answer:** MoM equates sample moments to theoretical moments: $\bar{x} = \frac{\alpha}{\alpha+\beta}$ and $s^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$, solving algebraically for $\hat{\alpha}, \hat{\beta}$. MLE maximizes the log-likelihood $\sum_{i} [(\alpha-1)\ln x_i + (\beta-1)\ln(1-x_i) - \ln B(\alpha, \beta)]$, requiring numerical root-finding via digamma functions $\psi(\alpha) - \psi(\alpha+\beta) = \overline{\ln x}$.

**Q9: State the Central Limit Theorem and explain why convergence is slow for poker winnings.**
> **Answer:** For i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, $\frac{\sum X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$. Poker per-hand profit exhibits heavy right skewness ($\gamma_1 = 2.10$) and high excess kurtosis ($\gamma_2 = 4.65$) due to rare large pot wins. By the Berry-Esseen theorem, the approximation error decays as $\mathcal{O}(\gamma_1 / \sqrt{N})$, requiring thousands of hands before normal approximations become valid.

**Q10: What is a two-way factorial ANOVA interaction effect, and how is it calibrated by permutation?**
> **Answer:** In a design $Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$, the interaction $(\alpha\beta)_{ij}$ measures whether the effect of factor A differs across levels of factor B. The interaction $F$-test statistic $F = \frac{\text{MS}_{AB}}{\text{MS}_E}$ is tested against permutations that shuffle residuals or cell labels, calibrating exact non-parametric p-values without assuming strict normality.

---

### Category B: Combinatorics, Evaluator, and Open Data (Units 2, 3, 5)

**Q11: Why are there 169 canonical starting hand classes in Texas Hold'em?**
> **Answer:** A 52-card deck has $\binom{52}{2} = 1,326$ starting deals. Suit isomorphism partitions them into 13 pocket pairs ($\binom{4}{2}=6$ combos each, 78 combos), 78 suited hands ($\binom{4}{1}=4$ combos each, 312 combos), and 78 offsuit hands ($4 \times 3 = 12$ combos each, 936 combos): $13 + 78 + 78 = 169$ classes, summing to $78 + 312 + 936 = 1,326$ combinations.

**Q12: How many distinct 7-card poker hands exist, and what is the exact probability of a Royal Flush?**
> **Answer:** There are $\binom{52}{7} = 133,784,560$ distinct 7-card combinations. Exactly 4,324 contain a Royal Flush (4 suit choices $\times \binom{47}{2} = 4 \times 1,081 = 4,324$), giving an exact probability of $\frac{4,324}{133,784,560} \approx 3.232 \times 10^{-5}$ (1 in 30,940).

**Q13: How does your vectorized evaluator achieve ~3.95 million hands per second in pure NumPy?**
> **Answer:** It encodes 5-card and 7-card ranks as bitmasks and base-15 kicker integers. Suitedness is detected by suit-histogram binning; straights are identified via bitwise shifts and an Ace-low wrap mask (`0x100F`); kickers are packed as $\sum_{i} r_i 15^{4-i}$. All operations run on NumPy C-contiguous arrays without Python loops.

**Q14: What was the result of verifying your evaluator against Treys and the UCI dataset?**
> **Answer:** Verified against `treys` on 20,000 random hands with 0 category mismatches and 0 rank ordering errors. Verified against all 1,025,010 labelled hands in the UCI Poker Hand repository with 100.0% category agreement across both training and testing splits.

**Q15: Why did the UCI training split fail the goodness-of-fit test against theoretical hand probabilities?**
> **Answer:** Goodness-of-Fit Table 1 yielded $\chi^2 = 16.12, \text{df}=7, p = 0.024$ on the training split (25,010 hands). This statistically significant rejection occurred because the dataset creators intentionally over-sampled rare hand classes (such as Straight Flushes and Royal Flushes) to assist machine learning classifiers. The testing split (1,000,000 hands) passed with $\chi^2 = 1.66, \text{df}=8, p = 0.990$.

**Q16: What is showdown selection bias in the PHH dataset, and how did you measure it?**
> **Answer:** In real cash game hand histories, cards are only revealed when players do not fold before showdown. We measured this by comparing showdown frequencies with random deals: High Card hands collapsed from $17.41\%$ to $3.82\%$ ($-78.1\%$), while Two Pair ($33.15\%$ vs $23.50\%$) and Full House ($7.91\%$ vs $2.60\%$) were significantly elevated.

**Q17: How did you estimate the effective preflop player tightness $q$ from PHH showdowns?**
> **Answer:** We fitted $q$ by matching the observed showdown winning category distribution against simulated games where players only play the top $q\%$ of starting hands. The resulting method-of-moments estimate was $q \approx 0.25 - 0.50$, reflecting the fact that players fold weak holdings preflop.

**Q18: What table sizes predominate in the 2009 HandHQ dataset?**
> **Answer:** 6-max tables account for approximately $65\%$ of cash game hands, heads-up tables account for $25\%$, and 9-max full ring tables account for $10\%$.

**Q19: Why does pooling sparse classes matter in a chi-square goodness-of-fit test?**
> **Answer:** Cochran's rule requires expected cell counts $E_i \ge 5$ for the asymptotic $\chi^2$ approximation to hold. For rare classes (Straight Flush and Royal Flush), expected counts in small samples can fall below 1, distorting the test statistic unless adjacent tail categories are pooled and degrees of freedom adjusted accordingly.

**Q20: What is the exact combinatorial probability of hitting a flush by the river from a flop 4-flush?**
> **Answer:** On the flop, 5 cards are seen (4 of the flush suit). There are $52 - 5 = 47$ unseen cards, containing $13 - 4 = 9$ flush cards and 38 non-flush cards. The probability of missing both turn and river is $\binom{38}{2} / \binom{47}{2} = 703 / 1,081 \approx 0.65032$. Thus, the probability of hitting at least one flush card is $1 - 703/1,081 = 378/1,081 \approx 0.349676$ (34.97%).

---

### Category C: Table Size Dynamics, Equity Distributions, and The Formula (G1, G2, G7)

**Q21: Why does the independent order-statistic model $F_{\max}(x) = [F(x)]^n$ fail for Texas Hold'em?**
> **Answer:** Texas Hold'em hands are not independent: all $n$ players share the same 5 community board cards. If the board contains a pair or flush draw, every player's conditional probability of achieving that hand increases simultaneously. This positive dependence widens the TV distance to $0.051 - 0.137$ and causes $F^n$ to overestimate split pots by up to 20 percentage points.

**Q22: How does the shape of the starting-hand equity distribution change as table size $n$ increases?**
> **Answer:** At $n=2$, equities are symmetric around $0.50$ with $\text{Beta}(12.34, 12.34)$. As $n$ increases, the mean shifts to $1/n$ and the distribution becomes heavily right-skewed: at $n=6$, $\text{Beta}(6.61, 33.05)$, and at $n=9$, $\text{Beta}(5.89, 47.12)$.

**Q23: Why does the ratio $\hat{\beta}/\hat{\alpha}$ equal $n-1$ in your Beta equity fits?**
> **Answer:** For a Beta distribution, the mean is $\mu = \frac{\alpha}{\alpha+\beta}$. Because the combo-weighted mean equity strictly satisfies $\mu = 1/n$, we have $\frac{\alpha}{\alpha+\beta} = \frac{1}{n} \implies \alpha + \beta = n\alpha \implies \beta = (n-1)\alpha$, which implies $\beta/\alpha = n-1$ identically.

**Q24: What is rank reversal, and why does 72o beat 32o heads-up but lose to it at full ring?**
> **Answer:** Heads-up, high card value dominates showdowns; the 7 in 72o frequently wins when both players miss the board ($35.03\%$ vs $32.22\%$). At $n \ge 6$, winning hands require pairs, straights, or flushes; 32o has straight potential (A-2-3-4-5 wheel) whereas 72o is the largest unconnectable gap in poker, causing 32o equity ($9.37\%$) to surpass 72o ($9.14\%$).

**Q25: By how many ranks do suited connectors rise from heads-up to full ring?**
> **Answer:** Suited connectors rise by up to 67 rank positions. For example, 65s rises from rank 133 at $n=2$ to rank 66 at $n=9$, while unsuited high cards drop by up to 61 positions (K6o falls from rank 61 to 122). Spearman rank correlation decays from $1.000$ to $0.829$.

**Q26: State your simple power-law formula (Model M1) and the smooth scaling law for $\beta_n$.**
> **Answer:** $\hat{E}_n(h) = \frac{1}{n} \frac{(2 \cdot p_2(h))^{\beta_n}}{\kappa_n} \approx \frac{1}{n} (2 \cdot p_2(h))^{\beta_n}$, where $\beta_n = 1.0 + 0.230 \sqrt{n - 2}$. The exponent $\beta_n$ grows smoothly with table size, bending the heads-up equity $p_2$ to reflect multiway power.

**Q27: How does your two-feature Model M4 improve on Model M1?**
> **Answer:** Model M1 is strictly monotone in $p_2$ and cannot capture rank reversal. Model M4 adds a multiway feature vector with suitedness ($\gamma_{\text{suit}}$) and gap ($\gamma_{\text{gap}}$) bonuses: $\hat{E}_n = \frac{1}{n} (2 p_2)^{\beta_n} \exp(\gamma_{\text{suit}} \mathbf{1}_{\text{suit}} + \gamma_{\text{gap}} \text{gap})$, achieving a cross-validated MAE of $0.92$ pp ($< 1.0$ pp tolerance).

**Q28: How well did the formula calibrate on out-of-sample deals?**
> **Answer:** On independent validation seed 1337 across all 169 classes and $n = 2..9$, predicted equity achieved a Pearson correlation of $r = 0.9868$ with empirical simulation, with a linear calibration slope of $0.991$ and intercept $0.001$.

**Q29: Explain the 3-line phone recipe for players at a live table.**
> **Answer:**
> 1. Double heads-up equity: $x = 2 \times p_2$.
> 2. Raise to power: $y = x^{\beta_n}$, where $\beta_n = 1.0 + 0.230 \sqrt{n - 2}$.
> 3. Divide by table size: $\hat{E}_n = y / n$.

**Q30: What is the return per unit staked $R_n$, and why does it expand for premium hands?**
> **Answer:** $R_n(h) = n E_n(h) - 1$. For a break-even hand ($E_n = 1/n$), $R_n = 0$. For Pocket Aces, although absolute equity drops from $85.0\%$ ($n=2$) to $34.7\%$ ($n=9$), the multiway pot size multiplies winnings: $R_2 = 2(0.85) - 1 = +0.70$ (+70%), whereas $R_9 = 9(0.347) - 1 = +2.13$ (+213%).

---

### Category D: Advanced Inference, Martingales, Blinds, and Limits (Units 1, 3, 4)

**Q31: Numerically verify the discrete martingale property of equity across streets.**
> **Answer:** For hand AA at $n=6$, simulated preflop equity was $0.497$. Flop conditional expectations averaged $0.495$ (error $0.002$), turn conditional expectations averaged $0.493$ (error $0.004$), and river showdown averages were $0.497$ (error $0.000$). Maximum expectation drift across all tested hands was $< 0.015$, confirming $E[E_{t+1} \mid \mathcal{F}_t] = E_t$.

**Q32: Why does the Rule of 4 break down quadratically for large out counts?**
> **Answer:** The Rule of 4 estimates flop-to-river probability as $4k\%$. The exact probability is $P = 1 - \frac{\binom{47-k}{2}}{\binom{47}{2}} = \frac{93k - k^2}{2,162}$. For small $k$, $93k / 2,162 \approx 0.043k \approx 4k\%$. For large $k$, the $-k^2 / 2,162$ term reduces the probability. At $k = 15$ outs, exact is $54.12\%$ while Rule of 4 predicts $60.00\%$ ($+5.88$ pp error); at $k = 21$ outs, exact is $69.94\%$ while Rule of 4 gives $84.00\%$ ($+14.06$ pp error).

**Q33: How does opponent tightness shift starting-hand equity?**
> **Answer:** Against an All-Hands field (100%), AA has $85.0\%$ heads-up and 72o has $35.0\%$. Against a Rock field (top 8% hands: pairs 77+, AK, AQ), 72o equity drops to $26.8\%$ heads-up and collapses to $< 3\%$ multiway (over $60\%$ relative drop), whereas AA retains $> 93\%$ of its baseline equity ($79.3\%$).

**Q34: Derive the closed-form blind drain per hand in an $n$-player table with 0.5 SB / 1.0 BB.**
> **Answer:** Each full orbit of $n$ hands, a player posts one small blind ($0.5$ bb) and one big blind ($1.0$ bb), totaling $1.5$ bb per orbit. Assuming stationary position rotation, the deterministic blind cost per hand is $\frac{1.5}{n}$ bb/hand (e.g., $0.75$ bb/hand heads-up, $0.25$ bb/hand at 6-max, $0.167$ bb/hand at 9-max).

**Q35: Why does a tight Rock strategy lose heavily heads-up but win at full ring?**
> **Answer:** A Rock plays only the top 8% of hands, folding 92% of the time. Heads-up, the blind cost is $75.0$ bb/100 hands; folding 92% forfeits blinds rapidly, leading to a net loss of $-28.5$ bb/100 hands. At 9-max, blind drain drops to $16.7$ bb/100 hands, allowing premium hands to recoup blinds and turn profitable.

**Q36: What did your two-way ANOVA test reveal about table selection?**
> **Answer:** A balanced $5 \times 4$ factorial design (5 strategies $\times$ 4 opponent mixes, 10,000 hands per cell) yielded a statistically significant interaction effect ($F = 12.18, p < 0.001, \eta^2 = 0.071$). This proves that no single strategy is globally optimal; profitability depends critically on the table lineup, validating the mathematical necessity of table selection.

**Q37: What is the Brownian ruin approximation and how closely did it match your simulated random walks?**
> **Answer:** For a random walk with positive drift $\mu$ and variance $\sigma^2$, the continuous Brownian approximation to ruin with initial bankroll $B$ is $P(\text{ruin}) \approx \exp\left(-\frac{2\mu B}{\sigma^2}\right)$. Across simulated bankrolls $B \in [20, 200]$ bb, empirical ruin rates tracked the Brownian formula within $1.2$ to $3.1$ percentage points.

**Q38: How many hands are required to separate two close poker strategies at 95% confidence?**
> **Answer:** For win rates differing by $\Delta = 2.0$ bb/100 with standard deviation $\sigma = 75$ bb/100, the sample size for $80\%$ statistical power at $\alpha = 0.05$ is $N \approx 2 \left( \frac{z_{\alpha/2} + z_{\beta}}{\Delta} \right)^2 \sigma^2 \approx 1,160,000$ hands, demonstrating that short-term win rates are dominated by statistical noise.

**Q39: What are the primary academic limitations of this study?**
> **Answer:**
> 1. *Showdown Selection Bias:* In PHH data, folded hole cards are unobserved, overestimating observed hand quality.
> 2. *Limp-or-Fold Showdown Abstraction:* Equal contributions do not model bet sizing, fold equity, or bluffing.
> 3. *Board Card Correlation:* Positive dependence between players breaks independent order-statistic models.
> 4. *Continuous Beta Support:* Equities are modeled continuously, but real rankings are discrete.

**Q40: Why is this project framed strictly around play chips and applied probability?**
> **Answer:** In accordance with course guidelines (THM6242), Texas Hold'em serves as an educational laboratory for conditional expectation, order statistics, sampling distributions, and generating functions. Real-money gambling advice is strictly excluded to maintain academic rigor and focus on mathematical foundations.
