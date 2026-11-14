# Hand-Solved Problems Sheet — Hold'em Lab
## Applied Probability Distributions (THM6242)

> **Instructions for the Student Author (CONTEXT.md Section 2A):**
> Solve each of the six problems below by hand on paper using the given real project numbers.
> Each problem corresponds to one of the course units and reinforces a core mathematical concept from the project.
> When you have completed your calculations, check your numerical results using `check_answer(example_id, value)` in Section 13 of `holdem_lab.ipynb`.
> **Do not write solution steps into the repository** — your handwritten solutions belong in your personal coursework folder and report appendix.

---

## Problem E1: Discrete Order Statistic of Maximum (Tie-Aware)
**Topic:** Unit 4 (Order Statistics) & Unit 2 (Discrete Distributions)

### Setup in Words
In Texas Hold'em showdowns, determining the winning hand requires finding the maximum among multiple players. Consider a simplified discrete hand-ranking model where hands are grouped into three mutually exclusive, ordered categories:
- Category 0: High Card ($x = 0$)
- Category 1: One Pair ($x = 1$)
- Category 2: Two Pair or Better ($x = 2$)

Suppose two players ($n = 2$) each hold a hand drawn independently from this distribution, denoted $X_1, X_2 \sim P(X)$ (i.i.d.). Let $M = \max(X_1, X_2)$ denote the category of the winning hand.

### Given Numbers
The marginal category probabilities from the project's exact 5-card enumeration (`figures/data/V-CARD-COUNTS_exact_probabilities.csv`) are:
- $P(X = 0) = 0.5012$
- $P(X = 1) = 0.4226$
- $P(X = 2) = 0.0763$
*(Note: 0.5012 + 0.4226 + 0.0763 = 1.0000)*

### What to Compute
1. Compute the cumulative distribution values $F(0) = P(X \le 0)$ and $F(1) = P(X \le 1)$.
2. Compute $P(M = 1)$, the probability that the winning category is exactly One Pair, using the discrete order-statistic formula:
   $$P(M = 1) = F(1)^2 - F(0)^2$$
3. Compute the tie probability at One Pair, $P(X_1 = 1 \text{ and } X_2 = 1) = [P(X = 1)]^2$.

---

## Problem E2: Exact Combinatorial Draw Probability & $1/n$ Exchangeability Identity
**Topic:** Unit 2 (Combinatorics) & Unit 1 (Expectation and Exchangeability)

### Setup in Words
- **Part A (Draw Probability by Counting):** A player holds a four-card flush draw on the flop (four spades seen between their two hole cards and the three flop community cards). Two community cards remain to be dealt (the turn and the river). Find the exact probability of hitting at least one additional spade to complete the flush by the river.
- **Part B (Exchangeability Identity):** To verify the fundamental identity $E[\text{equity}] = 1/n$ from first principles, consider a miniature deck of four cards: $A\spadesuit, A\heartsuit, K\spadesuit, K\heartsuit$. Two players ($n = 2$) are each dealt two cards from this shuffled four-card deck. Any hand with two Aces beats two Kings; split pots divide the pot equally ($0.5$ share).

### Given Numbers
- **Part A:**
  - Standard deck size: $N = 52$ cards (13 of each suit).
  - Cards seen on flop: 5 (4 spades, 1 non-spade).
  - Unseen cards remaining: $52 - 5 = 47$ cards.
  - Spades remaining in unseen cards: $13 - 4 = 9$ spades.
  - Non-spades remaining in unseen cards: $47 - 9 = 38$ non-spades.
  - Draw size: 2 cards (turn and river).
- **Part B:**
  - Mini-deck: 4 cards ($A\spadesuit, A\heartsuit, K\spadesuit, K\heartsuit$).
  - Number of players: $n = 2$, each dealt 2 cards.

### What to Compute
1. **(Part A)** Total number of distinct 2-card combinations for turn and river from the 47 unseen cards: $\binom{47}{2}$.
2. **(Part A)** Number of 2-card combinations containing zero spades from the 38 non-spades: $\binom{38}{2}$.
3. **(Part A)** Exact probability of hitting at least one spade by the river:
   $$P(\text{flush hit}) = 1 - \frac{\binom{38}{2}}{\binom{47}{2}}$$
4. **(Part B)** Total number of 2-card hands Player 1 can receive out of 4 cards: $\binom{4}{2}$.
5. **(Part B)** Player 1's equity $E_2 = P(\text{P1 wins}) + 0.5 \times P(\text{tie})$, verifying $E_2 = 1/2$.

---

## Problem E3: Beta Distribution Parameters by Method of Moments
**Topic:** Unit 2 (Beta Distribution) & Unit 1 (Method of Moments)

### Setup in Words
In Section 6 (G2) of Hold'em Lab, the distribution of starting-hand equities across the 169 canonical hand classes is modelled as a continuous random variable on $[0, 1]$ following a $\text{Beta}(\alpha, \ beta)$ distribution. At table size $n = 6$, estimate the shape parameters $\hat{\alpha}$ and $\hat{\beta}$ using the Method of Moments (MoM) from the combo-weighted sample mean and sample variance of equity.

### Given Numbers
From the project's simulated 169-class equity table at $n = 6$ (`figures/data/F-G2-EQUITY-TABLE_table2_beta_gof.csv`):
- Combo-weighted equity mean: $\bar{x} = \frac{1}{6} \approx 0.166667$
- Combo-weighted equity variance: $s^2 = 0.00341606$

### What to Compute
1. The method-of-moments parameter estimate:
   $$\hat{\alpha} = \bar{x} \left( \frac{\bar{x}(1 - \bar{x})}{s^2} - 1 \right)$$
2. The method-of-moments parameter estimate:
   $$\hat{\beta} = (1 - \bar{x}) \left( \frac{\bar{x}(1 - \bar{x})}{s^2} - 1 \right)$$
3. The parameter ratio $\hat{\beta} / \hat{\alpha}$, verifying that it equals $\frac{1 - \bar{x}}{\bar{x}} = n - 1 = 5$.

---

## Problem E4: Chi-Square Goodness-of-Fit Test
**Topic:** Unit 3 (Sampling Distributions: Chi-Square Goodness of Fit)

### Setup in Words
In Section 2 (UCI Poker Hand verification), we test whether observed 5-card hand frequencies in the training split agree with the theoretical multinomial probabilities of standard poker. Consider a sub-table consisting of four intermediate hand categories: Straight, Flush, Full House, and Four of a Kind. We evaluate the Pearson chi-square goodness-of-fit statistic:
$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}$$

### Given Numbers
Observed counts ($O_i$) and expected counts ($E_i$) from the UCI training split (25,010 hands total, `figures/data/V-UCI-EVAL_table1_gof.csv`):
| Category | Observed ($O_i$) | Theoretical Expected ($E_i$) |
|---|---|---|
| Straight | 93 | 98.16 |
| Flush | 54 | 49.15 |
| Full House | 36 | 36.03 |
| Four of a Kind | 6 | 6.00 |

### What to Compute
1. The individual squared standardized residuals $r_i = \frac{(O_i - E_i)^2}{E_i}$ for each of the four categories.
2. The overall Pearson chi-square test statistic $\chi^2 = \sum_{i=1}^4 r_i$.
3. The degrees of freedom $\nu$ for this goodness-of-fit sub-table with $k = 4$ categories and fixed theoretical probabilities.

---

## Problem E5: Geometric Waiting Time via Probability Generating Function
**Topic:** Unit 4 (Power Series / PGFs) & Unit 2 (Geometric Distribution)

### Setup in Words
In Section 10 (G6), we analyze the waiting time (number of deals) until a player receives Pocket Aces (AA). Let $X$ denote the number of independent starting hands dealt to a player up to and including the deal where they first receive Pocket Aces. Here $X$ follows a Geometric distribution on {1, 2, 3, ...} with success parameter $p$.
The Probability Generating Function (PGF) of $X$ is defined as:
$$G(s) = E[s^X] = \sum_{k=1}^\infty s^k (1-p)^{k-1} p = \frac{ps}{1 - (1-p)s}$$

### Given Numbers
- Out of $\binom{52}{2} = 1,326$ possible starting hands, exactly $\binom{4}{2} = 6$ combinations are Pocket Aces ($A\spadesuit A\heartsuit, A\spadesuit A\diamondsuit, A\spadesuit A\clubsuit, A\heartsuit A\diamondsuit, A\heartsuit A\clubsuit, A\diamondsuit A\clubsuit$).
- Probability per deal:
  $$p = \frac{6}{1326} = \frac{1}{221} \approx 0.00452489$$

### What to Compute
1. Differentiate $G(s)$ to obtain $G'(s)$, and evaluate at $s = 1$ to find the mean waiting time $E[X] = G'(1) = \frac{1}{p}$.
2. Differentiate again to find $G''(s)$, and evaluate at $s = 1$ to find the second factorial moment $E[X(X-1)] = G''(1) = \frac{2(1-p)}{p^2}$.
3. Using the factorial moment formula $\text{Var}(X) = G''(1) + G'(1) - [G'(1)]^2$, compute the theoretical variance $\text{Var}(X) = \frac{1-p}{p^2}$ and the standard deviation $\sigma = \sqrt{\text{Var}(X)}$.

---

## Problem E6: Starting-Hand Formula Prediction vs Empirical Simulation
**Topic:** Unit 1 (Mathematical Modelling) & Section 8 (Formula Specification)

### Setup in Words
In Section 7 (G7), Hold'em Lab develops a simple closed-form formula (Model M1) to estimate starting-hand win probability and equity share at an $n$-player table using only heads-up equity $p_2$ and table size $n$.
The formula is:
$$\hat{E}_n = \frac{1}{n} (2 \times p_2)^{\beta_n}$$
where the power-law exponent follows the smooth empirical law:
$$\beta_n = 1.0 + 0.230 \sqrt{n - 2}$$
Calculate the formula equity prediction for Pocket Kings (KK) at a 6-player table ($n = 6$) and evaluate its prediction error relative to the empirical simulation table.

### Given Numbers
From the project's data files (`figures/data/equity_table.csv` and `data/formula_params.json`):
- Hand: Pocket Kings (KK)
- Heads-up equity ($n = 2$): $p_2 = 0.8288$ (or 82.88%)
- Table size: $n = 6$ players
- Smooth law scaling constant: $b = 0.230$
- Empirical equity at $n = 6$: $E_6 = 0.4279$ (or 42.79%)

### What to Compute
1. The exponent $\beta_6 = 1.0 + 0.230 \sqrt{6 - 2}$.
2. The base term $x = 2 \times p_2$.
3. The formula prediction $\hat{E}_6 = \frac{1}{6} x^{\beta_6}$.
4. The absolute error $|\hat{E}_6 - E_6|$ in percentage points (pp).
