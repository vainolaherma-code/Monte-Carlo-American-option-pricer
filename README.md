# Monte-Carlo-American-option-pricer
# American Option Pricer: Longstaff-Schwartz Monte Carlo

A Python implementation of the Longstaff-Schwartz least-squares Monte Carlo (LSM) algorithm for pricing American calls and puts, validated against two independent methods.

## Overview

American options can be exercised at any time, so their price depends on finding the optimal exercise strategy. LSM estimates it by simulating asset paths, then working backwards in time and regressing the discounted future cash flows on the current stock price to estimate the value of holding the option. The option is exercised whenever its immediate payoff exceeds that estimate.

**Model:** risk-neutral geometric Brownian motion (Black-Scholes dynamics), no dividends.
**Method:** plain Monte Carlo with exact log-price simulation and a cubic polynomial regression basis. No variance reduction is used.

## Validation

The LSM output is checked against two independent methods rather than trusted on its own:

1. **Black-Scholes (European).** With no dividends, early exercise is never optimal for an American call, so its price must equal the European price.
2. **Cox-Ross-Rubinstein binomial tree (500 steps).** This prices American options directly with no simulation noise, and serves as ground truth for the American put, which has no closed form.

## Results

Parameters: S0 = 100, K = 100, r = 5%, σ = 20%, T = 1 year, 100 time steps, 100,000 paths, seed 42.

| Option | LSM (Monte Carlo) | Std. error | Binomial tree | European Black-Scholes | LSM vs. binomial |
|--------|-------------------|------------|---------------|------------------------|------------------|
| Call   | 10.4741           | 0.0466     | 10.4466       | 10.4506                | 0.59 std. errors |
| Put    | 6.0492            | 0.0225     | 6.0888        | 5.5735                 | 1.76 std. errors |

- The American call matches the European Black-Scholes price within Monte Carlo error, as theory requires.
- The American put sits well above its European counterpart (6.05 vs. 5.57), which is the value of early exercise, and agrees with the binomial tree within two standard errors.
- LSM is known to be biased slightly low, since the regression gives a suboptimal but feasible exercise policy. A richer basis or a finer time grid reduces this bias.

## Usage

```bash
pip install numpy matplotlib
python american_lsm.py
```

This prints the validation table above and saves `american_lsm_diagnostics.png` to the working directory. The pricing functions can also be imported and used on their own:

```python
from american_lsm import simulate_gbm_paths, price_american_lsm

paths = simulate_gbm_paths(S0=100, r=0.05, sigma=0.2, T=1.0,
                           n_steps=100, n_paths=100_000, seed=42)
price, std_error = price_american_lsm(paths, K=100, r=0.05, T=1.0,
                                      option_type="put")
```

## Files

- `american_lsm.py`: simulation, LSM pricer, Black-Scholes and binomial cross-checks, and diagnostic plots

## Reference

Longstaff, F. A. and Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach.* The Review of Financial Studies, 14(1), 113-147.
