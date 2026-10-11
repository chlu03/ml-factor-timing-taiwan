"""Factor-timing long-short backtest and performance measures."""
import numpy as np
import pandas as pd

# Number of groups used in the original project (10 groups = deciles, 1848 groups = one stock per side)
GROUPS = [10, 20, 50, 100, 184, 205, 231, 264, 308, 370, 462, 616, 924, 1848]


def choose_factor(pred_row):
    """The factor with the largest absolute predicted IC, and the sign of that IC."""
    factor = pred_row.abs().idxmax()
    return factor, np.sign(pred_row[factor])


def long_short(factor, ret, n_groups, sign, simple=True):
    """Long the top group and short the bottom group (reversed when the IC is negative)."""
    ok = factor.notna() & ret.notna()
    f, r = factor[ok], ret[ok]
    if simple:
        r = np.exp(r) - 1                                    # log return -> simple return
    k = max(len(f) // n_groups, 1)
    order = f.sort_values(kind='mergesort').index
    spread = r[order[-k:]].mean() - r[order[:k]].mean()
    return sign * spread


def backtest(pred, pairs, groups=GROUPS, simple=True):
    """Monthly long-short returns for each group count, plus the chosen factor."""
    rows, chosen = {}, {}
    for month, pred_row in pred.dropna(how='all').iterrows():
        factor, sign = choose_factor(pred_row)
        f, r = pairs[factor][0][month], pairs[factor][1][month]
        if r.notna().sum() == 0:
            continue
        rows[month] = {g: long_short(f, r, g, sign, simple) for g in groups}
        chosen[month] = factor
    return pd.DataFrame(rows).T, pd.Series(chosen)


def performance(returns):
    """Annualized return, volatility, Sharpe ratio, t-statistic, maximum drawdown, and hit rate."""
    r = pd.Series(returns).dropna()
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        'Annual return': r.mean() * 12,
        'Annual volatility': r.std() * np.sqrt(12),
        'Sharpe ratio': r.mean() / r.std() * np.sqrt(12),
        't-stat': r.mean() / r.std() * np.sqrt(len(r)),
        'Max drawdown': drawdown.min(),
        'Hit rate': (r > 0).mean(),
    }
