"""Factor and return panels (rows = stocks, columns = months)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
FACTORS = ['bm', 'size', 'mom']


def load_panels(data_dir=DATA):
    panels = {name: pd.read_csv(Path(data_dir) / f'{name}.csv', index_col='code', dtype={'code': str})
              for name in ['close', 'market_cap', 'pbr']}
    twii = pd.read_csv(Path(data_dir) / 'twii.csv', index_col='month')['twii_return']
    return panels, twii


def positive(df):
    return df.where(df > 0)


def compute_factors(panels):
    """Next-month log return and the three factors, all aligned to the same month t.

    ret[t]  = ln(P[t+1] / P[t])     return earned after forming the portfolio at the end of month t
    bm[t]   = 1 / PBR[t]            book-to-market
    size[t] = ln(market cap[t])
    mom[t]  = ln(P[t-1] / P[t-12])  past 11-month return, skipping the most recent month
    """
    close = positive(panels['close']).copy()
    log_price = np.log(close).copy()
    ret = log_price.shift(-1, axis=1) - log_price
    factors = {
        'bm': 1 / positive(panels['pbr']),
        'size': np.log(positive(panels['market_cap'])),
        'mom': log_price.shift(1, axis=1) - log_price.shift(12, axis=1),
    }
    return {name: (factor, ret) for name, factor in factors.items()}


def original_factors(panels):
    """Reproduce the original spreadsheet logic, including its two problems.

    1. Missing values: whenever a factor value or the next-month return is missing, both are
       replaced by the cross-sectional median of that month.
    2. Momentum alignment: the momentum sheet starts 12 months later than the return sheet, and
       the two were matched by column position. Momentum at month t was therefore paired with
       the return from month t-12 to t-11, which lies inside the momentum window itself.
    """
    pairs = compute_factors(panels)
    factor, ret = pairs['mom']
    pairs['mom'] = (factor, ret.shift(12, axis=1))   # column t now holds the return of month t-12
    return {name: _median_impute(f, r) for name, (f, r) in pairs.items()}


def _median_impute(factor, ret):
    missing = factor.isna() | ret.isna()
    factor = factor.mask(missing, factor.median(), axis=1)
    ret = ret.mask(missing, ret.median(), axis=1)
    return factor, ret
