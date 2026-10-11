"""Monthly information coefficients (IC)."""
import numpy as np
import pandas as pd


def monthly_ic(factor, ret, min_stocks=30):
    """Pearson correlation between the factor at month t and the return from t to t+1."""
    values = {}
    for month in factor.columns:
        f, r = factor[month], ret[month]
        ok = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        values[month] = np.corrcoef(f[ok], r[ok])[0, 1] if ok.sum() >= min_stocks else np.nan
    return pd.Series(values)


def ic_table(pairs):
    """DataFrame of monthly IC, one column per factor."""
    return pd.DataFrame({name: monthly_ic(f, r) for name, (f, r) in pairs.items()})
