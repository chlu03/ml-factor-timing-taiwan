"""Walk-forward prediction of next month's IC.

At the end of month t the portfolio is formed with factor values at t, so IC[t] is the quantity
to predict. The latest known IC is IC[t-1], because it needs the price at the end of month t.
Every model is refitted each month using only pairs whose target is known by then.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 999
NN_LAYERS = {
    'NN1': (32,),
    'NN2': (32, 16),
    'NN3': (32, 16, 8),
    'NN4': (32, 16, 8, 4),
    'NN5': (32, 16, 8, 4, 2),
}


def model_factories():
    """New, unfitted models (same settings as the original notebooks where possible)."""
    factories = {
        'OLS': lambda: LinearRegression(),
        'RF': lambda: RandomForestRegressor(n_estimators=100, max_depth=3, n_jobs=-1, random_state=SEED),
    }
    for name, layers in NN_LAYERS.items():
        factories[name] = (lambda layers=layers: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='adam',
                         learning_rate_init=0.01, alpha=1e-4, batch_size=32, max_iter=100,
                         early_stopping=True, validation_fraction=0.2, n_iter_no_change=5,
                         random_state=SEED)))
    return factories


def cross_section_features(factor):
    """Original design: the factor values of all stocks in a month are the features (one row per month).
    Missing values are filled with that month's median."""
    filled = factor.apply(lambda col: col.fillna(col.median()))
    return filled.T                                          # rows = months, columns = stocks


def lagged_ic_features(ic, lags=12):
    """Alternative design: the last `lags` monthly ICs of all factors."""
    cols = {f'{name}_lag{k}': ic[name].shift(k - 1) for name in ic.columns for k in range(1, lags + 1)}
    return pd.DataFrame(cols, index=ic.index)               # row s holds IC[s], ..., IC[s-lags+1]


def walk_forward(features, target, test_months, factory, refit_every=1):
    """Predict target[t] for each test month t from features[t-1], training on all (features[s], target[s+1])
    with s+1 <= t-1. The model is refitted every `refit_every` months (every month by default)."""
    months = list(target.index)
    X = features.reindex(months)
    y_next = target.shift(-1)                                 # row s holds target[s+1]
    preds, model = {}, None
    for k, t in enumerate(test_months):
        i = months.index(t)
        x_now = X.iloc[[i - 1]]
        if model is None or k % refit_every == 0:
            train = X.iloc[:i - 1].join(y_next.iloc[:i - 1].rename('_y')).dropna()
            if len(train) >= 24:
                model = factory()
                model.fit(train.drop(columns='_y').values, train['_y'].values)
        if model is None or x_now.isna().any(axis=None):
            preds[t] = np.nan
            continue
        preds[t] = float(model.predict(x_now.values)[0])
    return pd.Series(preds)


# The random forest on 1,848 cross-sectional features is slow, so by default it is refitted once a year.
# Set to {} to refit every model every month.
REFIT_EVERY = {('RF', 'cross_section'): 12}


def predict_all(ic, pairs, test_months, feature_set='cross_section', models=None):
    """Predicted IC for every model and factor. Returns {model name: DataFrame (months x factors)}."""
    factories = model_factories()
    names = models or list(factories)
    out = {}
    for name in names:
        cols = {}
        for factor in ic.columns:
            if feature_set == 'cross_section':
                X = cross_section_features(pairs[factor][0])
            else:
                X = lagged_ic_features(ic)
            cols[factor] = walk_forward(X, ic[factor], test_months, factories[name],
                                        refit_every=REFIT_EVERY.get((name, feature_set), 1))
        out[name] = pd.DataFrame(cols)
    return out


def baseline_predictions(ic, test_months, window=12):
    """Perfect foresight (the true IC) and the average IC of the previous `window` months."""
    return {
        'Perfect foresight': ic.loc[test_months],
        f'Past {window}-month mean': ic.rolling(window).mean().shift(1).loc[test_months],
    }


def oos_r2(pred, ic, test_months, window=12):
    """Out-of-sample R^2 against two benchmarks: predicting zero (as in the original) and the
    past `window`-month mean IC."""
    true = ic.loc[test_months]
    hist = ic.rolling(window).mean().shift(1).loc[test_months]
    err = ((pred - true) ** 2).sum().sum()
    return {'R2 vs zero': 1 - err / (true ** 2).sum().sum(),
            'R2 vs past mean': 1 - err / ((hist - true) ** 2).sum().sum()}
