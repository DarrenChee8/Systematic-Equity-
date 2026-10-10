"""How similar are two signals? Average cross-sectional rank correlation over time."""
import pandas as pd


def signal_correlation(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ranks = {k: v.rank(axis=1) for k, v in signals.items()}
    names = list(signals)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = ranks[a].corrwith(ranks[b], axis=1).mean()
            out.loc[a, b] = out.loc[b, a] = c
    return out


def series_correlation(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of the signals' monthly payoffs: do they work at the same times?"""
    return pd.DataFrame(series).corr()
