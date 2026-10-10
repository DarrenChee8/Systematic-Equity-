"""Cross-sectional transforms: each row (date) is treated independently."""
import pandas as pd


def winsorise(x: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = x.quantile(lower, axis=1)
    hi = x.quantile(upper, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def sector_zscore(x: pd.DataFrame, sectors: pd.Series, min_names: int = 5) -> pd.DataFrame:
    """Winsorise and z-score within each sector, so stocks are compared with their peers.

    Tickers without a sector go into an 'Unknown' group. Sector-dates with fewer than
    min_names stocks are left blank.
    """
    groups = sectors.reindex(x.columns).fillna("Unknown")
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for _, cols in groups.groupby(groups).groups.items():
        sub = x[cols]
        z = zscore(winsorise(sub))
        out[cols] = z.where(sub.notna().sum(axis=1) >= min_names, axis=0)
    return out


def composite(parts: list[pd.DataFrame], sectors: pd.Series, min_parts: int = 2) -> pd.DataFrame:
    """Average several z-scored characteristics into one theme score, then re-standardise."""
    stack = pd.concat(parts, keys=range(len(parts)))
    mean = stack.groupby(level=1).mean()
    count = stack.notna().groupby(level=1).sum()
    return sector_zscore(mean.where(count >= min(min_parts, len(parts))), sectors)
