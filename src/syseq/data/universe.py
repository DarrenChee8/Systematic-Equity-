"""Rebuild historical S&P 500 membership from today's list plus the log of changes."""
import pandas as pd


def reconstruct_membership(
    current: list[str], changes: pd.DataFrame, month_ends: pd.DatetimeIndex
) -> pd.DataFrame:
    """Walk backwards from today's constituents, undoing each change.

    changes: columns ['date', 'added', 'removed']; added/removed may be NaN.
    A stock counts as a member at a month-end if it was in the index at that day's close.
    """
    changes = changes.sort_values("date", ascending=False)
    members = set(current)
    rows = {}
    pending = list(changes.itertuples(index=False))
    i = 0
    for d in sorted(month_ends, reverse=True):
        # undo every change that happened after this month-end
        while i < len(pending) and pending[i].date > d:
            c = pending[i]
            if isinstance(c.added, str):
                members.discard(c.added)
            if isinstance(c.removed, str):
                members.add(c.removed)
            i += 1
        rows[d] = set(members)
    tickers = sorted(set().union(*rows.values()))
    out = pd.DataFrame(False, index=sorted(rows), columns=tickers)
    for d, m in rows.items():
        out.loc[d, list(m)] = True
    return out


def membership_from_spans(spans: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """spans: columns ['ticker', 'start_date', 'end_date']; a blank end_date means still a member.

    A ticker can have several spans (it left the index and came back).
    """
    tickers = sorted(spans["ticker"].unique())
    out = pd.DataFrame(False, index=month_ends, columns=tickers)
    for row in spans.itertuples(index=False):
        inside = month_ends >= row.start_date
        if pd.notna(row.end_date):
            inside &= month_ends < row.end_date
        out.loc[inside, row.ticker] = True
    return out
