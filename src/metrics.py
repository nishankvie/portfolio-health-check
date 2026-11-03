"""Calculate per-company risk metrics from the cleaned synthetic portfolio data."""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

ESTIMATE_COLUMNS = ["revenue_estimated", "cash_estimated", "headcount_estimated"]


def latest_cash(company):
    """Return the cash balance in the most recent month."""
    return company["cash_eur"].iloc[-1]


def avg_burn_3m(company):
    """Return the mean net burn over the last 3 months."""
    return company["net_burn_eur"].iloc[-3:].mean()


def runway_months(cash, avg_burn):
    """Return how many months the cash lasts at the current burn rate."""
    if cash == 0:
        return 0.0
    # Zero or negative burn means cash isn't shrinking, so it never runs out
    if avg_burn <= 0:
        return np.inf
    return cash / avg_burn


def mom_growth_3m(company):
    """Return the average monthly revenue growth over the last 3 months."""
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_3m_ago = company["revenue_eur"].iloc[-4]
    # Geometric mean: the steady monthly rate that turns revenue_3m_ago into revenue_now
    return (revenue_now / revenue_3m_ago) ** (1 / 3) - 1


def annualised_growth(mom_growth):
    """Return the yearly growth implied by compounding a monthly rate 12 times."""
    return (1 + mom_growth) ** 12 - 1


def yoy_growth(company):
    """Return revenue growth compared with the same month a year earlier."""
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_12m_ago = company["revenue_eur"].iloc[-13]
    return revenue_now / revenue_12m_ago - 1


def burn_trend_pct(company):
    """Return the % change in average burn: last 3 months vs the 3 months before."""
    recent_burn = company["net_burn_eur"].iloc[-3:].mean()
    previous_burn = company["net_burn_eur"].iloc[-6:-3].mean()
    if previous_burn == 0:
        return np.nan
    # abs() keeps the sign meaningful when burn is negative: below 0 always means burn fell
    return (recent_burn - previous_burn) / abs(previous_burn) * 100


def burn_multiple(company):
    """Return cash burned per euro of new annual revenue over the last 3 months."""
    burn_last_3m = company["net_burn_eur"].iloc[-3:].sum()
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_3m_ago = company["revenue_eur"].iloc[-4]
    # Monthly revenue x 12 = annual run rate, so this is the new annual revenue added
    new_annual_revenue = (revenue_now - revenue_3m_ago) * 12
    if new_annual_revenue <= 0:
        return np.nan
    return burn_last_3m / new_annual_revenue


def estimated_cells(company):
    """Return how many revenue, cash and headcount values were interpolated."""
    return int(company[ESTIMATE_COLUMNS].sum().sum())


def company_metrics(clean_df):
    """Return one row of risk metrics per company."""
    # Every metric reads "last" and "N months earlier" by position, so order is essential
    df = clean_df.sort_values(["company_id", "month"])

    rows = []
    for company_id, company in df.groupby("company_id"):
        cash = latest_cash(company)
        avg_burn = avg_burn_3m(company)
        mom_growth = mom_growth_3m(company)

        rows.append({
            "company_id": company_id,
            "company_name": company["company_name"].iloc[0],
            "sector": company["sector"].iloc[0],
            "latest_cash": cash,
            "avg_burn_3m": avg_burn,
            "runway_months": runway_months(cash, avg_burn),
            "mom_growth_3m": mom_growth,
            "annualised_growth": annualised_growth(mom_growth),
            "yoy_growth": yoy_growth(company),
            "burn_trend_pct": burn_trend_pct(company),
            "burn_multiple": burn_multiple(company),
            "estimated_cells": estimated_cells(company),
        })

    metrics = pd.DataFrame(rows)
    return metrics


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )
    metrics = company_metrics(clean)

    rounded_columns = [
        "runway_months",
        "mom_growth_3m",
        "annualised_growth",
        "yoy_growth",
        "burn_multiple",
    ]
    display = metrics.copy()
    display[rounded_columns] = display[rounded_columns].round(2)
    print(display.to_string(index=False))
