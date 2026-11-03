"""Calculate per-company risk metrics from the cleaned synthetic portfolio data."""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

ESTIMATE_COLUMNS = ["revenue_estimated", "cash_estimated", "headcount_estimated"]

# Symbols: t = latest month, w = window in months, C = cash, B = net burn, R = revenue.
# B̄_w(t) is the average burn over the w months ending at month t.
FORMULAS = {
    "runway_months": {
        "latex": r"\text{runway} = \frac{C_t}{\bar{B}_w(t)}",
        "explanation": "How many months the current cash lasts at the recent average burn "
                       "(0 if cash is already gone, infinite if the company isn't burning cash).",
    },
    "avg_burn": {
        "latex": r"\bar{B}_w(t) = \frac{1}{w} \sum_{i=0}^{w-1} B_{t-i}",
        "explanation": "The average monthly net burn over the last w months.",
    },
    "mom_growth": {
        "latex": r"g_w = \left( \frac{R_t}{R_{t-w}} \right)^{1/w} - 1",
        "explanation": "The steady monthly growth rate that would turn revenue w months ago "
                       "into revenue now.",
    },
    "annualised_growth": {
        "latex": r"g_{\text{year}} = (1 + g_w)^{12} - 1",
        "explanation": "The yearly growth you get if the monthly rate compounds for 12 months.",
    },
    "yoy_growth": {
        "latex": r"\text{YoY} = \frac{R_t}{R_{t-12}} - 1",
        "explanation": "How much revenue changed compared with the same month one year earlier.",
    },
    "burn_trend_pct": {
        "latex": r"\text{trend} = \frac{\bar{B}_w(t) - \bar{B}_w(t-w)}"
                 r"{\left| \bar{B}_w(t-w) \right|} \times 100",
        "explanation": "The % change in average burn between the last w months and the w months "
                       "before, where a negative value always means burn fell.",
    },
    "burn_multiple": {
        "latex": r"\text{burn multiple} = \frac{\sum_{i=0}^{w-1} B_{t-i}}{12 \, (R_t - R_{t-w})}",
        "explanation": "Euros of cash burned for each euro of new annual revenue added over the "
                       "last w months (blank and noted 'profitable' if the company burned no net "
                       "cash, or 'not growing' if revenue didn't grow).",
    },
}


def latest_cash(company):
    """Return the cash balance in the most recent month."""
    return company["cash_eur"].iloc[-1]


def average_burn(company, window=3):
    """Return the mean net burn over the last `window` months."""
    return company["net_burn_eur"].iloc[-window:].mean()


def runway_months(cash, avg_burn):
    """Return how many months the cash lasts at the current burn rate."""
    if cash == 0:
        return 0.0
    # Zero or negative burn means cash isn't shrinking, so it never runs out
    if avg_burn <= 0:
        return np.inf
    return cash / avg_burn


def monthly_growth(company, window=3):
    """Return the average monthly revenue growth over the last `window` months."""
    revenue_now = company["revenue_eur"].iloc[-1]
    # window months before the latest month is position -(window + 1)
    revenue_window_ago = company["revenue_eur"].iloc[-(window + 1)]
    # Geometric mean: the steady monthly rate that turns revenue_window_ago into revenue_now
    return (revenue_now / revenue_window_ago) ** (1 / window) - 1


def annualised_growth(mom_growth):
    """Return the yearly growth implied by compounding a monthly rate 12 times."""
    return (1 + mom_growth) ** 12 - 1


def yoy_growth(company):
    """Return revenue growth compared with the same month a year earlier."""
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_12m_ago = company["revenue_eur"].iloc[-13]
    return revenue_now / revenue_12m_ago - 1


def burn_trend_pct(company, window=3):
    """Return the % change in average burn: last `window` months vs the `window` before."""
    recent_burn = company["net_burn_eur"].iloc[-window:].mean()
    previous_burn = company["net_burn_eur"].iloc[-2 * window:-window].mean()
    if previous_burn == 0:
        return np.nan
    # abs() keeps the sign meaningful when burn is negative: below 0 always means burn fell
    return (recent_burn - previous_burn) / abs(previous_burn) * 100


def burn_and_new_revenue(company, window=3):
    """Return total net burn and new annual revenue over the last `window` months."""
    burn_in_window = company["net_burn_eur"].iloc[-window:].sum()
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_window_ago = company["revenue_eur"].iloc[-(window + 1)]
    # Monthly revenue x 12 = annual run rate, so this is the new annual revenue added
    new_annual_revenue = (revenue_now - revenue_window_ago) * 12
    return burn_in_window, new_annual_revenue


def burn_multiple(company, window=3):
    """Return cash burned per euro of new annual revenue over the last `window` months."""
    burn_in_window, new_annual_revenue = burn_and_new_revenue(company, window)
    # The ratio only means something when cash is burned to buy growth
    if burn_in_window <= 0 or new_annual_revenue <= 0:
        return np.nan
    return burn_in_window / new_annual_revenue


def burn_multiple_note(company, window=3):
    """Return why burn_multiple is blank: 'profitable', 'not growing', or ''."""
    burn_in_window, new_annual_revenue = burn_and_new_revenue(company, window)
    # Checked first: if no cash was burned, efficiency of burn is not the question
    if burn_in_window <= 0:
        return "profitable"
    if new_annual_revenue <= 0:
        return "not growing"
    return ""


def estimated_cells(company):
    """Return how many revenue, cash and headcount values were interpolated."""
    return int(company[ESTIMATE_COLUMNS].sum().sum())


def company_metrics(clean_df, window=3):
    """Return one row of risk metrics per company, using `window` recent months."""
    if not 1 <= window <= 12:
        raise ValueError(
            f"window must be between 1 and 12 months, got {window}. "
            "At least 1 month is needed to average anything, and the burn trend compares "
            "the last window months with the window before, so it needs 2 x window months "
            "of history: 12 is the most a 24-month dataset can support."
        )

    # Every metric reads "last" and "N months earlier" by position, so order is essential
    df = clean_df.sort_values(["company_id", "month"])

    rows = []
    for company_id, company in df.groupby("company_id"):
        cash = latest_cash(company)
        avg_burn = average_burn(company, window)
        mom_growth = monthly_growth(company, window)

        rows.append({
            "company_id": company_id,
            "company_name": company["company_name"].iloc[0],
            "sector": company["sector"].iloc[0],
            "latest_cash": cash,
            "avg_burn": avg_burn,
            "runway_months": runway_months(cash, avg_burn),
            "mom_growth": mom_growth,
            "annualised_growth": annualised_growth(mom_growth),
            "yoy_growth": yoy_growth(company),
            "burn_trend_pct": burn_trend_pct(company, window),
            "burn_multiple": burn_multiple(company, window),
            "burn_multiple_note": burn_multiple_note(company, window),
            "estimated_cells": estimated_cells(company),
        })

    metrics = pd.DataFrame(rows)
    return metrics


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )
    rounded_columns = [
        "runway_months",
        "mom_growth",
        "annualised_growth",
        "yoy_growth",
        "burn_multiple",
    ]

    for window in [3, 6]:
        metrics = company_metrics(clean, window=window)
        display = metrics.copy()
        display[rounded_columns] = display[rounded_columns].round(2)
        print(f"\nWindow = {window} months:")
        print(display.to_string(index=False))
