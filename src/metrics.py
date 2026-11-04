"""Calculate per-company risk metrics from the cleaned synthetic portfolio data."""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

ESTIMATE_COLUMNS = ["revenue_estimated", "cash_estimated", "headcount_estimated"]

# Thresholds for the traffic-light flags, kept in one place so they're easy to change
FLAG_RULES = {
    "red_runway": 6,
    "amber_runway": 12,
    "max_burn_trend": 20,
    "max_burn_multiple": 3,
}

# Higher number = more serious, so comparing numbers finds the worse flag
FLAG_SEVERITY = {"green": 0, "amber": 1, "red": 2}

# Valuation = ARR x sector multiple. These multiples are ILLUSTRATIVE ASSUMPTIONS, not market
# data, chosen only to show the method. Dilution is ignored: we assume ownership_pct stays at
# its entry level, although later funding rounds would normally shrink it.
SECTOR_MULTIPLES = {
    "SaaS": 6,
    "Fintech": 5,
    "Healthtech": 4,
    "Climate": 4,
    "Marketplace": 3,
}

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
    "flag": {
        "latex": r"\text{flag} = \begin{cases} "
                 r"\text{red} & \text{if } C_t = 0 \text{ or runway} < r_{\text{red}} \\ "
                 r"\text{amber} & \text{if runway} < r_{\text{amber}} \text{ or trend} > \tau "
                 r"\text{ or burn multiple} > m \\ "
                 r"\text{green} & \text{otherwise} \end{cases}",
        # Numbers come from FLAG_RULES so the text can't drift from the thresholds
        "explanation": f"Red if out of cash or runway is under {FLAG_RULES['red_runway']} months; "
                       f"amber if runway is under {FLAG_RULES['amber_runway']} months, burn rose "
                       f"more than {FLAG_RULES['max_burn_trend']}%, or the burn multiple is above "
                       f"{FLAG_RULES['max_burn_multiple']}; otherwise green, and the most serious "
                       "matching rule decides the flag.",
    },
    "valuation": {
        "latex": r"V = 12 \, R_t \times M_{\text{sector}}",
        "explanation": "Annual revenue run rate times an assumed sector multiple, or 0 if the "
                       "company has run out of cash.",
    },
    "moic": {
        "latex": r"\text{MOIC} = \frac{V \times o / 100}{I}",
        "explanation": "What our stake is worth now (valuation times ownership %) divided by "
                       "what we invested (I), so 2.0 means the money doubled on paper.",
    },
    "dpi": {
        "latex": r"\text{DPI} = \frac{D}{P}",
        "explanation": "Cash actually returned to investors (distributions, D) divided by "
                       "cash they paid in (P).",
    },
    "tvpi": {
        "latex": r"\text{TVPI} = \frac{\text{NAV} + D}{P}",
        "explanation": "Total value, unrealised stakes (NAV) plus cash already returned, "
                       "divided by cash paid in.",
    },
}


def has_history(company, months_needed):
    """Return True if the company has at least `months_needed` rows of data."""
    return len(company) >= months_needed


def latest_cash(company):
    """Return the cash balance in the most recent month."""
    return company["cash_eur"].iloc[-1]


def average_burn(company, window=3):
    """Return the mean net burn over the last `window` months."""
    if not has_history(company, window):
        return np.nan
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
    # Comparing with window months earlier needs window + 1 data points
    if not has_history(company, window + 1):
        return np.nan
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
    if not has_history(company, 13):
        return np.nan
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_12m_ago = company["revenue_eur"].iloc[-13]
    return revenue_now / revenue_12m_ago - 1


def burn_trend_pct(company, window=3):
    """Return the % change in average burn: last `window` months vs the `window` before."""
    if not has_history(company, 2 * window):
        return np.nan
    recent_burn = company["net_burn_eur"].iloc[-window:].mean()
    previous_burn = company["net_burn_eur"].iloc[-2 * window:-window].mean()
    if previous_burn == 0:
        return np.nan
    # abs() keeps the sign meaningful when burn is negative: below 0 always means burn fell
    return (recent_burn - previous_burn) / abs(previous_burn) * 100


def burn_and_new_revenue(company, window=3):
    """Return total net burn and new annual revenue over the last `window` months."""
    if not has_history(company, window + 1):
        return np.nan, np.nan
    burn_in_window = company["net_burn_eur"].iloc[-window:].sum()
    revenue_now = company["revenue_eur"].iloc[-1]
    revenue_window_ago = company["revenue_eur"].iloc[-(window + 1)]
    # Monthly revenue x 12 = annual run rate, so this is the new annual revenue added
    new_annual_revenue = (revenue_now - revenue_window_ago) * 12
    return burn_in_window, new_annual_revenue


def burn_multiple(company, window=3):
    """Return cash burned per euro of new annual revenue over the last `window` months."""
    burn_in_window, new_annual_revenue = burn_and_new_revenue(company, window)
    if pd.isna(burn_in_window):
        return np.nan
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


def worse_flag(current_flag, new_flag):
    """Return whichever of the two flags is more serious."""
    if FLAG_SEVERITY[new_flag] > FLAG_SEVERITY[current_flag]:
        return new_flag
    return current_flag


def add_flags(metrics_df, rules=FLAG_RULES):
    """Add a red/amber/green flag and plain-English reasons to each company."""
    df = metrics_df.copy()
    flags = []
    all_reasons = []

    for _, company in df.iterrows():
        flag = "green"
        reasons = []
        runway = company["runway_months"]
        burn_trend = company["burn_trend_pct"]
        burn_mult = company["burn_multiple"]

        # Red: no cash, or very short runway. elif so a runway is reported only once
        if company["latest_cash"] == 0:
            flag = worse_flag(flag, "red")
            reasons.append("out of cash")
        elif runway < rules["red_runway"]:
            flag = worse_flag(flag, "red")
            reasons.append(f"runway {runway:.1f} months (under {rules['red_runway']})")
        elif runway < rules["amber_runway"]:
            flag = worse_flag(flag, "amber")
            reasons.append(f"runway {runway:.1f} months (under {rules['amber_runway']})")

        # Amber: burn rising fast. NaN compares as False, so a missing trend never flags
        if burn_trend > rules["max_burn_trend"]:
            flag = worse_flag(flag, "amber")
            # One decimal, so 20.5% doesn't print as "20% (over 20%)"
            reasons.append(f"burn up {burn_trend:.1f}% on the previous period "
                           f"(over {rules['max_burn_trend']}%)")

        # Amber: burning a lot of cash per euro of new revenue
        if burn_mult > rules["max_burn_multiple"]:
            flag = worse_flag(flag, "amber")
            reasons.append(f"burn multiple {burn_mult:.1f} (over {rules['max_burn_multiple']})")

        # No rule matched: say what's good, so green is explained too
        if not reasons:
            if runway == np.inf:
                reasons.append("profitable")
            else:
                reasons.append(f"runway {runway:.0f} months (over {rules['amber_runway']})")

        # Information only: tells the reader to trust the numbers a little less
        n_estimated = company["estimated_cells"]
        if n_estimated > 0:
            word = "value" if n_estimated == 1 else "values"
            reasons.append(f"includes {n_estimated} estimated {word}")

        flags.append(flag)
        all_reasons.append("; ".join(reasons))

    df["flag"] = flags
    df["reasons"] = all_reasons
    return df


def company_valuations(clean_df, metrics_df, multiples=SECTOR_MULTIPLES):
    """Return one row per company with ARR, valuation, our stake's value and MOIC."""
    # The latest month of each company holds its current revenue and its deal terms
    df = clean_df.sort_values(["company_id", "month"])
    latest = df.groupby("company_id").tail(1)

    columns = ["company_id", "company_name", "sector", "amount_invested_eur", "ownership_pct"]
    valuations = latest[columns].copy()
    valuations["arr"] = latest["revenue_eur"] * 12

    multiple = valuations["sector"].map(multiples)
    valuations["valuation"] = valuations["arr"] * multiple

    # A company with no cash can't keep operating, so we write its value down to zero
    cash = valuations["company_id"].map(metrics_df.set_index("company_id")["latest_cash"])
    out_of_cash = cash == 0
    valuations.loc[out_of_cash, "valuation"] = 0

    valuations["stake_value"] = valuations["valuation"] * valuations["ownership_pct"] / 100
    valuations["moic"] = valuations["stake_value"] / valuations["amount_invested_eur"]

    total_stake_value = valuations["stake_value"].sum()
    valuations["share_of_value"] = valuations["stake_value"] / total_stake_value * 100

    return valuations.reset_index(drop=True)


def fund_summary(valuations_df):
    """Return fund-level totals and ratios as a dictionary."""
    paid_in = valuations_df["amount_invested_eur"].sum()
    nav = valuations_df["stake_value"].sum()
    # No company has been sold or listed yet, so no cash has come back to investors
    distributions = 0

    return {
        "paid_in": float(paid_in),
        "nav": float(nav),
        "distributions": float(distributions),
        "dpi": distributions / paid_in,
        "tvpi": (nav + distributions) / paid_in,
        "written_off": int((valuations_df["valuation"] == 0).sum()),
        "top_2_share_pct": float(valuations_df["share_of_value"].nlargest(2).sum()),
    }


def months_between(earlier, later):
    """Return the number of whole calendar months from `earlier` to `later`."""
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def rolling_metrics(clean_df, window=3, rules=FLAG_RULES):
    """Return metrics and flags as they would have looked at the end of each month."""
    months = sorted(clean_df["month"].unique())
    # The burn trend needs 2 x window months, so earlier months can't be fully assessed
    first_index = 2 * window - 1

    results = []
    for as_of_month in months[first_index:]:
        # Only rows up to as_of_month: on that date, later months hadn't happened yet.
        # Using them would let the flags "know the future" and make the backtest look
        # better than anything we could achieve in real life
        history = clean_df[clean_df["month"] <= as_of_month]

        flagged = add_flags(company_metrics(history, window), rules)
        flagged["as_of_month"] = as_of_month
        results.append(flagged)

    return pd.concat(results, ignore_index=True)


def warning_backtest(rolling_df, clean_df):
    """For each company that ran out of cash, return how many months earlier it was flagged red."""
    rows = []
    out_of_cash = clean_df[clean_df["cash_eur"] == 0]

    for company_id, zero_rows in out_of_cash.groupby("company_id"):
        first_zero_month = zero_rows["month"].min()

        company_flags = rolling_df[rolling_df["company_id"] == company_id]
        red_months = company_flags.loc[company_flags["flag"] == "red", "as_of_month"]
        first_red_month = red_months.min() if len(red_months) > 0 else pd.NaT

        # Only a red flag strictly before the cash ran out counts as a warning
        if pd.notna(first_red_month) and first_red_month < first_zero_month:
            months_of_warning = months_between(first_red_month, first_zero_month)
        else:
            months_of_warning = np.nan

        rows.append({
            "company_id": company_id,
            "first_zero_cash_month": first_zero_month,
            "first_red_month": first_red_month,
            "months_of_warning": months_of_warning,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )
    rolling = rolling_metrics(clean, window=3)
    rolling.to_csv(DATA_DIR / "rolling_metrics.csv", index=False)

    first_month = rolling["as_of_month"].min()
    last_month = rolling["as_of_month"].max()
    print(f"Flags per month, {first_month:%Y-%m} to {last_month:%Y-%m} (G/A/R):")
    for company_id, company_rows in rolling.groupby("company_id"):
        company_rows = company_rows.sort_values("as_of_month")
        letters = "".join(flag[0].upper() for flag in company_rows["flag"])
        print(f"  {company_id}  {letters}")

    backtest = warning_backtest(rolling, clean)
    print("\nWarning backtest (companies that ran out of cash):")
    print(backtest.to_string(index=False))

    warned = backtest["months_of_warning"].notna().sum()
    print(f"\nAverage months of warning: {backtest['months_of_warning'].mean():.1f} "
          f"({warned} of {len(backtest)} companies warned before running out)")
