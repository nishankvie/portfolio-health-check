"""Generate a synthetic VC portfolio dataset. All companies are fictional."""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

# Made-up words so no name can be mistaken for a real company
FICTIONAL_NAMES = [
    "Zorblex Pay",
    "Quandrift Health",
    "Flimmora Cloud",
    "Vexalune Energy",
    "Trundlepop Market",
    "Gribbleton Ledger",
    "Mivvora Care",
    "Plonkstack Software",
    "Sunquirrel Climate",
    "Bazzlewick Bazaar",
    "Krindle Finance",
    "Nobbleworth Bio",
    "Snorvik Analytics",
    "Wumblegrid Power",
    "Fizzlethorpe Goods",
]

SECTORS = ["Fintech", "Healthtech", "SaaS", "Climate", "Marketplace"]
PROFILES = ["strong", "steady", "struggling"]
PROFILE_PROBABILITIES = [0.3, 0.4, 0.3]

# Monthly revenue growth range (low, high) for each profile
GROWTH_RANGES = {
    "strong": (0.05, 0.08),
    "steady": (0.01, 0.03),
    "struggling": (-0.02, 0.01),
}

N_MONTHS = 24
FIRST_MONTH = "2024-11-01"
COST_PER_EMPLOYEE_EUR = 6000


def generate_companies(n_companies=15, seed=42):
    """Return a DataFrame of fictional startups with investment details."""
    if n_companies > len(FICTIONAL_NAMES):
        raise ValueError(f"Only {len(FICTIONAL_NAMES)} fictional names available")

    rng = np.random.default_rng(seed)

    company_ids = [f"C{i:02d}" for i in range(1, n_companies + 1)]
    company_names = FICTIONAL_NAMES[:n_companies]

    sectors = rng.choice(SECTORS, size=n_companies)

    possible_dates = pd.date_range("2022-01-01", "2024-06-01", freq="MS")
    investment_dates = rng.choice(possible_dates, size=n_companies)

    raw_amounts = rng.uniform(250_000, 3_000_000, size=n_companies)
    amounts_invested = (np.round(raw_amounts / 50_000) * 50_000).astype(int)

    ownership_pcts = np.round(rng.uniform(5, 25, size=n_companies), 1)

    # Post-money valuation implied by the deal: what we paid / the share we got
    entry_valuations = np.round(amounts_invested / (ownership_pcts / 100)).astype(int)

    profiles = rng.choice(PROFILES, size=n_companies, p=PROFILE_PROBABILITIES)

    companies = pd.DataFrame({
        "company_id": company_ids,
        "company_name": company_names,
        "sector": sectors,
        "investment_date": investment_dates,
        "amount_invested_eur": amounts_invested,
        "ownership_pct": ownership_pcts,
        "entry_valuation_eur": entry_valuations,
        "profile": profiles,
    })
    return companies


def generate_monthly(companies, seed=42):
    """Return a DataFrame with one row per company per month of financials."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(FIRST_MONTH, periods=N_MONTHS, freq="MS")

    rows = []
    for _, company in companies.iterrows():
        # Drawn once per company: these describe the company's underlying trend
        revenue = rng.uniform(20_000, 200_000)
        low_growth, high_growth = GROWTH_RANGES[company["profile"]]
        revenue_growth = rng.uniform(low_growth, high_growth)
        costs = revenue * rng.uniform(1.1, 2)
        cost_growth = rng.uniform(0.01, 0.03)
        cash = rng.uniform(2_000_000, 6_000_000)

        for month_number, month in enumerate(months):
            # Month 0 keeps the starting values; later months compound from the previous one
            if month_number > 0:
                noise = rng.normal(0, 0.05)
                revenue = revenue * (1 + revenue_growth + noise)
                costs = costs * (1 + cost_growth)

            net_burn = costs - revenue
            # A company can't hold negative cash; at 0 it has run out
            cash = max(cash - net_burn, 0)

            rows.append({
                "company_id": company["company_id"],
                "month": month,
                "revenue_eur": round(revenue),
                "costs_eur": round(costs),
                "net_burn_eur": round(net_burn),
                "cash_eur": round(cash),
                "headcount": round(costs / COST_PER_EMPLOYEE_EUR),
            })

    monthly = pd.DataFrame(rows)
    return monthly


def make_messy(companies, monthly, seed=42):
    """Save a clean truth table and a deliberately messy copy; return the messy one."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(exist_ok=True)

    # (1) One wide table: each monthly row gets its company's details
    merged = monthly.merge(companies, on="company_id")

    # (2) The answer key. profile is dropped because a real dataset wouldn't contain it
    truth = merged.drop(columns="profile")
    truth.to_csv(DATA_DIR / "truth_portfolio.csv", index=False)

    # (3) Work on a copy so the errors never leak into truth
    messy = truth.copy()

    # Pick 93 different rows up front and give each error its own slice,
    # so no row gets two errors and every error can be counted later
    picked = rng.choice(messy.index, size=93, replace=False)
    name_rows = picked[0:15]
    text_revenue_rows = picked[15:35]
    nan_revenue_rows = picked[35:45]
    nan_headcount_rows = picked[45:50]
    negative_headcount_rows = picked[50:53]
    x100_revenue_rows = picked[53:55]
    date_format_rows = picked[55:75]
    duplicate_rows = picked[75:85]
    delete_rows = picked[85:93]

    # Columns must be able to hold the new kinds of value before we plant them:
    # revenue will contain text, headcount will contain NaN (which is a float)
    messy["revenue_eur"] = messy["revenue_eur"].astype(object)
    messy["headcount"] = messy["headcount"].astype(float)
    messy["month"] = messy["month"].dt.strftime("%Y-%m-%d")

    # Error: company name variants (lowercase, extra spaces, or " GmbH")
    for row in name_rows:
        name = messy.loc[row, "company_name"]
        variant = rng.choice(["lowercase", "spaces", "gmbh"])
        if variant == "lowercase":
            messy.loc[row, "company_name"] = name.lower()
        elif variant == "spaces":
            messy.loc[row, "company_name"] = "  " + name.replace(" ", "  ") + " "
        else:
            messy.loc[row, "company_name"] = name + " GmbH"

    # Error: revenue typed in Austrian format, e.g. 12345 -> "€12.345"
    for row in text_revenue_rows:
        value = messy.loc[row, "revenue_eur"]
        messy.loc[row, "revenue_eur"] = f"€{value:,}".replace(",", ".")

    # Error: missing values
    messy.loc[nan_revenue_rows, "revenue_eur"] = np.nan
    messy.loc[nan_headcount_rows, "headcount"] = np.nan

    # Error: negative headcounts (a sign typo)
    messy.loc[negative_headcount_rows, "headcount"] = -messy.loc[negative_headcount_rows, "headcount"]

    # Error: revenue entered in cents instead of euros
    messy.loc[x100_revenue_rows, "revenue_eur"] = messy.loc[x100_revenue_rows, "revenue_eur"] * 100

    # Error: some months written as dd.mm.yyyy instead of yyyy-mm-dd
    messy.loc[date_format_rows, "month"] = truth.loc[date_format_rows, "month"].dt.strftime("%d.%m.%Y")

    # Error: rows deleted entirely
    messy = messy.drop(index=delete_rows)

    # Error: duplicated rows (exact copies appended to the end)
    duplicates = messy.loc[duplicate_rows]
    messy = pd.concat([messy, duplicates])

    # (4) Shuffle so the errors aren't sitting in a neat block at the bottom
    messy = messy.sample(frac=1, random_state=rng)
    messy = messy.reset_index(drop=True)

    # (5) Save the file the cleaning step will start from
    messy.to_csv(DATA_DIR / "raw_portfolio.csv", index=False)
    return messy


if __name__ == "__main__":
    companies = generate_companies()
    monthly = generate_monthly(companies)
    messy = make_messy(companies, monthly)

    print(f"Rows in truth: {len(monthly)}")
    print(f"Rows in raw:   {len(messy)}")
    print(f"Duplicate rows in raw: {messy.duplicated().sum()}")

    print("\nNaNs per column in raw:")
    print(messy.isna().sum())

    print("\n5 example messy rows:")
    print(messy.head().to_string())
