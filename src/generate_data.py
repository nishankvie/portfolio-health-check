"""Generate a synthetic VC portfolio dataset. All companies are fictional."""

import numpy as np
import pandas as pd

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
        costs = revenue * rng.uniform(1.2, 2.5)
        cost_growth = rng.uniform(0.01, 0.03)
        cash = rng.uniform(1_000_000, 5_000_000)

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


if __name__ == "__main__":
    companies = generate_companies()
    monthly = generate_monthly(companies)

    print(f"Monthly rows: {len(monthly)}")
    print(monthly.head())

    last_month = monthly["month"].max()
    final_month = monthly[monthly["month"] == last_month]
    final_month = final_month.merge(companies[["company_id", "profile"]], on="company_id")
    print(f"\nCash in {last_month:%Y-%m}:")
    print(final_month[["company_id", "profile", "cash_eur"]].to_string(index=False))
