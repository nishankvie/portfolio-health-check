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


if __name__ == "__main__":
    print(generate_companies())
