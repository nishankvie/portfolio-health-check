"""Monte Carlo simulation of each company's future cash, using the synthetic portfolio data."""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

# growth_factor scales the usual revenue growth; cost_shock raises costs once at the start.
# Jumps are sudden one-off events, like losing a big client: in any month, with probability
# jump_prob, revenue drops by jump_size on top of normal growth, and stays lower afterwards
BASE_SCENARIO = {
    "growth_factor": 1.0,
    "cost_shock": 0.0,
    "jump_prob": 0.0,
    "jump_size": 0.0,
}

# Each scenario starts from the base settings and overrides only what it changes
SCENARIOS = {
    "base": BASE_SCENARIO,
    "slowdown": {**BASE_SCENARIO, "growth_factor": 0.5},
    "cost_spike": {**BASE_SCENARIO, "cost_shock": 0.2},
    "downturn": {**BASE_SCENARIO, "growth_factor": 0.5, "jump_prob": 0.05, "jump_size": -0.2},
}


def estimate_params(company_df, lookback=12):
    """Return mean and std of monthly % changes in revenue and costs over recent months."""
    # lookback changes need lookback + 1 months: each change compares a month with the one before
    recent = company_df.sort_values("month").tail(lookback + 1)
    changes = recent[["revenue_eur", "costs_eur"]].pct_change().dropna()

    return {
        "revenue_mean": changes["revenue_eur"].mean(),
        "revenue_std": changes["revenue_eur"].std(),
        "costs_mean": changes["costs_eur"].mean(),
        "costs_std": changes["costs_eur"].std(),
    }


def simulate_company(company_df, n_sims=1000, horizon=18, seed=42, scenario=SCENARIOS["base"]):
    """Simulate future cash paths for one company under a scenario; return risk statistics."""
    company_df = company_df.sort_values("month")
    latest = company_df.iloc[-1]
    latest_revenue = latest["revenue_eur"]
    latest_costs = latest["costs_eur"] * (1 + scenario["cost_shock"])
    latest_cash = latest["cash_eur"]

    # Already out of cash: nothing uncertain to simulate
    if latest_cash <= 0:
        return {
            "prob_out_of_cash": 1.0,
            "median_months_to_zero": 0.0,
            "cash_p10": 0.0,
            "cash_p50": 0.0,
            "cash_p90": 0.0,
            "ran_out_sims": np.ones(n_sims, dtype=bool),
        }

    rng = np.random.default_rng(seed)
    params = estimate_params(company_df)
    revenue_mean = params["revenue_mean"] * scenario["growth_factor"]

    # One row per simulated future, one column per month ahead
    revenue_growth = rng.normal(revenue_mean, params["revenue_std"], size=(n_sims, horizon))
    costs_growth = rng.normal(params["costs_mean"], params["costs_std"], size=(n_sims, horizon))

    # Jumps: a uniform number in [0, 1) is below jump_prob with probability jump_prob.
    # Drawn after the growth rates, so the base scenario's numbers are unchanged
    jumps = rng.random(size=(n_sims, horizon)) < scenario["jump_prob"]
    # True counts as 1 and False as 0, so only jump months get jump_size added
    revenue_growth = revenue_growth + jumps * scenario["jump_size"]

    # cumprod along each row compounds the monthly growth factors: month k = start x product of k factors
    revenue_paths = latest_revenue * np.cumprod(1 + revenue_growth, axis=1)
    costs_paths = latest_costs * np.cumprod(1 + costs_growth, axis=1)

    burn_paths = costs_paths - revenue_paths
    # cumsum along each row adds up the burn so far: cash at month k = start - total burn to k
    cash_paths = latest_cash - np.cumsum(burn_paths, axis=1)

    ran_out = cash_paths <= 0
    hit_zero = ran_out.any(axis=1)
    prob_out_of_cash = hit_zero.mean()

    if hit_zero.any():
        # argmax finds the first True in each row; +1 turns position 0 into "month 1"
        months_to_zero = ran_out[hit_zero].argmax(axis=1) + 1
        median_months_to_zero = float(np.median(months_to_zero))
    else:
        median_months_to_zero = np.nan

    final_cash = np.maximum(cash_paths[:, -1], 0)
    cash_p10, cash_p50, cash_p90 = np.percentile(final_cash, [10, 50, 90])

    return {
        "prob_out_of_cash": float(prob_out_of_cash),
        "median_months_to_zero": median_months_to_zero,
        "cash_p10": float(cash_p10),
        "cash_p50": float(cash_p50),
        "cash_p90": float(cash_p90),
        # One True/False per simulation, so failures can be counted across companies
        "ran_out_sims": hit_zero,
    }


def simulate_portfolio(clean_df, n_sims=1000, horizon=12, seed=42):
    """Run simulate_company for every company; return one row per company."""
    # Companies are simulated independently: one company's good or bad luck has no effect on
    # another's. In reality a recession would hit many at once, so portfolio-wide risk is
    # probably understated. Growth is also drawn from a normal distribution, which ignores
    # jumps (losing a big customer overnight) and fat tails (extreme months happening more
    # often than a bell curve predicts), so real startup risk is likely underestimated too.
    # Each company gets its own seed (seed + position), because the same seed would give
    # every company the same random shocks, which is perfect correlation
    rows = []
    for position, (company_id, company) in enumerate(clean_df.groupby("company_id")):
        result = simulate_company(company, n_sims, horizon, seed + position)
        # The per-simulation array doesn't fit in a one-row-per-company table
        result.pop("ran_out_sims")

        rows.append({
            "company_id": company_id,
            "company_name": company["company_name"].iloc[0],
            **result,
        })

    return pd.DataFrame(rows)


def scenario_table(clean_df, horizon=18):
    """Return each company's probability of running out of cash under every scenario."""
    rows = []
    for position, (company_id, company) in enumerate(clean_df.groupby("company_id")):
        row = {"company_id": company_id, "company_name": company["company_name"].iloc[0]}

        for name, settings in SCENARIOS.items():
            # Same seed in every scenario, so differences come from the scenario, not from luck
            result = simulate_company(company, horizon=horizon, seed=42 + position, scenario=settings)
            row[name] = result["prob_out_of_cash"]

        rows.append(row)

    return pd.DataFrame(rows)


def portfolio_failures(clean_df, scenario, horizon=18):
    """Return the expected number of new failures and the chance of at least 1, 2 and 3."""
    ran_out_by_company = []
    for position, (company_id, company) in enumerate(clean_df.groupby("company_id")):
        # Companies already at zero aren't "additional" failures, so leave them out
        if company.sort_values("month")["cash_eur"].iloc[-1] <= 0:
            continue
        # seed + position matches scenario_table, so both report the same simulations
        result = simulate_company(company, horizon=horizon, seed=42 + position, scenario=scenario)
        ran_out_by_company.append(result["ran_out_sims"])

    # Rows = companies, columns = simulations. Summing down each column counts the
    # companies that ran out in that simulation, because True = 1 and False = 0
    failures_per_sim = np.vstack(ran_out_by_company).sum(axis=0)

    return {
        "expected_failures": float(failures_per_sim.mean()),
        "prob_at_least_1": float((failures_per_sim >= 1).mean()),
        "prob_at_least_2": float((failures_per_sim >= 2).mean()),
        "prob_at_least_3": float((failures_per_sim >= 3).mean()),
    }


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )

    table = scenario_table(clean)
    table = table.sort_values("downturn", ascending=False)
    for name in SCENARIOS:
        table[name] = table[name].map(lambda p: f"{p:.1%}")
    print("Probability of running out of cash within 18 months (synthetic data):")
    print(table.to_string(index=False))

    for name in ["base", "downturn"]:
        failures = portfolio_failures(clean, SCENARIOS[name])
        print(f"\nAdditional failures within 18 months, {name} scenario:")
        print(f"  Expected number:   {failures['expected_failures']:.2f}")
        print(f"  At least 1:        {failures['prob_at_least_1']:.1%}")
        print(f"  At least 2:        {failures['prob_at_least_2']:.1%}")
        print(f"  At least 3:        {failures['prob_at_least_3']:.1%}")
