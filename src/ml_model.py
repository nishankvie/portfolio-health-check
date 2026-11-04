"""Build a dataset for predicting which companies will run out of cash (synthetic data)."""

from pathlib import Path

import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

FEATURES = ["runway_capped", "mom_growth", "burn_trend_pct", "is_profitable"]

# Beyond 3 years the exact runway hardly changes the risk, and the model can't use infinity
RUNWAY_CAP_MONTHS = 36


def runs_out_within(company_cash, as_of_month, horizon):
    """Return 1 if cash is 0 in any month after as_of_month up to as_of_month + horizon, else 0."""
    end_month = as_of_month + pd.DateOffset(months=horizon)
    in_window = (company_cash["month"] > as_of_month) & (company_cash["month"] <= end_month)
    ran_out = (company_cash.loc[in_window, "cash_eur"] == 0).any()
    return int(ran_out)


def build_dataset(rolling_df, clean_df, horizon=6):
    """Return one row per company and month with features and a 'runs out soon' label."""
    df = rolling_df[["company_id", "as_of_month", "latest_cash", "runway_months",
                     "mom_growth", "burn_trend_pct", "avg_burn"]].copy()

    # Features: all come from rolling_df, which only used data up to as_of_month
    df["runway_capped"] = df["runway_months"].clip(upper=RUNWAY_CAP_MONTHS)
    df["is_profitable"] = (df["avg_burn"] <= 0).astype(int)

    # Label: the ONLY column allowed to look ahead, because it is the outcome we want to
    # predict. If a feature looked ahead, the model would learn from information it
    # won't have when used for real
    labels = []
    for _, row in df.iterrows():
        company_cash = clean_df[clean_df["company_id"] == row["company_id"]]
        labels.append(runs_out_within(company_cash, row["as_of_month"], horizon))
    df["label"] = labels

    # Already out of cash: there's nothing left to predict, and keeping these rows would
    # teach the model the trivial rule "cash is 0 now, so cash is 0 later"
    already_out = df["latest_cash"] == 0

    # Label unknown: the full horizon isn't in the data yet. Counting these as 0 would
    # wrongly say "survived" for companies that might run out next month
    last_month = clean_df["month"].max()
    window_incomplete = df["as_of_month"] + pd.DateOffset(months=horizon) > last_month

    df = df[~already_out & ~window_incomplete]

    # Missing features: logistic regression can't handle NaN, and guessing values would
    # make up information
    df = df.dropna(subset=FEATURES)

    return df[["company_id", "as_of_month"] + FEATURES + ["label"]].reset_index(drop=True)


if __name__ == "__main__":
    rolling = pd.read_csv(
        DATA_DIR / "rolling_metrics.csv", parse_dates=["as_of_month"], date_format="%Y-%m-%d"
    )
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )

    dataset = build_dataset(rolling, clean)
    dataset.to_csv(DATA_DIR / "ml_dataset.csv", index=False)

    print(f"Rows: {len(dataset)}")
    print("\nLabel counts:")
    print(dataset["label"].value_counts())

    companies_with_positive = dataset.loc[dataset["label"] == 1, "company_id"].nunique()
    print(f"\nCompanies with at least one label 1: {companies_with_positive}")

    print("\nExample rows with label 1:")
    print(dataset[dataset["label"] == 1].head().to_string(index=False))
