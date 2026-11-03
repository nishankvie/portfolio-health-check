"""Clean the synthetic raw portfolio data, logging every change made."""

from pathlib import Path

import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

REQUIRED_COLUMNS = [
    "company_id",
    "month",
    "revenue_eur",
    "costs_eur",
    "net_burn_eur",
    "cash_eur",
    "headcount",
    "company_name",
    "sector",
    "investment_date",
    "amount_invested_eur",
    "ownership_pct",
    "entry_valuation_eur",
]

MONEY_COLUMNS = ["revenue_eur", "costs_eur", "cash_eur"]
FIRST_MONTH = pd.Timestamp("2024-11-01")
LAST_MONTH = pd.Timestamp("2026-10-01")

# Each money column was rounded to whole euros on its own, so
# costs - net_burn can differ from revenue by up to 1 euro legitimately
ACCOUNTING_TOLERANCE_EUR = 2

# Columns that describe the company, not the month, so they're the same on every row
STATIC_COLUMNS = [
    "company_name",
    "sector",
    "investment_date",
    "amount_invested_eur",
    "ownership_pct",
    "entry_valuation_eur",
]
INTERPOLATE_COLUMNS = ["revenue_eur", "costs_eur", "cash_eur", "headcount"]

# One flag per scored column, so each value's score uses only cells that were actually guessed
ESTIMATE_FLAGS = {
    "revenue_eur": "revenue_estimated",
    "cash_eur": "cash_estimated",
    "headcount": "headcount_estimated",
}


def check_columns(df):
    """Raise a ValueError naming any required columns the data is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def log_change(log, company_id, month, column, issue, old_value, new_value):
    """Append one record of a change (or a flagged problem) to the log list."""
    log.append({
        "company_id": company_id,
        "month": month,
        "column": column,
        "issue": issue,
        "old_value": old_value,
        "new_value": new_value,
    })


def fix_names(df, log):
    """Set every row's company_name to that company's most common spelling."""
    df = df.copy()
    for company_id in df["company_id"].unique():
        company_rows = df["company_id"] == company_id
        # The correct spelling is the majority one; variants are rare typos
        most_common_name = df.loc[company_rows, "company_name"].value_counts().idxmax()

        for row in df.index[company_rows]:
            old_name = df.loc[row, "company_name"]
            if old_name != most_common_name:
                log_change(log, company_id, df.loc[row, "month"], "company_name",
                           "name variant", old_name, most_common_name)
                df.loc[row, "company_name"] = most_common_name
    return df


def fix_dates(df, log):
    """Parse month as a date: ISO format first, then dd.mm.yyyy for the rest."""
    df = df.copy()
    original_text = df["month"]

    parsed = pd.to_datetime(original_text, format="%Y-%m-%d", errors="coerce")

    # Only retry rows that had text but failed the ISO format
    failed = parsed.isna() & original_text.notna()
    second_try = pd.to_datetime(original_text[failed], format="%d.%m.%Y", errors="coerce")
    parsed[failed] = second_try

    for row in second_try.index:
        if pd.notna(second_try[row]):
            log_change(log, df.loc[row, "company_id"], second_try[row], "month",
                       "date format", original_text[row], second_try[row])

    df["month"] = parsed
    return df


def check_dates(df, log):
    """Log months that aren't a 1st-of-month inside the data period; change nothing."""
    months = df["month"]
    # NaT (unparseable dates) compares as False here, so those are not flagged twice
    not_first_of_month = months.dt.day != 1
    out_of_range = (months < FIRST_MONTH) | (months > LAST_MONTH)
    suspicious = not_first_of_month | out_of_range

    for row in df.index[suspicious]:
        log_change(log, df.loc[row, "company_id"], months[row], "month",
                   "suspicious date", months[row], months[row])


def parse_eur(value):
    """Turn Austrian-format text like '€12.345' or '12.345,50' into a float."""
    if not isinstance(value, str):
        return value

    text = value.replace("€", "").replace(" ", "")
    # Austrian format: dot separates thousands, comma marks decimals
    text = text.replace(".", "").replace(",", ".")
    return float(text)


def fix_numbers(df, log):
    """Convert money columns from text to numbers, logging formatted text cells."""
    df = df.copy()
    for column in MONEY_COLUMNS:
        for row in df.index:
            old_value = df.loc[row, column]
            # read_csv makes the whole column text if one cell is, so plain
            # digits like "52506" are not errors; only formatted text is
            is_formatted_text = isinstance(old_value, str) and not old_value.lstrip("-").isdigit()
            if is_formatted_text:
                log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], column,
                           "text number", old_value, parse_eur(old_value))

        df[column] = df[column].apply(parse_eur)
    return df


def remove_duplicates(df, log):
    """Drop exact duplicate rows, then extra rows for the same company and month."""
    df = df.copy()

    # Exact duplicates: every column identical to an earlier row
    exact_duplicate = df.duplicated()
    for row in df.index[exact_duplicate]:
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], "all",
                   "exact duplicate", None, "dropped")
    df = df[~exact_duplicate]

    # Conflicting duplicates: same company and month but different values.
    # We can't tell which is right, so keep the first and log the rest for review
    conflicting_duplicate = df.duplicated(subset=["company_id", "month"])
    for row in df.index[conflicting_duplicate]:
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], "all",
                   "conflicting duplicate", None, "dropped")
    df = df[~conflicting_duplicate]

    return df


def fix_impossible(df, log):
    """Blank out negative headcounts and repair revenue that breaks the accounting identity."""
    df = df.copy()

    # Negative headcount: impossible, and we can't know the true value
    negative_headcount = df["headcount"] < 0
    for row in df.index[negative_headcount]:
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], "headcount",
                   "negative headcount", df.loc[row, "headcount"], None)
    df.loc[negative_headcount, "headcount"] = None

    # Accounting mismatch: net_burn = costs - revenue must hold, so revenue = costs - net_burn
    expected_revenue = df["costs_eur"] - df["net_burn_eur"]
    has_revenue = df["revenue_eur"].notna()
    too_far_off = (df["revenue_eur"] - expected_revenue).abs() > ACCOUNTING_TOLERANCE_EUR
    mismatch = has_revenue & too_far_off
    for row in df.index[mismatch]:
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], "revenue_eur",
                   "accounting mismatch", df.loc[row, "revenue_eur"], expected_revenue[row])
    df.loc[mismatch, "revenue_eur"] = expected_revenue[mismatch]

    return df


def fix_investment_dates(df):
    """Convert investment_date from text to a real date."""
    df = df.copy()
    # No errors="coerce": this column was never messy, so any failure is a surprise worth crashing on
    df["investment_date"] = pd.to_datetime(df["investment_date"], format="%Y-%m-%d")
    return df


def fill_missing_months(df, log):
    """Add a row for every company-month that is missing, with company details filled in."""
    # Every company paired with every month: the table we should have
    company_ids = pd.DataFrame({"company_id": df["company_id"].unique()})
    # Take the period from the data itself, so a new export with other months still works
    first_month = df["month"].min()
    last_month = df["month"].max()
    months = pd.DataFrame({"month": pd.date_range(first_month, last_month, freq="MS")})
    grid = company_ids.merge(months, how="cross")

    # indicator=True adds a _merge column saying whether each row was found in df
    full = grid.merge(df, on=["company_id", "month"], how="left", indicator=True)

    new_rows = full["_merge"] == "left_only"
    for row in full.index[new_rows]:
        log_change(log, full.loc[row, "company_id"], full.loc[row, "month"], "all",
                   "missing month", None, "row added")
    full = full.drop(columns="_merge")

    # Static details are identical on every row of a company, so copy them from its other rows
    for column in STATIC_COLUMNS:
        company_value = full.groupby("company_id")[column].transform("first")
        full[column] = full[column].fillna(company_value)
        # The NaNs in new rows turned whole-number columns into floats; restore the original type
        full[column] = full[column].astype(df[column].dtype)

    return full


def fill_missing_values(df, log):
    """Recalculate revenue where possible, then interpolate remaining gaps per company."""
    df = df.copy()

    # Recalculate: if costs and net_burn are known, revenue follows exactly from the identity
    can_recalculate = (
        df["revenue_eur"].isna() & df["costs_eur"].notna() & df["net_burn_eur"].notna()
    )
    recalculated_revenue = df["costs_eur"] - df["net_burn_eur"]
    for row in df.index[can_recalculate]:
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], "revenue_eur",
                   "revenue recalculated", None, recalculated_revenue[row])
    df.loc[can_recalculate, "revenue_eur"] = recalculated_revenue[can_recalculate]

    # Interpolate: order matters, because interpolation fills a gap from its neighbours
    df = df.sort_values(["company_id", "month"]).reset_index(drop=True)
    was_missing = df[INTERPOLATE_COLUMNS].isna()

    for column in INTERPOLATE_COLUMNS:
        # groupby + transform runs interpolate separately inside each company
        df[column] = df.groupby("company_id")[column].transform(
            lambda values: values.interpolate(limit_direction="both")
        )

    # Burn must stay consistent with the estimated revenue and costs.
    # Rows where only headcount was filled keep their real burn
    money_filled = was_missing["revenue_eur"] | was_missing["costs_eur"]
    df.loc[money_filled, "net_burn_eur"] = (
        df.loc[money_filled, "costs_eur"] - df.loc[money_filled, "revenue_eur"]
    )

    df["headcount"] = df["headcount"].round()

    for column, flag in ESTIMATE_FLAGS.items():
        df[flag] = was_missing[column]

    any_filled = was_missing.any(axis=1)
    for row in df.index[any_filled]:
        filled_columns = [column for column in INTERPOLATE_COLUMNS if was_missing.loc[row, column]]
        log_change(log, df.loc[row, "company_id"], df.loc[row, "month"], ", ".join(filled_columns),
                   "value estimated", None, "interpolated")

    return df


def score_against_truth(clean, truth):
    """Print how close the cleaned values are to the truth, for real and estimated rows."""
    truth = truth.copy()
    truth["month"] = pd.to_datetime(truth["month"], format="%Y-%m-%d")
    merged = clean.merge(truth, on=["company_id", "month"], suffixes=("", "_truth"))

    for column, flag in ESTIMATE_FLAGS.items():
        print(f"\n{column}:")
        for is_estimated in [False, True]:
            rows = merged[merged[flag] == is_estimated]
            true_values = rows[column + "_truth"]
            absolute_error = (rows[column] - true_values).abs()

            # Compare to 1% of the truth instead of dividing, so true zeros (empty cash) work
            within_one_pct = absolute_error <= 0.01 * true_values.abs()
            label = "estimated" if is_estimated else "real     "
            print(f"  {label}: {within_one_pct.mean():6.1%} within 1% of truth (n={len(rows)})")

            if is_estimated:
                # Percent error is undefined when the truth is 0, so leave those rows out
                nonzero = true_values != 0
                pct_error = absolute_error[nonzero] / true_values[nonzero].abs()
                print(f"  estimated average error: {pct_error.mean():.1%} (n={nonzero.sum()})")


def clean_portfolio(raw_df):
    """Run every cleaning step in order; return the clean data and the change log."""
    check_columns(raw_df)
    log = []

    # Format fixes must run first so duplicates look identical
    cleaned = fix_names(raw_df, log)
    cleaned = fix_dates(cleaned, log)
    check_dates(cleaned, log)
    cleaned = fix_numbers(cleaned, log)

    cleaned = remove_duplicates(cleaned, log)
    # Impossible values must be fixed before filling, or they'd leak into the estimates
    cleaned = fix_impossible(cleaned, log)
    cleaned = fix_investment_dates(cleaned)

    cleaned = fill_missing_months(cleaned, log)
    cleaned = fill_missing_values(cleaned, log)

    log_df = pd.DataFrame(log)
    return cleaned, log_df


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR / "raw_portfolio.csv")
    cleaned, log_df = clean_portfolio(raw)

    cleaned.to_csv(DATA_DIR / "clean_portfolio.csv", index=False)
    log_df.to_csv(DATA_DIR / "cleaning_log.csv", index=False)

    print(f"Final rows: {len(cleaned)}")
    print("\nLog entries per issue:")
    print(log_df["issue"].value_counts())

    truth = pd.read_csv(DATA_DIR / "truth_portfolio.csv")
    print("\nScore against truth:")
    score_against_truth(cleaned, truth)
