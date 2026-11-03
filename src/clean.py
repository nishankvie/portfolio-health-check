"""Clean the synthetic raw portfolio data, logging every change made."""

from pathlib import Path

import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

MONEY_COLUMNS = ["revenue_eur", "costs_eur", "cash_eur"]
FIRST_MONTH = pd.Timestamp("2024-11-01")
LAST_MONTH = pd.Timestamp("2026-10-01")

# Each money column was rounded to whole euros on its own, so
# costs - net_burn can differ from revenue by up to 1 euro legitimately
ACCOUNTING_TOLERANCE_EUR = 2


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


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR / "raw_portfolio.csv")
    log = []

    # Step 1 fixes: these must run first so duplicates look identical
    cleaned = fix_names(raw, log)
    cleaned = fix_dates(cleaned, log)
    check_dates(cleaned, log)
    cleaned = fix_numbers(cleaned, log)

    rows_before = len(cleaned)
    cleaned = remove_duplicates(cleaned, log)
    rows_after = len(cleaned)

    cleaned = fix_impossible(cleaned, log)
    cleaned = fix_investment_dates(cleaned)

    print(f"Rows before duplicate removal: {rows_before}")
    print(f"Rows after duplicate removal:  {rows_after}")

    print("\nLog entries per issue:")
    print(pd.DataFrame(log)["issue"].value_counts())
