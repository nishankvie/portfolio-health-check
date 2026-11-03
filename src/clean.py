"""Clean the synthetic raw portfolio data, logging every change made."""

from pathlib import Path

import pandas as pd

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

MONEY_COLUMNS = ["revenue_eur", "costs_eur", "cash_eur"]
FIRST_MONTH = pd.Timestamp("2024-11-01")
LAST_MONTH = pd.Timestamp("2026-10-01")


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


if __name__ == "__main__":
    raw = pd.read_csv(DATA_DIR / "raw_portfolio.csv")
    log = []

    cleaned = fix_names(raw, log)
    cleaned = fix_dates(cleaned, log)
    check_dates(cleaned, log)
    cleaned = fix_numbers(cleaned, log)

    print(f"Unique company names: {cleaned['company_name'].nunique()}")

    print("\nColumn types:")
    print(cleaned.dtypes)

    print(f"\nEmpty dates: {cleaned['month'].isna().sum()}")

    print("\nLog entries per issue:")
    print(pd.DataFrame(log)["issue"].value_counts())
