"""Build the text of the weekly portfolio update from the synthetic portfolio data."""

import re
from pathlib import Path

import pandas as pd

from src.metrics import add_flags, company_metrics
from src.simulate import simulate_portfolio

# All text in this report is generated from fixed templates filled with calculated numbers,
# not written by AI: the same data always produces exactly the same sentences

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

FLAG_EMOJI = {"red": "🔴", "amber": "🟠", "green": "🟢"}

# Beyond this the exact runway is noise for a reader; matches the ML model's runway cap
RUNWAY_DISPLAY_CAP = 36


def tidy_reason(reason):
    """Return one reason reworded for readers, or None if it belongs elsewhere."""
    # Estimated values go in the data quality section, not in the company line
    if reason.startswith("includes"):
        return None

    # Drop threshold notes at the end, e.g. " (over 20%)" or " (under 6)"
    reason = re.sub(r" \((over|under) [^)]*\)$", "", reason)

    # Round percentages to whole numbers, e.g. "46.7%" -> "47%"
    reason = re.sub(r"(\d+\.\d+)%", lambda match: f"{float(match.group(1)):.0f}%", reason)

    runway = re.fullmatch(r"runway ([\d.]+) months", reason)
    if runway and float(runway.group(1)) > RUNWAY_DISPLAY_CAP:
        reason = f"runway over {RUNWAY_DISPLAY_CAP} months"

    return reason


def tidy_reasons(reasons_text):
    """Return the add_flags reasons text reworded for the report."""
    tidied = [tidy_reason(reason) for reason in reasons_text.split("; ")]
    return "; ".join(reason for reason in tidied if reason is not None)


def written_off_line(row):
    """Return the short line for a company that has run out of cash."""
    return f"{FLAG_EMOJI[row['flag']]} **{row['company_name']}** ({row['sector']}): out of cash."


def company_line(row, mc_prob=None, horizon=None):
    """Return one markdown sentence describing a company's flag and reasons."""
    emoji = FLAG_EMOJI[row["flag"]]
    line = f"{emoji} **{row['company_name']}** ({row['sector']}): {tidy_reasons(row['reasons'])}"

    # Compare with None, not truthiness: a probability of 0.0 is valid and should be shown
    if mc_prob is not None:
        line += f"; {mc_prob:.0%} chance of running out within {horizon} months (Monte Carlo)"

    return line + "."


def attention_lists(flags_df, mc_df, horizon=18):
    """Return (needs_attention, watch, doing_well, written_off) lists of lines, riskiest first."""
    df = flags_df.merge(mc_df[["company_id", "prob_out_of_cash"]], on="company_id")
    # company_id breaks ties, so companies with equal probabilities keep a stable order
    df = df.sort_values(["prob_out_of_cash", "company_id"], ascending=[False, True])

    lists = {"red": [], "amber": [], "green": [], "written_off": []}
    for _, row in df.iterrows():
        # No cash left: a simulation adds nothing, so these get a short line of their own
        if row["latest_cash"] == 0:
            lists["written_off"].append(written_off_line(row))
        # The Monte Carlo sentence is only useful where there is a risk to size
        elif row["flag"] in ("red", "amber"):
            line = company_line(row, mc_prob=row["prob_out_of_cash"], horizon=horizon)
            lists[row["flag"]].append(line)
        else:
            lists[row["flag"]].append(company_line(row))

    return lists["red"], lists["amber"], lists["green"], lists["written_off"]


def flag_changes(rolling_df):
    """Return lines like 'Nobbleworth: 🟢 → 🟠' for flags that changed since last month."""
    months = sorted(rolling_df["as_of_month"].unique())
    if len(months) < 2:
        return []

    latest = rolling_df[rolling_df["as_of_month"] == months[-1]].set_index("company_id")
    previous = rolling_df[rolling_df["as_of_month"] == months[-2]].set_index("company_id")

    changes = []
    for company_id in latest.index:
        # A company missing last month has nothing to compare against
        if company_id not in previous.index:
            continue
        old_flag = previous.loc[company_id, "flag"]
        new_flag = latest.loc[company_id, "flag"]
        if old_flag != new_flag:
            name = latest.loc[company_id, "company_name"]
            changes.append(f"{name}: {FLAG_EMOJI[old_flag]} → {FLAG_EMOJI[new_flag]}")

    return changes


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )
    rolling = pd.read_csv(
        DATA_DIR / "rolling_metrics.csv", parse_dates=["as_of_month"], date_format="%Y-%m-%d"
    )

    flags = add_flags(company_metrics(clean, window=3))
    monte_carlo = simulate_portfolio(clean, horizon=18)

    needs_attention, watch, doing_well, written_off = attention_lists(flags, monte_carlo, horizon=18)

    sections = [
        ("Needs attention", needs_attention),
        ("Watch", watch),
        ("Doing well", doing_well),
        ("Written off", written_off),
    ]
    for title, lines in sections:
        print(f"\n## {title} ({len(lines)})")
        for line in lines:
            print(f"- {line}")

    print("\n## Flag changes since last month")
    changes = flag_changes(rolling)
    if changes:
        for change in changes:
            print(f"- {change}")
    else:
        print("- No flags changed.")
