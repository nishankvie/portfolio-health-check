"""Build the text of the weekly portfolio update from the synthetic portfolio data."""

import re
from pathlib import Path

import pandas as pd

from src.metrics import (
    FLAG_RULES,
    SECTOR_MULTIPLES,
    add_flags,
    company_metrics,
    company_valuations,
    fund_summary,
)
from src.simulate import simulate_portfolio

# All text in this report is generated from fixed templates filled with calculated numbers,
# not written by AI: the same data always produces exactly the same sentences

# Resolve data/ and reports/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

ASSUMPTIONS = [
    "All data is synthetic: the companies are fictional and the numbers were generated.",
    "Valuations use illustrative sector revenue multiples, not market data.",
    "Dilution from later funding rounds is ignored, so stake values are likely overstated.",
    "Monte Carlo draws normal (bell-curve) noise, which ignores sudden jumps and extreme months.",
    "In the base scenario each company is simulated independently, so shared shocks like a "
    "recession are not captured.",
]

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


def bullet_list(lines, empty_text="None."):
    """Return lines as a markdown bullet list, or one bullet saying there are none."""
    if not lines:
        return f"- {empty_text}"
    return "\n".join(f"- {line}" for line in lines)


def fund_snapshot_lines(clean_df, metrics_df, multiples=SECTOR_MULTIPLES):
    """Return the fund snapshot as a list of short text lines."""
    summary = fund_summary(company_valuations(clean_df, metrics_df, multiples=multiples))
    return [
        f"Paid in: €{summary['paid_in']:,.0f}",
        f"NAV: €{summary['nav']:,.0f}",
        f"TVPI: {summary['tvpi']:.2f}",
        f"DPI: {summary['dpi']:.2f} (no exits yet)",
        f"Top 2 companies: {summary['top_2_share_pct']:.0f}% of NAV",
    ]


def data_quality_lines(log_df, metrics_df):
    """Return (fixes per issue, companies with estimated values) as two lists of lines."""
    issue_counts = log_df["issue"].value_counts()
    fixes = [f"{issue}: {count}" for issue, count in issue_counts.items()]

    estimated = metrics_df[metrics_df["estimated_cells"] > 0]
    estimated = estimated.sort_values(["estimated_cells", "company_id"], ascending=[False, True])
    companies = []
    for _, row in estimated.iterrows():
        word = "value" if row["estimated_cells"] == 1 else "values"
        companies.append(f"{row['company_name']}: {row['estimated_cells']} estimated {word}")

    return fixes, companies


def build_update(clean_df, log_df, rolling_df, window=3, horizon=18,
                 rules=FLAG_RULES, multiples=SECTOR_MULTIPLES):
    """Return the full portfolio update as one markdown string."""
    latest_month = clean_df["month"].max()

    metrics = company_metrics(clean_df, window=window)
    flags = add_flags(metrics, rules=rules)
    monte_carlo = simulate_portfolio(clean_df, horizon=horizon)
    needs_attention, watch, doing_well, written_off = attention_lists(flags, monte_carlo, horizon)
    changes = flag_changes(rolling_df)
    fixes, estimated_companies = data_quality_lines(log_df, metrics)

    # "Label: number" pairs avoid awkward plurals like "1 companies need"
    summary = (
        f"**Needs attention:** {len(needs_attention)} · **Watch:** {len(watch)} · "
        f"**Written off:** {len(written_off)} · **Flag changes:** {len(changes)}"
    )

    # Each part is a block of text; joining with blank lines gives clean markdown spacing
    parts = [
        f"# Portfolio update: {latest_month:%B %Y}",
        summary,
        "## Fund snapshot",
        bullet_list(fund_snapshot_lines(clean_df, metrics, multiples)),
        f"## Needs attention ({len(needs_attention)})",
        bullet_list(needs_attention),
        f"## Watch ({len(watch)})",
        bullet_list(watch),
        f"## Doing well ({len(doing_well)})",
        bullet_list(doing_well),
        f"## Written off ({len(written_off)})",
        bullet_list(written_off),
        "## Flag changes since last month",
        bullet_list(changes, empty_text="No flags changed."),
        "## Data quality",
        "Cleaning fixes by issue type:",
        bullet_list(fixes),
        "Companies whose metrics include estimated (interpolated) values:",
        bullet_list(estimated_companies),
        "## Assumptions",
        bullet_list(ASSUMPTIONS),
    ]
    return "\n\n".join(parts) + "\n"


def save_update(text, latest_month):
    """Save the update as reports/portfolio_update_YYYY-MM.md and return the path."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"portfolio_update_{latest_month:%Y-%m}.md"
    # utf-8 so the emoji and € sign are saved correctly on every operating system
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )
    rolling = pd.read_csv(
        DATA_DIR / "rolling_metrics.csv", parse_dates=["as_of_month"], date_format="%Y-%m-%d"
    )
    log = pd.read_csv(DATA_DIR / "cleaning_log.csv")

    update = build_update(clean, log, rolling)
    print(update)

    saved_path = save_update(update, clean["month"].max())
    print(f"Saved to {saved_path}")
