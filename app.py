"""Streamlit app for the Portfolio Health Check (synthetic data, student project)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.clean import clean_portfolio
from src.metrics import (
    FLAG_RULES,
    FLAG_SEVERITY,
    FORMULAS,
    SECTOR_MULTIPLES,
    add_flags,
    company_metrics,
    company_valuations,
    fund_summary,
    rolling_metrics,
)
from src.report import FLAG_EMOJI

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = DATA_DIR / "raw_portfolio.csv"
ESTIMATE_COLUMNS = ["revenue_estimated", "cash_estimated", "headcount_estimated"]
PAGES = ["Data cleaning", "Portfolio", "Risk", "Update", "Pension"]

st.set_page_config(page_title="Portfolio Health Check", layout="wide")


def load_into_session(raw_df, source_name):
    """Clean raw data and store raw, clean and log tables in session state."""
    # Clean first: if clean_portfolio raises, the previously stored data stays untouched
    clean_df, log_df = clean_portfolio(raw_df)
    st.session_state["raw_df"] = raw_df
    st.session_state["clean_df"] = clean_df
    st.session_state["log_df"] = log_df
    st.session_state["source_name"] = source_name


def init_session_state():
    """Load and clean the sample data once, on the first run of the session."""
    if "raw_df" not in st.session_state:
        load_into_session(pd.read_csv(SAMPLE_FILE), "sample data")


def for_display(df):
    """Return a copy with mixed-type text columns as strings, so the table can show them."""
    display = df.copy()
    # The log's old/new values mix numbers, text and dates in one column, which the
    # table widget can't store as a single type
    for column in ["month", "old_value", "new_value"]:
        if column in display.columns:
            display[column] = display[column].astype(str)
    return display


def handle_upload():
    """Replace the stored data with an uploaded CSV, cleaned, unless it was already processed."""
    uploaded = st.file_uploader("Upload a raw portfolio CSV", type="csv")
    if uploaded is None:
        return

    # The uploader keeps its file on every rerun; only clean a file we haven't seen yet,
    # otherwise "Reset to sample data" would be undone on the very next rerun
    if st.session_state.get("processed_upload_id") == uploaded.file_id:
        return
    st.session_state["processed_upload_id"] = uploaded.file_id

    try:
        load_into_session(pd.read_csv(uploaded), uploaded.name)
        st.success(f"Cleaned {uploaded.name}.")
    except ValueError as error:
        st.error(f"Could not clean {uploaded.name}: {error}")


def page_data_cleaning():
    """Show the cleaning page: upload, summary metrics, tables and downloads."""
    st.title("Data cleaning")
    st.markdown(
        "The cleaner follows one rule: **fix what's safe, flag what isn't, log everything.** "
        "Format problems (dates, Austrian number text, name variants, duplicates) are fixed. "
        "Impossible values are repaired from the accounting identity or blanked, gaps are "
        "estimated and marked, and every change is written to the cleaning log."
    )

    handle_upload()
    if st.button("Reset to sample data"):
        load_into_session(pd.read_csv(SAMPLE_FILE), "sample data")

    raw_df = st.session_state["raw_df"]
    clean_df = st.session_state["clean_df"]
    log_df = st.session_state["log_df"]
    st.caption(f"Showing: {st.session_state['source_name']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows in", len(raw_df))
    col2.metric("Rows out", len(clean_df))
    col3.metric("Total fixes", len(log_df))
    col4.metric("Estimated values", int(clean_df[ESTIMATE_COLUMNS].sum().sum()))

    raw_tab, clean_tab, log_tab = st.tabs(["Raw data", "Cleaned data", "Cleaning log"])
    with raw_tab:
        st.dataframe(raw_df, width="stretch")
    with clean_tab:
        st.dataframe(clean_df, width="stretch")
    with log_tab:
        issues = ["All"] + sorted(log_df["issue"].unique())
        chosen = st.selectbox("Filter by issue", issues)
        shown = log_df if chosen == "All" else log_df[log_df["issue"] == chosen]
        st.dataframe(for_display(shown), width="stretch")

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download cleaned CSV",
        data=clean_df.to_csv(index=False),
        file_name="clean_portfolio.csv",
        mime="text/csv",
    )
    download_col2.download_button(
        "Download cleaning log",
        data=log_df.to_csv(index=False),
        file_name="cleaning_log.csv",
        mime="text/csv",
    )


@st.cache_data
def cached_rolling_metrics(clean_df, window, rules):
    """Return rolling_metrics, cached: it reruns company_metrics once per month, so it's slow."""
    # Streamlit reruns the whole script on every click; the cache returns the stored result
    # when clean_df, window and rules are the same as a previous call
    return rolling_metrics(clean_df, window=window, rules=rules)


def portfolio_controls():
    """Show the window, flag rule and valuation controls; return (window, rules, multiples)."""
    window = st.slider("Metric window (months)", min_value=1, max_value=12, value=3)

    with st.expander("Flag rules"):
        rules = {
            "red_runway": st.slider("Red if runway under (months)", 1, 12,
                                    FLAG_RULES["red_runway"]),
            "amber_runway": st.slider("Amber if runway under (months)", 3, 24,
                                      FLAG_RULES["amber_runway"]),
            "max_burn_trend": st.slider("Amber if burn rises more than (%)", 0, 100,
                                        FLAG_RULES["max_burn_trend"]),
            "max_burn_multiple": st.slider("Amber if burn multiple above", 1, 10,
                                           FLAG_RULES["max_burn_multiple"]),
        }

    with st.expander("Valuation assumptions"):
        st.caption("ARR multiples per sector. Illustrative assumptions, not market data.")
        multiples = {}
        for sector, default in SECTOR_MULTIPLES.items():
            multiples[sector] = st.number_input(
                f"{sector} multiple", min_value=0.0, value=float(default), step=0.5
            )

    return window, rules, multiples


def runway_text(runway):
    """Return runway as readable text: 'profitable' when cash never runs out."""
    if runway == np.inf:
        return "profitable"
    return f"{runway:.1f} months"


def burn_multiple_text(value, note):
    """Return the burn multiple, or the note explaining why there is none."""
    if pd.isna(value):
        return note
    return f"{value:.1f}"


def company_table(flagged, valuations):
    """Return the display table for the portfolio page, sorted red -> amber -> green."""
    df = flagged.merge(valuations[["company_id", "moic"]], on="company_id")
    df["severity"] = df["flag"].map(FLAG_SEVERITY)
    df = df.sort_values(["severity", "company_id"], ascending=[False, True])

    return pd.DataFrame({
        "Flag": df["flag"].map(FLAG_EMOJI),
        "Company": df["company_name"],
        "Sector": df["sector"],
        "Runway": df["runway_months"].apply(runway_text),
        "Monthly growth": df["mom_growth"] * 100,
        "Burn trend": df["burn_trend_pct"],
        "Burn multiple": [burn_multiple_text(value, note) for value, note
                          in zip(df["burn_multiple"], df["burn_multiple_note"])],
        "MOIC": df["moic"],
        "Reasons": df["reasons"],
    })


def show_formulas(window, rules):
    """Show every metric's formula and explanation, with the current settings."""
    with st.expander("How each metric is calculated"):
        st.markdown(f"Current window: **w = {window} months**")
        for name, formula in FORMULAS.items():
            st.markdown(f"**{name}**")
            st.latex(formula["latex"])
            st.caption(formula["explanation"])
            # The stored explanation uses the default thresholds; show the slider values too
            if name == "flag":
                st.caption(
                    f"Current rules: red under {rules['red_runway']} months; amber under "
                    f"{rules['amber_runway']} months, burn up more than {rules['max_burn_trend']}%, "
                    f"or burn multiple above {rules['max_burn_multiple']}."
                )


def show_company_detail(clean_df, window, rules):
    """Show one company's cash and revenue chart and its monthly flag history."""
    st.subheader("Company detail")
    names = clean_df.drop_duplicates("company_id").set_index("company_id")["company_name"]
    company_id = st.selectbox(
        "Company", names.index, format_func=lambda cid: f"{cid} · {names[cid]}"
    )

    company = clean_df[clean_df["company_id"] == company_id].sort_values("month")
    chart = px.line(
        company, x="month", y=["cash_eur", "revenue_eur"],
        labels={"value": "EUR", "month": "Month", "variable": ""},
        title=f"{names[company_id]}: cash and monthly revenue",
    )
    st.plotly_chart(chart, width="stretch")

    rolling = cached_rolling_metrics(clean_df, window, rules)
    history = rolling[rolling["company_id"] == company_id].sort_values("as_of_month")
    if history.empty:
        st.caption("Not enough months of data for a flag history with this window.")
        return
    emojis = "".join(FLAG_EMOJI[flag] for flag in history["flag"])
    first, last = history["as_of_month"].min(), history["as_of_month"].max()
    st.markdown(f"**Flag history** ({first:%Y-%m} to {last:%Y-%m}, one per month)")
    st.markdown(emojis)


def page_portfolio():
    """Show the portfolio page: controls, fund snapshot, company table and detail."""
    st.title("Portfolio")
    clean_df = st.session_state["clean_df"]
    window, rules, multiples = portfolio_controls()

    metrics = company_metrics(clean_df, window=window)
    flagged = add_flags(metrics, rules=rules)
    valuations = company_valuations(clean_df, metrics, multiples=multiples)
    summary = fund_summary(valuations)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Paid-in", f"€{summary['paid_in']:,.0f}")
    col2.metric("NAV", f"€{summary['nav']:,.0f}")
    col3.metric("TVPI", f"{summary['tvpi']:.2f}x")
    col4.metric("DPI", f"{summary['dpi']:.2f}x")
    col5.metric("Top 2 share", f"{summary['top_2_share_pct']:.0f}%")

    st.dataframe(
        company_table(flagged, valuations),
        hide_index=True,
        width="stretch",
        column_config={
            "Monthly growth": st.column_config.NumberColumn(format="%.1f%%"),
            "Burn trend": st.column_config.NumberColumn(format="%.0f%%"),
            "MOIC": st.column_config.NumberColumn(format="%.2fx"),
            "Reasons": st.column_config.TextColumn(width="large"),
        },
    )

    show_formulas(window, rules)
    show_company_detail(clean_df, window, rules)


def page_coming_soon(name):
    """Show a placeholder for a page that isn't built yet."""
    st.title(name)
    st.info("Coming soon")


init_session_state()

page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("All data is synthetic and this tool is a student project, not investment advice.")

if page == "Data cleaning":
    page_data_cleaning()
elif page == "Portfolio":
    page_portfolio()
else:
    page_coming_soon(page)
