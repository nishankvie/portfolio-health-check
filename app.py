"""Streamlit app for the Portfolio Health Check (synthetic data, student project)."""

import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from src.ml_model import (
    add_rule_flags,
    build_dataset,
    evaluate,
    risk_today,
    train_forest_cv,
    train_logistic_cv,
)
from src.pension import (
    ASSUMPTIONS,
    PAYMENT_FREQUENCIES,
    VALUATION_DATE,
    annuity_contributions,
    assumptions_table,
    dbo_monte_carlo_draws,
    dbo_percentiles,
    dbo_portfolio,
    death_prob,
    duration,
    employee_age_range,
    employee_day_counts,
    employee_export_table,
    exact_rate_move,
    excel_workbook,
    expected_cash_flows,
    run_model_checks,
    run_record,
    sensitivity_table,
    standard_outputs,
    survival_prob,
    validate_employees,
    validate_mortality_table,
)
from src.report import FLAG_EMOJI, build_update
from src.simulate import SCENARIOS, portfolio_failures, scenario_table, simulate_company

ML_WINDOW = 3

# Session-state keys of the Portfolio page widgets, with their starting values
WIDGET_DEFAULTS = {
    "window": 3,
    "red_runway": FLAG_RULES["red_runway"],
    "amber_runway": FLAG_RULES["amber_runway"],
    "max_burn_trend": FLAG_RULES["max_burn_trend"],
    "max_burn_multiple": FLAG_RULES["max_burn_multiple"],
}
for _sector, _multiple in SECTOR_MULTIPLES.items():
    WIDGET_DEFAULTS[f"multiple_{_sector}"] = float(_multiple)

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = DATA_DIR / "raw_portfolio.csv"
EMPLOYEES_FILE = DATA_DIR / "employees_raw.csv"
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


def keep_widget_settings():
    """Make sure every Portfolio widget key exists and survives page switches."""
    # Streamlit deletes a widget's value when the widget isn't drawn on a run (e.g. while
    # another page is shown). Writing the value back ourselves turns it into ordinary session
    # state, which is kept; on the very first run the default is stored instead
    for key, default in WIDGET_DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, default)


def current_rules():
    """Return the flag rules currently set on the Portfolio page."""
    return {name: st.session_state.get(name, FLAG_RULES[name]) for name in FLAG_RULES}


def current_multiples():
    """Return the sector multiples currently set on the Portfolio page."""
    return {sector: st.session_state.get(f"multiple_{sector}", float(default))
            for sector, default in SECTOR_MULTIPLES.items()}


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
    # No value= arguments: each widget takes its value from session_state[key], which
    # keep_widget_settings has already filled with the default or the user's last choice
    window = st.slider("Metric window (months)", min_value=1, max_value=12, key="window")

    with st.expander("Flag rules"):
        st.slider("Red if runway under (months)", 1, 12, key="red_runway")
        st.slider("Amber if runway under (months)", 3, 24, key="amber_runway")
        st.slider("Amber if burn rises more than (%)", 0, 100, key="max_burn_trend")
        st.slider("Amber if burn multiple above", 1, 10, key="max_burn_multiple")

    with st.expander("Valuation assumptions"):
        st.caption("ARR multiples per sector. Illustrative assumptions, not market data.")
        for sector in SECTOR_MULTIPLES:
            st.number_input(f"{sector} multiple", min_value=0.0, step=0.5,
                            key=f"multiple_{sector}")

    return window, current_rules(), current_multiples()


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


@st.cache_data
def cached_scenario_table(clean_df, horizon):
    """Return each company's probability of running out under every scenario (cached)."""
    return scenario_table(clean_df, horizon=horizon)


@st.cache_data
def cached_portfolio_failures(clean_df, scenario_name, horizon):
    """Return the portfolio failure statistics for one scenario (cached)."""
    return portfolio_failures(clean_df, SCENARIOS[scenario_name], horizon=horizon)


@st.cache_data
def cached_cash_bands(clean_df, company_id, scenario_name, horizon):
    """Return one company's monthly P10/P50/P90 cash (cached)."""
    # Same seed as scenario_table (42 + position in the sorted list), so the chart and the
    # table describe the same simulated futures
    position = sorted(clean_df["company_id"].unique()).index(company_id)
    company = clean_df[clean_df["company_id"] == company_id]
    result = simulate_company(company, horizon=horizon, seed=42 + position,
                              scenario=SCENARIOS[scenario_name])
    return result["cash_bands"]


@st.cache_data
def cached_ml_dataset(clean_df):
    """Return the labelled ML dataset with rule flags, built from window-3 rolling metrics."""
    rolling = cached_rolling_metrics(clean_df, ML_WINDOW, FLAG_RULES)
    return add_rule_flags(build_dataset(rolling, clean_df), rolling)


@st.cache_data
def cached_cv_proba(clean_df, model_name):
    """Return leave-one-company-out probabilities for the chosen model (cached)."""
    dataset = cached_ml_dataset(clean_df)
    if model_name == "Logistic regression":
        return train_logistic_cv(dataset)
    return train_forest_cv(dataset)


@st.cache_data
def cached_risk_today(clean_df):
    """Return today's ML risk and 6-month Monte Carlo probability per company (cached)."""
    rolling = cached_rolling_metrics(clean_df, ML_WINDOW, FLAG_RULES)
    return risk_today(rolling, clean_df)


def as_percent(series):
    """Return probabilities (0-1) as percentages (0-100) for display."""
    return series * 100


def company_names(clean_df):
    """Return a lookup from company_id to company_name."""
    return clean_df.drop_duplicates("company_id").set_index("company_id")["company_name"]


def fan_chart(bands, start_month, title):
    """Return a plotly chart: P50 line with a shaded P10-P90 band."""
    months = [start_month + pd.DateOffset(months=int(k)) for k in bands["month_ahead"]]
    chart = go.Figure()
    # Upper edge first with no line, then the lower edge filled up to it ("tonexty")
    chart.add_trace(go.Scatter(x=months, y=bands["p90"], mode="lines", line={"width": 0},
                               name="P90", showlegend=False))
    chart.add_trace(go.Scatter(x=months, y=bands["p10"], mode="lines", line={"width": 0},
                               fill="tonexty", fillcolor="rgba(70, 130, 180, 0.25)",
                               name="P10 to P90"))
    chart.add_trace(go.Scatter(x=months, y=bands["p50"], mode="lines",
                               line={"color": "steelblue", "width": 2}, name="P50 (median)"))
    chart.update_layout(title=title, xaxis_title="Month", yaxis_title="Cash (EUR)")
    return chart


def risk_monte_carlo_section(clean_df):
    """Show scenario probabilities, portfolio failures and a cash fan chart."""
    st.header("Monte Carlo")
    st.warning("Exploratory, work in progress: companies are simulated independently, using growth "
               "rates fitted to 12 months of synthetic data, so these probabilities only "
               "demonstrate the method.")
    horizon = st.slider("Horizon (months)", 6, 24, 18)
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))

    table = cached_scenario_table(clean_df, horizon).copy()
    for name in SCENARIOS:
        table[name] = as_percent(table[name])
    table = table.sort_values(scenario_name, ascending=False)
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={name: st.column_config.NumberColumn(format="%.1f%%") for name in SCENARIOS},
    )

    failures = cached_portfolio_failures(clean_df, scenario_name, horizon)
    st.markdown(f"**Additional failures within {horizon} months ({scenario_name})**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected failures", f"{failures['expected_failures']:.2f}")
    col2.metric("At least 1", f"{failures['prob_at_least_1']:.0%}")
    col3.metric("At least 2", f"{failures['prob_at_least_2']:.0%}")
    col4.metric("At least 3", f"{failures['prob_at_least_3']:.0%}")

    latest_month = clean_df["month"].max()
    latest = clean_df[clean_df["month"] == latest_month]
    with_cash = sorted(latest.loc[latest["cash_eur"] > 0, "company_id"])
    names = company_names(clean_df)
    company_id = st.selectbox("Company (with cash)", with_cash,
                              format_func=lambda cid: f"{cid} · {names[cid]}")
    bands = cached_cash_bands(clean_df, company_id, scenario_name, horizon)
    chart = fan_chart(bands, latest_month, f"{names[company_id]}: simulated cash, {scenario_name}")
    st.plotly_chart(chart, width="stretch")

    st.caption(
        "Assumptions: 1,000 simulations; monthly revenue and cost growth drawn from a normal "
        "distribution fitted to the last 12 months; companies simulated independently (no shared "
        "shocks, except that a scenario applies to all); normal noise ignores fat tails, and jumps "
        "appear only in the downturn scenario; synthetic data."
    )


def risk_ml_section(clean_df):
    """Show the rule-vs-model comparison and today's risk table."""
    st.header("Machine learning")
    st.warning("Exploratory, work in progress: the models learn from very few companies that ran "
               "out of money, in synthetic data, so these scores only demonstrate the method.")
    model_name = st.radio("Model", ["Logistic regression", "Random forest"], horizontal=True)
    threshold = st.slider("Probability threshold", 0.1, 0.9, 0.5, step=0.05)

    dataset = cached_ml_dataset(clean_df)
    proba = cached_cv_proba(clean_df, model_name)
    y = dataset["label"]

    detectors = [
        ("Rule: red", dataset["rule_red"]),
        ("Rule: red or amber", dataset["rule_red_amber"]),
        (f"{model_name} (threshold {threshold:.2f})", (proba >= threshold).astype(int)),
    ]
    rows = []
    for name, predictions in detectors:
        result = evaluate(y, predictions, name)
        rows.append({
            "Detector": name,
            "Precision": result["precision"],
            "Recall": result["recall"],
            "Caught": f"{result['TP']} of {result['TP'] + result['FN']}",
            "False alarms": result["FP"],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown("**Risk today** (companies with cash)")
    today = cached_risk_today(clean_df).copy()
    today.insert(1, "company_name", today["company_id"].map(company_names(clean_df)))
    today["ml_risk"] = as_percent(today["ml_risk"])
    today["mc_prob_6m"] = as_percent(today["mc_prob_6m"])
    st.dataframe(
        today.sort_values("ml_risk", ascending=False), hide_index=True, width="stretch",
        column_config={
            "ml_risk": st.column_config.NumberColumn("ML risk (6 months)", format="%.1f%%"),
            "mc_prob_6m": st.column_config.NumberColumn("Monte Carlo (6 months)", format="%.1f%%"),
        },
    )

    st.caption(
        "Caveats: synthetic data; only 4 companies ever ran out, so the model learns from very "
        "few events; scores use leave-one-company-out validation (each company is predicted by a "
        f"model that never saw it); the model mostly relies on runway; ML features use a fixed "
        f"{ML_WINDOW}-month window, independent of the Portfolio page slider."
    )


def page_risk():
    """Show the risk page: Monte Carlo and machine learning sections."""
    st.title("Risk")
    clean_df = st.session_state["clean_df"]
    risk_monte_carlo_section(clean_df)
    st.divider()
    risk_ml_section(clean_df)


def page_update():
    """Show the generated portfolio update, using the Portfolio page settings."""
    st.title("Update")
    clean_df = st.session_state["clean_df"]
    log_df = st.session_state["log_df"]
    window = st.session_state.get("window", WIDGET_DEFAULTS["window"])
    rules = current_rules()
    multiples = current_multiples()

    horizon = st.slider("Monte Carlo horizon (months)", 6, 24, 18)
    rolling = cached_rolling_metrics(clean_df, window, rules)
    update = build_update(clean_df, log_df, rolling, window=window, horizon=horizon,
                          rules=rules, multiples=multiples)

    latest_month = clean_df["month"].max()
    st.download_button(
        "Download update (.md)",
        data=update,
        file_name=f"portfolio_update_{latest_month:%Y-%m}.md",
        mime="text/markdown",
    )
    st.caption(
        f"Generated from fixed templates and calculated numbers, not by AI. Uses the settings "
        f"from the Portfolio page (window {window} months, red under {rules['red_runway']} "
        f"months of runway)."
    )
    st.divider()
    st.markdown(update)


# ---------------------------------------------------------------------------------------------
# Pension page. All calculations live in src/pension.py; this section only arranges and formats.
# ---------------------------------------------------------------------------------------------

def eur(value):
    """Return a euro amount as text with thousands separators, e.g. €2,821,996."""
    return f"€{value:,.0f}"


def tex_eur(value):
    """Return a euro amount for LaTeX: '{,}' stops the comma adding a space in maths mode."""
    return r"\text{€}" + f"{value:,.0f}".replace(",", "{,}")


def tex_int(value):
    """Return a whole number with thousands separators for LaTeX."""
    return f"{value:,}".replace(",", "{,}")


def load_employees_into_session(raw_df, source_name):
    """Check that an employee file can be validated, then store it with its name."""
    # Validate first: if it raises, the previously stored employee data stays untouched
    validate_employees(raw_df)
    st.session_state["employees_raw"] = raw_df
    st.session_state["employees_source"] = source_name


def handle_employee_upload():
    """Replace the stored employee data with an uploaded CSV, unless already processed."""
    uploaded = st.file_uploader("Upload an employee CSV", type="csv", key="employee_upload")
    if uploaded is None:
        return
    if st.session_state.get("processed_employee_upload_id") == uploaded.file_id:
        return
    st.session_state["processed_employee_upload_id"] = uploaded.file_id

    try:
        load_employees_into_session(pd.read_csv(uploaded), uploaded.name)
    except ValueError as error:
        st.error(f"Could not validate {uploaded.name}: {error}")


@st.cache_data
def cached_validate(raw_df, retirement_age):
    """Return (valid employees, change log, client queries) for one retirement age (cached)."""
    return validate_employees(raw_df, VALUATION_DATE, retirement_age)


@st.cache_data
def cached_dbo(valid_df, assumptions):
    """Return (per-employee DBO table, total DBO) for one set of assumptions (cached)."""
    return dbo_portfolio(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_sensitivity(valid_df, assumptions):
    """Return the sensitivity table around the given assumptions (cached)."""
    return sensitivity_table(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_dbo_draws(valid_df, assumptions):
    """Return the 500 simulated total DBOs (cached, so each new setting runs only once)."""
    return dbo_monte_carlo_draws(valid_df, VALUATION_DATE, a=assumptions)


@st.cache_data
def cached_annuity_contributions(age, assumptions):
    """Return each payment's contribution to the annuity factor (cached)."""
    return annuity_contributions(age, assumptions)


@st.cache_data
def cached_rate_move(valid_df, assumptions, move):
    """Return (total DBO, % change) with the discount rate moved by move (cached)."""
    return exact_rate_move(valid_df, VALUATION_DATE, assumptions, move)


@st.cache_data
def cached_standard_outputs(results, valid_df, assumptions):
    """Return next year's service cost, interest cost and the interest simplification (cached)."""
    return standard_outputs(results, valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_model_checks(valid_df, assumptions):
    """Return the model checks for one set of data and assumptions (cached)."""
    return run_model_checks(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_cash_flows(valid_df, assumptions):
    """Return expected yearly pension payments, undiscounted and discounted (cached)."""
    return expected_cash_flows(valid_df, VALUATION_DATE, assumptions)


def sensitivity_chart(table):
    """Return a horizontal bar chart of % change in DBO per scenario, largest at the top."""
    changes = table[table["scenario"] != "base"].copy()
    # Plotly draws the first row at the bottom, so sort smallest first to put the largest on top
    changes["size"] = changes["pct_change"].abs()
    changes = changes.sort_values("size")
    chart = px.bar(changes, x="pct_change", y="scenario", orientation="h",
                   labels={"pct_change": "% change in total DBO", "scenario": ""},
                   title="Sensitivity of total DBO")
    return chart


def pension_data_section(raw, log_df, valid, queries, results):
    """Show validation counts, the client questions and anyone valued as retiring today."""
    st.header("Data checks")
    st.caption(f"Showing: {st.session_state['employees_source']}")
    issues = log_df["issue"] if not log_df.empty else pd.Series(dtype=str)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Raw rows", len(raw))
    col2.metric("Duplicates removed", int((issues == "exact duplicate").sum()))
    col3.metric("Text salaries converted", int((issues == "text number").sum()))
    col4.metric("Valid employees", len(valid))
    col5.metric("Client queries", len(queries))

    st.markdown("**Questions for the client**")
    st.dataframe(
        queries, hide_index=True, width="stretch",
        column_config={
            "employee_id": st.column_config.TextColumn("Employee", width="small"),
            "issue": st.column_config.TextColumn("Issue", width="medium"),
            "question": st.column_config.TextColumn("Question for the client", width="large"),
            "effect": st.column_config.TextColumn("Effect on the valuation", width="medium"),
        },
    )
    st.caption("We ask instead of guessing, because a guessed date or salary would change that "
               "person's pension.")

    if results is None:
        return
    immediate = results[results["immediate_retirement"]]
    if not immediate.empty:
        st.markdown("**Valued as immediate retirement; please confirm status**")
        columns = ["employee_id", "age", "past_service", "yearly_salary", "earned_pension", "dbo"]
        st.dataframe(
            immediate[columns].sort_values("dbo", ascending=False), hide_index=True, width="stretch",
            column_config={
                "employee_id": st.column_config.TextColumn("Employee"),
                "age": st.column_config.NumberColumn("Age", format="%.1f"),
                "past_service": st.column_config.NumberColumn("Service (years)", format="%.1f"),
                "yearly_salary": st.column_config.NumberColumn("Yearly salary", format="€%,.0f"),
                "earned_pension": st.column_config.NumberColumn("Pension earned / year",
                                                                format="€%,.0f"),
                "dbo": st.column_config.NumberColumn("DBO", format="€%,.0f"),
            },
        )
        st.caption("These people are at or past the retirement age. They are valued as retiring "
                   "today, on their current salary.")


def pension_assumptions_section():
    """Show the assumption settings, each with one short caption, and return the assumptions."""
    st.header("Assumptions")
    col1, col2, col3, col4, col5 = st.columns(5)
    discount_pct = col1.slider("Discount rate (%)", 1.0, 6.0, 3.5, step=0.1)
    col1.caption("Interest rate used to value future payments.")
    growth_pct = col2.slider("Salary growth (%)", 0.0, 5.0, 2.5, step=0.1)
    col2.caption("How fast salaries rise until retirement.")
    retirement_age = col3.slider("Retirement age", 60, 67, 65)
    col3.caption("The age the pension starts.")
    accrual_pct = col4.slider("Accrual rate (%)", 0.5, 2.0, 1.0, step=0.1)
    col4.caption("Pension earned per year of service, as a share of final salary.")
    mortality_factor = col5.slider("Mortality factor", 0.8, 1.2, 1.0, step=0.05)
    col5.caption("Below 1 means people live longer.")

    col1, col2 = st.columns(2)
    payments_per_year = col1.selectbox("Payments per year", PAYMENT_FREQUENCIES, index=2)
    col1.caption("Austrian pensions are commonly paid 14 times a year.")
    increase_pct = col2.slider("Pension increase after retirement (%)", 0.0, 3.0, 0.0, step=0.1)
    col2.caption("Yearly rise once the pension has started.")

    # Sliders show percentages for readability; the model works with decimals (3.5% = 0.035)
    return dict(
        ASSUMPTIONS,
        discount_rate=discount_pct / 100,
        salary_growth=growth_pct / 100,
        retirement_age=retirement_age,
        accrual_rate=accrual_pct / 100,
        mortality_factor=mortality_factor,
        payments_per_year=payments_per_year,
        pension_increase=increase_pct / 100,
    )


def pension_uploads_section():
    """Show the two uploads; return (mortality upload, area for messages about it)."""
    handle_employee_upload()
    mortality_upload = st.file_uploader("Upload a mortality table (CSV with columns age, q_x)",
                                        type="csv", key="mortality_upload")
    status = st.container()
    st.caption("A table replaces the illustrative formula, and the mortality factor still scales "
               "it. Ages must be continuous and every q_x between 0 and 1. Official tables such "
               "as AVÖ 2018-P are licensed and not included.")
    return mortality_upload, status


def apply_uploaded_mortality(a, upload, valid, status):
    """Return the assumptions using an uploaded table, or unchanged if there is none or it fails."""
    if upload is None:
        return a

    try:
        # getvalue reads from the start each time, unlike a file that was already read once
        table_df = pd.read_csv(io.BytesIO(upload.getvalue()))
    except ValueError as error:
        with status:
            st.error(f"Could not read {upload.name}: {error}. Using the illustrative formula.")
        return a

    youngest_age, oldest_age = employee_age_range(valid, VALUATION_DATE)
    q_table, problems = validate_mortality_table(table_df, youngest_age, oldest_age)
    if problems:
        with status:
            st.error(f"{upload.name} was not used: " + " ".join(problems) +
                     " The illustrative formula is used instead.")
        return a

    with status:
        st.caption(f"Using {upload.name}, ages {min(q_table)} to {max(q_table)}.")
    return dict(a, q_table=q_table, max_age=max(q_table),
                mortality_source=f"Uploaded table: {upload.name}")


def mortality_formula(a):
    """Return the q_y formula in LaTeX for the mortality basis in use."""
    if a["q_table"] is not None:
        return r"q_y = m \cdot q_y^{\text{table}}"
    return r"q_y = m \cdot \left(A + B \, c^{y}\right)"


def mortality_substitution(age, a):
    """Return q_y for one whole age in LaTeX, with its numbers filled in."""
    q = death_prob(age, a)
    factor = a["mortality_factor"]
    if a["q_table"] is not None:
        return rf"q_{{{age}}} = {factor:g} \cdot {a['q_table'][age]:.5f} = {q:.5f}"
    return (rf"q_{{{age}}} = {factor:g} \cdot \left({a['makeham_A']:.4f} + {a['makeham_B']:.6f} "
            rf"\cdot {a['makeham_c']:g}^{{{age}}}\right) = {q:.5f}")


def survival_step_latex(r, a):
    """Return (symbolic, substituted) LaTeX for survival to retirement, using the model's method."""
    x, n = r["age"], r["years_to_retirement"]
    if r["immediate_retirement"]:
        return (r"n = 0 \quad \Rightarrow \quad {}_0p_x = 1",
                rf"x = {x:.2f} \geq R = {a['retirement_age']}, \qquad {{}}_0p_{{{x:.2f}}} = 1")

    R = a["retirement_age"]
    whole_age = int(x)
    part_year = x - whole_age
    q_first = death_prob(whole_age, a)
    product = survival_prob(whole_age, R, a)
    denominator = 1 - part_year * q_first
    symbolic = (rf"{{}}_np_x = \frac{{\prod_{{y=\lfloor x \rfloor}}^{{{R - 1}}} \left(1 - q_y\right)}}"
                r"{1 - t \, q_{\lfloor x \rfloor}}, \qquad t = x - \lfloor x \rfloor, \qquad "
                + mortality_formula(a))
    substituted = (mortality_substitution(whole_age, a) + rf", \qquad t = {part_year:.2f}, \qquad "
                   rf"{{}}_{{{n:.2f}}}p_{{{x:.2f}}} = \frac{{\prod_{{y={whole_age}}}^{{{R - 1}}}"
                   rf" (1 - q_y)}}{{1 - {part_year:.2f} \cdot {q_first:.5f}}} = "
                   rf"\frac{{{product:.4f}}}{{{denominator:.4f}}} = "
                   rf"{r['survival_to_retirement']:.4f}")
    return symbolic, substituted


def annuity_step_latex(r, a, contributions, age_text):
    """Return (symbolic, substituted) LaTeX for the annuity with instalments and increases."""
    last_k = len(contributions) - 1
    i, j, f = a["discount_rate"], a["pension_increase"], a["payments_per_year"]
    v = 1 / (1 + i)
    annual = rf"\ddot{{a}}_{{{age_text}}}"
    with_timing = rf"\ddot{{a}}^{{(f)}}_{{{age_text}}}"

    timing_formula = (rf"{with_timing} \approx {annual} - \frac{{f-1}}{{2f}} \left[1 + "
                      rf"\sum_{{k=1}}^{{{last_k}}} \left((1+j)^{{k}} - (1+j)^{{k-1}}\right) "
                      rf"v^{{k}} \, {{}}_kp_{{{age_text}}}\right] \quad \text{{(approximation)}}")

    symbolic = (r"\begin{gathered} "
                rf"{annual} = \sum_{{k=0}}^{{{last_k}}} v^{{k}} \, (1+j)^{{k}} \, "
                rf"{{}}_kp_{{{age_text}}}, \qquad v = \frac{{1}}{{1+i}} \\ {timing_formula}"
                r" \end{gathered}")

    terms = ["1"]
    for k in range(1, min(3, last_k + 1)):
        item = contributions.iloc[k]
        level = rf"{item['pension_level']:.4f} \cdot " if j > 0 else ""
        terms.append(rf"{item['discount']:.4f} \cdot {level}{item['survival']:.4f}")
    adjustment = r["timing_adjustment"]
    if j > 0:
        timing_line = (rf"{with_timing} \approx {r['annual_annuity']:.4f} - {-adjustment:.4f} "
                       rf"= {r['annuity_at_retirement']:.4f}, \qquad f = {f}, \quad j = {j:g}")
    else:
        timing_line = (rf"{with_timing} \approx {r['annual_annuity']:.4f} - "
                       rf"\frac{{{f} - 1}}{{2 \cdot {f}}} = {r['annual_annuity']:.4f} - "
                       rf"{-adjustment:.4f} = {r['annuity_at_retirement']:.4f}")

    substituted = (r"\begin{gathered} "
                   rf"{annual} = {' + '.join(terms)} + \cdots = {r['annual_annuity']:.4f}, \qquad "
                   rf"v = \frac{{1}}{{1 + {i:g}}} = {v:.4f} \\ {timing_line} \end{{gathered}}")
    return symbolic, substituted


def step_formulas(row, r, a, age_text, contributions):
    """Return each method step as (title, formula in symbols, the same with numbers)."""
    R = a["retirement_age"]
    x, s, n = r["age"], r["past_service"], r["years_to_retirement"]
    g, i = a["salary_growth"], a["discount_rate"]
    days_alive, days_employed = employee_day_counts(row, VALUATION_DATE)
    survival_symbolic, survival_substituted = survival_step_latex(r, a)
    annuity_symbolic, annuity_substituted = annuity_step_latex(r, a, contributions, age_text)

    # In an f-string, {value} inserts a number and {{ }} produces a literal LaTeX brace.
    # So S_{{{R}}} becomes S_{65}: two braces give "{", {R} gives 65, two braces give "}"
    return [
        ("Age and service",
         r"x = \frac{\text{days alive}}{365.25}, \qquad s = \frac{\text{days employed}}{365.25}, "
         r"\qquad n = \max(R - x,\, 0)",
         rf"x = \frac{{{tex_int(days_alive)}}}{{365.25}} = {x:.2f}, \qquad "
         rf"s = \frac{{{tex_int(days_employed)}}}{{365.25}} = {s:.2f}, \qquad "
         rf"n = \max({R} - {x:.2f},\, 0) = {n:.2f}"),
        ("Yearly salary",
         r"S = \text{monthly salary} \times 14",
         rf"S = {tex_eur(row['monthly_salary_eur'])} \times 14 = {tex_eur(r['yearly_salary'])}"),
        ("Projected final salary",
         rf"S_{{{age_text}}} = S \cdot (1+g)^{{n}}",
         rf"S_{{{age_text}}} = {tex_eur(r['yearly_salary'])} \cdot (1 + {g:g})^{{{n:.2f}}} "
         rf"= {tex_eur(r['projected_final_salary'])}"),
        ("Pension earned so far",
         rf"E = \text{{accrual}} \cdot s \cdot S_{{{age_text}}}",
         rf"E = {a['accrual_rate']:g} \cdot {s:.2f} \cdot {tex_eur(r['projected_final_salary'])} "
         rf"= {tex_eur(r['earned_pension'])}"),
        ("Survival to retirement", survival_symbolic, survival_substituted),
        ("Annuity factor", annuity_symbolic, annuity_substituted),
        ("DBO",
         rf"\text{{DBO}} = E \cdot \ddot{{a}}^{{(f)}}_{{{age_text}}} \cdot v^{{n}} \cdot {{}}_np_x",
         rf"\text{{DBO}} = {tex_eur(r['earned_pension'])} \cdot {r['annuity_at_retirement']:.4f} "
         rf"\cdot {r['discount_factor']:.4f} \cdot {r['survival_to_retirement']:.4f} "
         rf"= {tex_eur(r['dbo'])}, \qquad v^{{{n:.2f}}} = \frac{{1}}{{(1 + {i:g})^{{{n:.2f}}}}} "
         rf"= {r['discount_factor']:.4f}"),
    ]


def pension_employee_section(valued, results, a):
    """Show one employee's calculation as a short table, with the formulas collapsed; return it."""
    st.header("One employee, step by step")
    employee_id = st.selectbox("Employee", valued["employee_id"])
    row = valued[valued["employee_id"] == employee_id].iloc[0]
    r = results[results["employee_id"] == employee_id].iloc[0]

    # Someone past the retirement age retires now, so their annuity starts at their exact age
    immediate = bool(r["immediate_retirement"])
    retirement_age_used = r["retirement_age_used"]
    age_text = f"{retirement_age_used:.2f}" if immediate else f"{a['retirement_age']}"

    steps = pd.DataFrame({
        "Step": ["Yearly salary", "Projected final salary", "Pension earned so far",
                 f"Annuity factor at {age_text}", "Discount factor", f"Survival to {age_text}", "DBO"],
        "Value": [eur(r["yearly_salary"]), eur(r["projected_final_salary"]), eur(r["earned_pension"]),
                  f"{r['annuity_at_retirement']:.4f}", f"{r['discount_factor']:.4f}",
                  f"{r['survival_to_retirement']:.4f}", eur(r["dbo"])],
    })
    st.dataframe(steps, hide_index=True, width="stretch")

    with st.expander("Show formulas"):
        st.caption("Values are rounded for display, so multiplying them by hand can differ by a "
                   "few euros.")
        contributions = cached_annuity_contributions(retirement_age_used, a)
        for title, symbolic, substituted in step_formulas(row, r, a, age_text, contributions):
            st.markdown(f"**{title}**")
            st.latex(symbolic)
            st.latex(substituted)
    return r


def pension_sensitivity_section(sensitivity, valued, a):
    """Show the sensitivity table, the tornado chart and duration as one number."""
    st.header("Sensitivities")
    st.dataframe(
        sensitivity, hide_index=True, width="stretch",
        column_config={
            "scenario": st.column_config.TextColumn("Scenario"),
            "assumption_value": st.column_config.NumberColumn("New value", format="%.3f"),
            "total_dbo": st.column_config.NumberColumn("Total DBO", format="€%,.0f"),
            "pct_change": st.column_config.NumberColumn("% change", format="%+.1f%%"),
        },
    )
    st.plotly_chart(sensitivity_chart(sensitivity), width="stretch")

    duration_years = duration(sensitivity)
    down_pct = cached_rate_move(valued, a, -0.01)[1]
    st.metric("Duration", f"{duration_years:.1f} years")
    st.caption(f"A small change in the discount rate moves the total by about {duration_years:.1f}% "
               f"per percentage point, but a full 1-point fall changes it by {down_pct:+.1f}%, "
               "because the effect is not a straight line.")


def pension_all_employees_section(results):
    """Show one row per employee, largest DBO first."""
    columns = ["employee_id", "age", "past_service", "yearly_salary", "dbo", "service_cost"]
    st.dataframe(
        results[columns].sort_values("dbo", ascending=False), hide_index=True, width="stretch",
        column_config={
            "employee_id": st.column_config.TextColumn("Employee"),
            "age": st.column_config.NumberColumn("Age", format="%.1f"),
            "past_service": st.column_config.NumberColumn("Service (years)", format="%.1f"),
            "yearly_salary": st.column_config.NumberColumn("Yearly salary", format="€%,.0f"),
            "dbo": st.column_config.NumberColumn("DBO", format="€%,.0f"),
            "service_cost": st.column_config.NumberColumn("Service cost", format="€%,.0f"),
        },
    )


def pension_costs_section(valued, a, r, total, outputs):
    """Show next year's service cost and interest cost, with the formulas collapsed."""
    st.header("Service cost and interest cost")
    service_cost = outputs["service_cost"]
    age_text = f"{r['retirement_age_used']:.2f}" if r["immediate_retirement"] else f"{a['retirement_age']}"

    col1, col2 = st.columns(2)
    col1.metric("Service cost, next year", eur(service_cost))
    col1.caption("Pension earned by working one more year.")
    col2.metric("Interest cost, next year", eur(outputs["interest_cost"]))
    col2.caption("One year of interest on the obligation.")

    with st.expander("Show formulas"):
        st.latex(rf"SC = \text{{accrual}} \cdot \min(1,\, n) \cdot S_{{{age_text}}} \cdot "
                 rf"\ddot{{a}}^{{(f)}}_{{{age_text}}} \cdot v^{{n}} \cdot {{}}_np_x")
        st.latex(rf"SC_{{\text{{{r['employee_id']}}}}} = {a['accrual_rate']:g} \cdot "
                 rf"{r['service_next_year']:.2f} \cdot {tex_eur(r['projected_final_salary'])} \cdot "
                 rf"{r['annuity_at_retirement']:.4f} \cdot {r['discount_factor']:.4f} \cdot "
                 rf"{r['survival_to_retirement']:.4f} = {tex_eur(r['service_cost'])}")
        st.caption("The total adds this up over all employees. Someone retiring within the year "
                   "earns only part of a year.")
        st.latex(r"IC = i \cdot \left(\text{DBO} + SC\right)")
        st.latex(rf"IC = {a['discount_rate']:g} \cdot \left({tex_eur(total)} + "
                 rf"{tex_eur(service_cost)}\right) = {tex_eur(outputs['interest_cost'])}")
        st.caption(f"Simplified: service cost counts from the start of the year, and the "
                   f"{eur(outputs['payments_next_year'])} of pensions due next year are not "
                   f"deducted, which would lower interest by about "
                   f"{eur(outputs['interest_on_payments'])}.")


# Which model checks stand behind each sentence. All checks still run; these are the ones shown
SANITY_CHECKS = [
    (["1"], "All future pension payments, discounted to today, add up to the total."),
    (["3"], "Someone with no service gets no pension."),
    (["4"], "Doubling the pension rate doubles the result."),
    (["5a", "5b"], "Higher interest rate gives a lower value; higher salary growth gives a higher value."),
    (["12"], "Next year's value equals today's value plus the pension earned this year plus "
             "interest, for employees not yet retired."),
]


def pension_sanity_section(checks):
    """Show the plain list of sanity checks, each with a tick or a cross."""
    st.header("Sanity checks")
    passed_by_number = {name.split()[0]: check_passed for name, check_passed, _ in checks}
    lines = []
    for numbers, sentence in SANITY_CHECKS:
        all_passed = all(passed_by_number[number] for number in numbers)
        lines.append(f"- {'✓' if all_passed else '✗'} {sentence}")
    st.markdown("\n".join(lines))


def notation_table():
    """Return the notation table as Markdown."""
    valuation_text = f"{pd.Timestamp(VALUATION_DATE):%d.%m.%Y}"
    return (
        "| Symbol | Meaning |\n"
        "|---|---|\n"
        f"| $x$ | exact age on {valuation_text} |\n"
        "| $s$ | past service in years |\n"
        "| $R$ | retirement age |\n"
        "| $n$ | years until retirement, $\\max(R - x, 0)$ |\n"
        "| $S$ | yearly salary today (monthly × 14) |\n"
        "| $g$ | yearly salary growth |\n"
        "| $i$ | discount rate |\n"
        "| $v$ | one-year discount factor, $1/(1+i)$ |\n"
        "| $j$ | yearly pension increase after retirement |\n"
        "| $f$ | payments per year |\n"
        "| $k$ | whole years since the pension started |\n"
        "| $y$ | a whole age |\n"
        "| $t$ | part of a year of age, $x - \\lfloor x \\rfloor$ |\n"
        "| $q_y$ | chance of dying within one year from whole age $y$ |\n"
        "| $m$ | mortality factor, which scales every $q_y$ |\n"
        f"| $A$, $B$, $c$ | values in the mortality formula: {ASSUMPTIONS['makeham_A']:g}, "
        f"{ASSUMPTIONS['makeham_B']:.6f}, {ASSUMPTIONS['makeham_c']:g} |\n"
        "| ${}_np_x$ | chance of surviving $n$ years from exact age $x$ |\n"
        "| $\\ddot{a}_{R}$ | value at age $R$ of 1 euro a year for life, paid at the start of each year |\n"
        "| $\\ddot{a}^{(f)}_{R}$ | the same pension paid in $f$ instalments a year |\n"
        "| $E$ | pension earned so far, per year |\n"
        "| $SC$ | service cost for the next year |\n"
        "| $IC$ | interest cost for the next year |"
    )


def annuity_chart(a):
    """Return the chart of what each yearly payment adds to the annuity factor at retirement."""
    R = a["retirement_age"]
    contributions = cached_annuity_contributions(R, a)
    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(go.Bar(x=contributions["payment_age"], y=contributions["contribution"],
                           name="Value of each payment"), secondary_y=False)
    chart.add_trace(go.Scatter(x=contributions["payment_age"], y=contributions["running_total"],
                               name="Running total", mode="lines"), secondary_y=True)
    chart.update_layout(title=f"Annuity factor at {R}, before the adjustment for "
                              f"{a['payments_per_year']} payments a year: "
                              f"{contributions['running_total'].iloc[-1]:.4f}")
    chart.update_xaxes(title_text="Payment age")
    chart.update_yaxes(title_text="Value of each payment", secondary_y=False)
    chart.update_yaxes(title_text="Running total", secondary_y=True)
    return chart


def cash_flow_chart(flows):
    """Return the chart of expected pension payments by calendar year."""
    chart = go.Figure()
    chart.add_trace(go.Bar(x=flows["year"], y=flows["expected_payment"],
                           name="Expected payments"))
    chart.add_trace(go.Bar(x=flows["year"], y=flows["timing_adjustment"],
                           name="Payment timing adjustment"))
    chart.add_trace(go.Scatter(x=flows["year"], y=flows["discounted_payment"],
                               name="Discounted to today", mode="lines"))
    chart.update_layout(title="Expected pension payments by calendar year",
                        xaxis_title="Year", yaxis_title="EUR", barmode="relative")
    return chart


def pension_details_section(valued, a):
    """Show the benefit rule, time basis, notation table, annuity chart and cash-flow chart."""
    st.markdown("**The pension promise**")
    st.latex(r"E = \text{accrual} \times \text{service} \times \text{final yearly salary}")
    st.caption("Paid every year for life from the retirement age. Yearly salary is the monthly "
               "salary times 14. Age, service and discounting use exact years (days ÷ 365.25). "
               "Death rates are by whole age, with deaths spread evenly within each year.")

    st.markdown("**Notation**")
    st.markdown(notation_table())

    R = a["retirement_age"]
    st.plotly_chart(annuity_chart(a), width="stretch")
    st.caption(f"Each bar is what one year's pension payment is worth at {R}. Later years count "
               "less because the person may have died and the money is further in the future. "
               "All bars together add up to the annuity factor.")

    st.plotly_chart(cash_flow_chart(cached_cash_flows(valued, a)), width="stretch")
    st.caption("The payment timing adjustment allows for the pension being paid in instalments "
               "through the year. The discounted line adds up to the total DBO.")


def pension_monte_carlo_section(valued, a):
    """Show the distribution of simulated total DBOs and its percentiles."""
    st.warning("Exploratory, work in progress: assumptions are drawn independently, so the range "
               "is likely too wide.")
    totals = cached_dbo_draws(valued, a)
    percentiles = dbo_percentiles(totals)

    chart = px.histogram(x=totals, nbins=40, labels={"x": "Total DBO (EUR)"},
                         title="500 simulated total DBOs")
    for name in ["p10", "p50", "p90"]:
        chart.add_vline(x=percentiles[name], line_dash="dash", annotation_text=name.upper())
    chart.update_layout(yaxis_title="Number of draws", showlegend=False)
    st.plotly_chart(chart, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("P10", eur(percentiles["p10"]))
    col2.metric("P50", eur(percentiles["p50"]))
    col3.metric("P90", eur(percentiles["p90"]))
    st.caption("Each draw picks a discount rate and a salary growth rate around the chosen values "
               "(standard deviation 0.5 points each) and recalculates the total.")


def pension_downloads_section(raw, valid, results, queries, outputs, total, duration_years, a):
    """Offer the run record and the full workings as downloads."""
    # Taken fresh on every run of the page, so the record says when these numbers were produced
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    record = run_record(raw, valid, outputs, total, duration_years, a, VALUATION_DATE, timestamp)
    workbook = excel_workbook(employee_export_table(results, valid), assumptions_table(a), record,
                              queries)

    col1, col2 = st.columns(2)
    col1.download_button("Download run record (CSV)", data=record.to_csv(index=False),
                         file_name="pension_run_record.csv", mime="text/csv")
    col2.download_button(
        "Download workbook (Excel)", data=workbook, file_name="pension_valuation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def pension_simplified_section():
    """List what this model leaves out."""
    st.markdown(
        "- Illustrative mortality formula. A real valuation would use AVÖ 2018-P.\n"
        "- One table for men and women.\n"
        "- No staff leaving before retirement.\n"
        "- One flat discount rate.\n"
        "- Active employees only.\n"
        "- No survivor pensions."
    )


def page_pension():
    """Show the pension valuation: key numbers first, supporting detail collapsed at the bottom."""
    st.title("Pension valuation")
    st.markdown("How much a company owes today for the pensions its staff have earned so far (the "
                "DBO). Fictional Austrian company, synthetic data.")

    if "employees_raw" not in st.session_state:
        load_employees_into_session(pd.read_csv(EMPLOYEES_FILE), "sample employees")

    # The page reads top to bottom, but the numbers at the top depend on settings and uploads
    # further down. So every place is reserved here in display order, then filled below
    headline = st.container()
    data_area = st.container()
    assumptions_area = st.container()
    employee_area = st.container()
    sensitivity_area = st.container()
    all_employees = st.expander("All employees")
    costs_area = st.container()
    sanity_area = st.container()
    details = st.expander("Calculation details")
    exploratory = st.expander("Exploratory, still in progress")
    downloads = st.expander("Downloads")
    own_data = st.expander("Use your own data")
    simplified = st.expander("What I simplified")

    with simplified:
        pension_simplified_section()
    with own_data:
        mortality_upload, mortality_status = pension_uploads_section()
    with assumptions_area:
        a = pension_assumptions_section()

    raw = st.session_state["employees_raw"]
    valid, log_df, queries = cached_validate(raw, a["retirement_age"])
    if valid.empty:
        with data_area:
            pension_data_section(raw, log_df, valid, queries, None)
            st.error("No employees to value with these settings.")
        return

    a = apply_uploaded_mortality(a, mortality_upload, valid, mortality_status)
    if a["q_table"] is not None:
        with assumptions_area:
            st.caption(f"Mortality: {a['mortality_source']}.")

    results, total = cached_dbo(valid, a)
    sensitivity = cached_sensitivity(valid, a)
    duration_years = duration(sensitivity)
    outputs = cached_standard_outputs(results, valid, a)

    with headline:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total DBO", eur(total))
        col2.metric("Employees valued", len(valid))
        col3.metric("Duration", f"{duration_years:.1f} years")
        col4.metric("Open client queries", len(queries))
    with data_area:
        pension_data_section(raw, log_df, valid, queries, results)
    with employee_area:
        selected = pension_employee_section(valid, results, a)
    with sensitivity_area:
        pension_sensitivity_section(sensitivity, valid, a)
    with all_employees:
        pension_all_employees_section(results)
    with costs_area:
        pension_costs_section(valid, a, selected, total, outputs)
    with sanity_area:
        pension_sanity_section(cached_model_checks(valid, a))
    with details:
        pension_details_section(valid, a)
    with exploratory:
        pension_monte_carlo_section(valid, a)
    with downloads:
        pension_downloads_section(raw, valid, results, queries, outputs, total, duration_years, a)


init_session_state()
keep_widget_settings()

page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption("All data is synthetic and this tool is a student project, not investment advice.")

if page == "Data cleaning":
    page_data_cleaning()
elif page == "Portfolio":
    page_portfolio()
elif page == "Risk":
    page_risk()
elif page == "Update":
    page_update()
else:
    page_pension()
