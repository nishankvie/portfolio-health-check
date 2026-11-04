"""Streamlit app for the Portfolio Health Check (synthetic data, student project)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    DAYS_PER_YEAR,
    VALUATION_DATE,
    dbo_monte_carlo,
    dbo_portfolio,
    duration,
    explain_employee,
    sensitivity_table,
    validate_employees,
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


def load_employees_into_session(raw_df, source_name):
    """Validate raw employee data and store raw, valid, log and queries in session state."""
    # Validate first: if it raises, the previously stored employee data stays untouched
    valid, log_df, query_df = validate_employees(raw_df)
    st.session_state["employees_raw"] = raw_df
    st.session_state["employees_valid"] = valid
    st.session_state["employees_queries"] = query_df
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
        st.success(f"Validated {uploaded.name}.")
    except ValueError as error:
        st.error(f"Could not validate {uploaded.name}: {error}")


def pension_assumption_controls():
    """Show the assumption sliders and return a full assumptions dictionary."""
    col1, col2, col3, col4, col5 = st.columns(5)
    discount_pct = col1.slider("Discount rate (%)", 1.0, 6.0, 3.5, step=0.1)
    growth_pct = col2.slider("Salary growth (%)", 0.0, 5.0, 2.5, step=0.1)
    retirement_age = col3.slider("Retirement age", 60, 67, 65)
    accrual_pct = col4.slider("Accrual rate (%)", 0.5, 2.0, 1.0, step=0.1)
    mortality_factor = col5.slider("Mortality factor", 0.8, 1.2, 1.0, step=0.05)

    # Sliders show percentages for readability; the model works with decimals (3.5% = 0.035)
    return dict(
        ASSUMPTIONS,
        discount_rate=discount_pct / 100,
        salary_growth=growth_pct / 100,
        retirement_age=retirement_age,
        accrual_rate=accrual_pct / 100,
        mortality_factor=mortality_factor,
    )


def below_retirement_age(valid_df, retirement_age):
    """Return the employees younger than the chosen retirement age at the valuation date."""
    age = (pd.Timestamp(VALUATION_DATE) - valid_df["birth_date"]).dt.days / DAYS_PER_YEAR
    return valid_df[age < retirement_age].reset_index(drop=True)


@st.cache_data
def cached_dbo(valid_df, assumptions):
    """Return (per-employee DBO table, total DBO) for one set of assumptions (cached)."""
    return dbo_portfolio(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_sensitivity(valid_df, assumptions):
    """Return the sensitivity table around the given assumptions (cached)."""
    return sensitivity_table(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_dbo_monte_carlo(valid_df, assumptions):
    """Return P10/P50/P90 of total DBO under uncertain rates (cached; about 5 seconds)."""
    return dbo_monte_carlo(valid_df, VALUATION_DATE, a=assumptions)


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


def page_pension():
    """Show the pension page: data validation, DBO, explanation, sensitivity and Monte Carlo."""
    st.title("Pension")
    st.markdown(
        "Values a fictional Austrian defined benefit plan (1% x years of service x final yearly "
        "salary from 65, salary paid 14 times a year) with the projected unit credit method."
    )

    if "employees_raw" not in st.session_state:
        load_employees_into_session(pd.read_csv(EMPLOYEES_FILE), "sample employees")
    handle_employee_upload()

    raw = st.session_state["employees_raw"]
    valid = st.session_state["employees_valid"]
    queries = st.session_state["employees_queries"]
    st.caption(f"Showing: {st.session_state['employees_source']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw rows", len(raw))
    col2.metric("Valid employees", len(valid))
    col3.metric("Client queries", len(queries))

    st.subheader("Client queries")
    st.dataframe(queries, hide_index=True, width="stretch")
    st.download_button("Download client queries", data=queries.to_csv(index=False),
                       file_name="client_queries.csv", mime="text/csv")

    st.subheader("Valuation")
    assumptions = pension_assumption_controls()

    valued = below_retirement_age(valid, assumptions["retirement_age"])
    excluded = len(valid) - len(valued)
    if excluded > 0:
        st.warning(f"{excluded} valid employee(s) are already at or above retirement age "
                   f"{assumptions['retirement_age']} and are left out: their pension would be "
                   "valued as already in payment, which this model doesn't cover.")
    if valued.empty:
        st.error("No employees to value with these settings.")
        return

    results, total = cached_dbo(valued, assumptions)
    sensitivity = cached_sensitivity(valued, assumptions)
    col1, col2 = st.columns(2)
    col1.metric("Total DBO", f"€{total:,.0f}")
    col2.metric("Duration", f"{duration(sensitivity):.1f} years")

    st.subheader("Step by step")
    employee_id = st.selectbox("Employee", valued["employee_id"])
    row = valued[valued["employee_id"] == employee_id].iloc[0]
    steps = pd.DataFrame(explain_employee(row, VALUATION_DATE, assumptions),
                         columns=["Step", "Value"])
    st.dataframe(steps, hide_index=True, width="stretch")

    st.subheader("Sensitivity")
    st.dataframe(
        sensitivity, hide_index=True, width="stretch",
        column_config={
            "assumption_value": st.column_config.NumberColumn("New value", format="%.3f"),
            "total_dbo": st.column_config.NumberColumn("Total DBO", format="€%.0f"),
            "pct_change": st.column_config.NumberColumn("% change", format="%+.1f%%"),
        },
    )
    st.plotly_chart(sensitivity_chart(sensitivity), width="stretch")

    st.subheader("Monte Carlo")
    percentiles = cached_dbo_monte_carlo(valued, assumptions)
    col1, col2, col3 = st.columns(3)
    col1.metric("P10", f"€{percentiles['p10']:,.0f}")
    col2.metric("P50", f"€{percentiles['p50']:,.0f}")
    col3.metric("P90", f"€{percentiles['p90']:,.0f}")

    st.caption(
        "Assumptions and simplifications: illustrative Makeham mortality (a real Austrian "
        "valuation would use the AVÖ 2018-P tables); turnover and death benefits before retirement "
        "ignored; only the DBO is shown, with no service cost or interest cost; Monte Carlo draws "
        "the discount rate and salary growth independently (std 0.5 points each, 500 draws); "
        "employees with open client queries are not valued; synthetic data."
    )


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
