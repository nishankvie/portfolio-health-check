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
    LIFE_EXPECTANCY_AGE,
    LIFE_EXPECTANCY_SCENARIO,
    MORTALITY_REFERENCE_AGE,
    PAYMENT_FREQUENCIES,
    VALUATION_DATE,
    annuity_contributions,
    annuity_values,
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
    life_expectancy,
    mortality_factor_for_extra_life,
    mortality_table,
    run_model_checks,
    run_record,
    sensitivity_table,
    standard_outputs,
    survival_curve,
    survival_prob,
    top_n_share,
    validate_employees,
    validate_mortality_table,
    weighted_average_payment_time,
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
        st.success(f"Validated {uploaded.name}.")
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
def cached_mortality_table(assumptions, from_age):
    """Return q_x and survival from from_age up to the maximum age (cached)."""
    return mortality_table(assumptions, from_age=from_age, to_age=assumptions["max_age"])


@st.cache_data
def cached_survival_curve(exact_age, assumptions):
    """Return survival from an exact age, by the same method as the valuation (cached)."""
    return survival_curve(exact_age, assumptions)


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
def cached_extra_life_factor(assumptions):
    """Return the mortality factor that adds one year to life expectancy at 65 (cached)."""
    return mortality_factor_for_extra_life(1.0, LIFE_EXPECTANCY_AGE, assumptions)


@st.cache_data
def cached_model_checks(valid_df, assumptions):
    """Return the model checks for one set of data and assumptions (cached)."""
    return run_model_checks(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_cash_flows(valid_df, assumptions):
    """Return expected yearly pension payments, undiscounted and discounted (cached)."""
    return expected_cash_flows(valid_df, VALUATION_DATE, assumptions)


@st.cache_data
def cached_average_payment_time(valid_df, assumptions):
    """Return the present-value-weighted average years until payment (cached)."""
    return weighted_average_payment_time(valid_df, VALUATION_DATE, assumptions)


def step_card(number, title, symbolic, substituted, explanation):
    """Show one method step: formula in symbols, the same with real numbers, and a sentence."""
    with st.container(border=True):
        st.markdown(f"**Step {number}. {title}**")
        st.latex(symbolic)
        st.latex(substituted)
        st.caption(explanation)


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


def model_checks_card(checks):
    """Show how many model checks passed, with every check and its detail in an expander."""
    passed = sum(1 for _, check_passed, _ in checks if check_passed)
    message = f"{passed} of {len(checks)} checks passed"
    with st.container(border=True):
        st.markdown("**Model checks**")
        if passed == len(checks):
            st.success(message)
        else:
            st.error(message)
        with st.expander("Show every check and its detail"):
            for name, check_passed, detail in checks:
                st.markdown(f"**{'Passed' if check_passed else 'FAILED'}: {name}**")
                st.caption(detail)


def pension_promise_section():
    """Explain the benefit formula, the 14-salaries rule and a worked example."""
    st.header("The pension promise")
    with st.container(border=True):
        st.markdown("From the retirement age, every year for life, the employee receives:")
        st.latex(r"E = \text{accrual} \times \text{service} \times \text{final yearly salary}")
        st.markdown(
            "In Austria, employees are normally paid **14 monthly salaries a year** (12 regular "
            "payments plus holiday pay in summer and Christmas pay in winter), so the yearly "
            "salary is the monthly salary times 14, not 12."
        )
        st.latex(r"S = \text{monthly salary} \times 14")
        st.latex(r"\text{Example: } 0.01 \times 30 \times (\text{€}5{,}000 \times 14) "
                 r"= \text{€}21{,}000 \text{ a year for life}")
        st.caption("30 years of service and a final monthly salary of €5,000 with a 1% accrual rate.")


def pension_data_section(retirement_age):
    """Show upload, validation counts, the client query list and why we query; return the results."""
    st.header("Data and validation")
    with st.container(border=True):
        handle_employee_upload()
        st.caption(f"Showing: {st.session_state['employees_source']}")

        raw = st.session_state["employees_raw"]
        # Validated on every run because the retirement-age slider changes who needs a query
        valid, log_df, queries = cached_validate(raw, retirement_age)
        issues = log_df["issue"] if not log_df.empty else pd.Series(dtype=str)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Raw rows", len(raw))
        col2.metric("Duplicates removed", int((issues == "exact duplicate").sum()))
        col3.metric("Text salaries converted", int((issues == "text number").sum()))
        col4.metric("Valid employees", len(valid))
        col5.metric("Client queries", len(queries))

    with st.container(border=True):
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
        st.download_button("Download client queries", data=queries.to_csv(index=False),
                           file_name="client_queries.csv", mime="text/csv")
        st.caption(
            "Why we ask instead of estimating: each row is one person's individual pension "
            "promise, and an employee record has no neighbouring values to borrow from (unlike a "
            "company's previous and next month). A guessed birth date or salary would directly "
            "change that person's liability, so only the client can supply the correct value. "
            "Employees with a missing or impossible value are not valued until the client answers. "
            "Employees at or past the retirement age are valued as retiring on the valuation date "
            "while the client confirms their status."
        )
    return valid, queries


def pension_assumptions_section():
    """Show the assumption settings and notation; return (assumptions, mortality upload, status area)."""
    st.header("Assumptions")
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        discount_pct = col1.slider("Discount rate (%)", 1.0, 6.0, 3.5, step=0.1)
        col1.caption("Higher rate, lower DBO: future payments are worth less in today's money.")
        growth_pct = col2.slider("Salary growth (%)", 0.0, 5.0, 2.5, step=0.1)
        col2.caption("Higher growth, higher DBO: the pension is based on the final salary.")
        retirement_age = col3.slider("Retirement age", 60, 67, 65)
        col3.caption("Later retirement, usually lower DBO: payments start later and are "
                     "discounted for longer, though more salary growth pushes the other way.")
        accrual_pct = col4.slider("Accrual rate (%)", 0.5, 2.0, 1.0, step=0.1)
        col4.caption("Higher accrual, higher DBO, in exact proportion.")
        mortality_factor = col5.slider("Mortality factor", 0.8, 1.2, 1.0, step=0.05)
        col5.caption("Below 1, people live longer: more payments, higher DBO (longevity risk).")

        col1, col2 = st.columns(2)
        payments_per_year = col1.selectbox("Payments per year", PAYMENT_FREQUENCIES, index=2)
        col1.caption("Austrian pensions are commonly paid 14 times a year.")
        increase_pct = col2.slider("Pension increase after retirement (%)", 0.0, 3.0, 0.0, step=0.1)
        col2.caption("Default 0%: once it starts, the pension stays the same in euros.")

    with st.container(border=True):
        st.markdown("**Mortality basis (optional table upload)**")
        mortality_upload = st.file_uploader("Upload a mortality table (CSV with columns age, q_x)",
                                            type="csv", key="mortality_upload")
        status = st.container()
        st.caption(
            "Without an upload, the model uses an illustrative Makeham-style formula applied "
            "directly to q_x, calibrated so that unisex life expectancy at 65 is roughly 20 years. "
            "An uploaded table replaces the formula, and the mortality factor still scales it. "
            "Ages must be continuous and every q_x must be between 0 and 1. Official tables such "
            "as AVÖ 2018-P are licensed and are not included in this tool."
        )

    with st.container(border=True):
        st.markdown(
            "**Time basis.** The whole model uses one rule. Age, past service, years to "
            "retirement, salary growth and discounting use exact years (days between dates ÷ "
            "365.25) at the valuation date. Death probabilities are for whole ages, and deaths "
            "are assumed to be spread evenly within each year of age (uniform distribution of "
            "deaths). Yearly pension dates count from the exact retirement date. Only the "
            "minimum-working-age query uses completed years, because that rule is a legal age."
        )

    with st.container(border=True):
        st.markdown(
            "**What this model assumes about the benefit**\n"
            "- No pension increases after retirement, unless a rate is set above.\n"
            "- One unisex mortality table for men and women.\n"
            "- No survivor benefits: the pension stops when the pensioner dies.\n"
            "- No benefit on death before retirement, and nobody leaves before retirement."
        )

    valuation_text = f"{pd.Timestamp(VALUATION_DATE):%d.%m.%Y}"
    with st.container(border=True):
        st.markdown("**Notation used below**")
        st.markdown(
            "| Symbol | Meaning |\n"
            "|---|---|\n"
            f"| $x$ | exact age at the valuation date ({valuation_text}), in years |\n"
            "| $s$ | past service: exact years worked so far |\n"
            "| $R$ | retirement age (chosen above) |\n"
            "| $n$ | years until retirement, $\\max(R - x, 0)$ |\n"
            "| $S$ | yearly salary today (monthly × 14) |\n"
            "| $g$ | yearly salary growth |\n"
            "| $i$ | discount rate |\n"
            "| $v$ | one-year discount factor, $1/(1+i)$ |\n"
            "| $j$ | yearly pension increase after retirement |\n"
            "| $f$ | payments per year (textbooks write this as $m$; here $m$ is the mortality factor) |\n"
            "| $y$ | a whole age |\n"
            "| $t$ | part of a year of age, $x - \\lfloor x \\rfloor$ |\n"
            "| $q_y$ | probability of dying within one year from whole age $y$ |\n"
            "| $m$ | mortality factor, which scales every $q_y$ |\n"
            f"| $A$ | Makeham-style constant risk: {ASSUMPTIONS['makeham_A']:g} (not used with an uploaded table) |\n"
            f"| $B$ | Makeham-style age-related scale: {ASSUMPTIONS['makeham_B']:.6f} (not used with an uploaded table) |\n"
            f"| $c$ | Makeham-style yearly growth in the age-related risk: {ASSUMPTIONS['makeham_c']:g} (not used with an uploaded table) |\n"
            "| $k$ | whole years since the pension started |\n"
            "| ${}_np_x$ | probability of surviving $n$ years from exact age $x$ |\n"
            "| $\\ddot{a}_{R}$ | value at age $R$ of 1 euro a year for life, paid at the start of each year |\n"
            "| $\\ddot{a}^{(f)}_{R}$ | the same pension paid in $f$ instalments a year |\n"
            "| $E$ | earned pension per year |\n"
            "| $e_{65}$ | life expectancy at 65 |\n"
            "| $m^{*}$ | mortality factor that adds one year to $e_{65}$ |\n"
            "| $SC$ | current service cost for the next year |\n"
            "| $IC$ | interest cost for the next year |"
        )

    # Sliders show percentages for readability; the model works with decimals (3.5% = 0.035)
    assumptions = dict(
        ASSUMPTIONS,
        discount_rate=discount_pct / 100,
        salary_growth=growth_pct / 100,
        retirement_age=retirement_age,
        accrual_rate=accrual_pct / 100,
        mortality_factor=mortality_factor,
        payments_per_year=payments_per_year,
        pension_increase=increase_pct / 100,
    )
    return assumptions, mortality_upload, status


def apply_uploaded_mortality(a, upload, valid, status):
    """Return the assumptions using an uploaded table, or unchanged if there is none or it fails."""
    if upload is None:
        return a

    try:
        # getvalue reads from the start each time, unlike a file that was already read once
        table_df = pd.read_csv(io.BytesIO(upload.getvalue()))
    except ValueError as error:
        with status:
            st.error(f"Could not read {upload.name}: {error}. The illustrative formula is used.")
        return a

    youngest_age, oldest_age = employee_age_range(valid, VALUATION_DATE)
    q_table, problems = validate_mortality_table(table_df, youngest_age, oldest_age)
    if problems:
        with status:
            st.error(f"{upload.name} was not used: " + " ".join(problems) +
                     " The illustrative formula is used instead.")
        return a

    with status:
        st.success(f"Using uploaded mortality table {upload.name}, ages {min(q_table)} to "
                   f"{max(q_table)}.")
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


def survival_step_text(r, a):
    """Return the plain-English explanation of the survival step for one employee."""
    if r["immediate_retirement"]:
        return "The pension starts on the valuation date, so no survival is needed before it starts."
    source = ("q comes from the uploaded table, scaled by the mortality factor m."
              if a["q_table"] is not None else
              "q comes from a Makeham-style formula applied directly to q_x, where A is a constant "
              "risk, B · c^y grows with age, and m is the mortality factor.")
    return (f"The chance of reaching retirement: survive each whole year of age in turn, using "
            f"whole-age death probabilities, after removing the part of age {int(r['age'])} already "
            "lived, with deaths spread evenly within the year. The survival chart below uses this "
            f"same calculation. {source}")


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


def pension_method_section(valued, results, a):
    """Show the 7 projected unit credit steps for one selected employee; return its result row."""
    st.header("Method: projected unit credit, step by step")
    st.caption("Values are shown rounded, while the model keeps full precision, so multiplying the "
               "displayed figures by hand can differ from the result by a few euros.")
    employee_id = st.selectbox("Employee", valued["employee_id"])
    row = valued[valued["employee_id"] == employee_id].iloc[0]
    r = results[results["employee_id"] == employee_id].iloc[0]

    R = a["retirement_age"]
    immediate = bool(r["immediate_retirement"])
    x, s, n = r["age"], r["past_service"], r["years_to_retirement"]
    g, i = a["salary_growth"], a["discount_rate"]
    days_alive, days_employed = employee_day_counts(row, VALUATION_DATE)
    # Someone past the retirement age retires now, so their annuity starts at their exact age
    retirement_age_used = r["retirement_age_used"]
    age_text = f"{retirement_age_used:.2f}" if immediate else f"{R}"
    contributions = cached_annuity_contributions(retirement_age_used, a)

    if immediate:
        st.warning(f"{employee_id} is at or past the retirement age of {R}, so they are valued as "
                   "retiring on the valuation date: current salary and an immediate annuity at "
                   "their exact age. Please confirm their status with the client.")

    # In an f-string, {value} inserts a number and {{ }} produces a literal LaTeX brace.
    # So S_{{{R}}} becomes S_{65}: two braces give "{", {R} gives 65, two braces give "}"
    step_card(
        1, "Age and service",
        r"x = \frac{\text{days alive}}{365.25}, \qquad s = \frac{\text{days employed}}{365.25}, "
        r"\qquad n = \max(R - x,\, 0)",
        rf"x = \frac{{{tex_int(days_alive)}}}{{365.25}} = {x:.2f}, \qquad "
        rf"s = \frac{{{tex_int(days_employed)}}}{{365.25}} = {s:.2f}, \qquad "
        rf"n = \max({R} - {x:.2f},\, 0) = {n:.2f}",
        f"{employee_id} is {x:.2f} years old and has worked {s:.2f} years. "
        + ("They are treated as retiring on the valuation date. " if immediate
           else f"They retire in {n:.2f} years. ")
        + "Exact years are used for age, service, salary growth and discounting alike.",
    )
    step_card(
        2, "Yearly salary",
        r"S = \text{monthly salary} \times 14",
        rf"S = {tex_eur(row['monthly_salary_eur'])} \times 14 = {tex_eur(r['yearly_salary'])}",
        "Austrian employees receive 14 monthly salaries a year, so the yearly salary is 14 times "
        "the monthly one.",
    )
    step_card(
        3, "Projected final salary",
        rf"S_{{{age_text}}} = S \cdot (1+g)^{{n}}",
        rf"S_{{{age_text}}} = {tex_eur(r['yearly_salary'])} \cdot (1 + {g:g})^{{{n:.2f}}} "
        rf"= {tex_eur(r['projected_final_salary'])}",
        ("They retire on the valuation date, so their current salary is the final salary."
         if immediate else
         f"Today's salary grows by {g:.1%} a year, compounded over the {n:.2f} years until "
         "retirement, because the plan pays on the final salary."),
    )
    step_card(
        4, "Earned pension",
        rf"E = \text{{accrual}} \cdot s \cdot S_{{{age_text}}}",
        rf"E = {a['accrual_rate']:g} \cdot {s:.2f} \cdot {tex_eur(r['projected_final_salary'])} "
        rf"= {tex_eur(r['earned_pension'])}",
        f"The pension a year already earned: {a['accrual_rate']:.1%} of the projected final salary "
        f"for each of the {s:.2f} years worked so far. Future years are not counted yet.",
    )

    survival_symbolic, survival_substituted = survival_step_latex(r, a)
    step_card(
        5, "Survival to retirement",
        survival_symbolic, survival_substituted,
        survival_step_text(r, a),
    )

    annuity_symbolic, annuity_substituted = annuity_step_latex(r, a, contributions, age_text)
    step_card(
        6, "Annuity factor",
        annuity_symbolic, annuity_substituted,
        f"What 1 euro a year for life is worth at age {age_text}: each yearly payment is "
        "discounted for the wait and weighted by the chance of being alive to receive it. The "
        "second line is an approximation for instalments arriving through the year instead of all "
        "at its start. With no pension increases (j = 0) the bracket equals 1, which is the "
        "standard formula. The extra term extends that standard formula to pensions that "
        "increase each year, and is also an approximation.",
    )
    step_card(
        7, "DBO",
        rf"\text{{DBO}} = E \cdot \ddot{{a}}^{{(f)}}_{{{age_text}}} \cdot v^{{n}} \cdot {{}}_np_x",
        rf"\text{{DBO}} = {tex_eur(r['earned_pension'])} \cdot {r['annuity_at_retirement']:.4f} "
        rf"\cdot {r['discount_factor']:.4f} \cdot {r['survival_to_retirement']:.4f} "
        rf"= {tex_eur(r['dbo'])}, \qquad v^{{{n:.2f}}} = \frac{{1}}{{(1 + {i:g})^{{{n:.2f}}}}} "
        rf"= {r['discount_factor']:.4f}",
        f"The earned pension, valued as a lifelong stream at age {age_text}, brought back to today "
        f"and weighted by the chance of reaching that age: this is what the company owes "
        f"{employee_id} today.",
    )

    with st.container(border=True):
        st.markdown(
            "**The projected unit credit idea.** Count only the service **already worked** (that "
            "is what makes it an obligation today), but value it at the salary expected **at "
            "retirement** (because that is the salary the plan will actually pay on). Future "
            "service is added year by year as it is worked."
        )
    return r


def pension_mortality_section(r, a):
    """Show q_y on a log scale, the survival curve, and the annuity factor contributions."""
    st.header("Mortality and the annuity factor")
    R = a["retirement_age"]

    source_is_formula = a["q_table"] is None
    with st.container(border=True):
        st.markdown(f"**{a['mortality_source']}**"
                    + (" (illustrative, not an official table)" if source_is_formula else ""))
        life_years = life_expectancy(65, a)
        survival_45_65 = survival_prob(MORTALITY_REFERENCE_AGE, 65, a)
        annual_65, with_timing_65 = annuity_values(65, a)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Life expectancy at 65", f"{life_years:.2f} years")
        col2.metric("Survival from 45 to 65", f"{survival_45_65:.4f}")
        col3.metric("Annuity at 65, paid yearly", f"{annual_65:.4f}")
        col4.metric(f"Annuity at 65, {a['payments_per_year']} payments", f"{with_timing_65:.4f}")
        if source_is_formula:
            st.caption("These are illustrative values, calibrated to a unisex life expectancy at "
                       "65 of roughly 20 years. They are not from an official table.")

    col1, col2 = st.columns(2)

    with col1:
        mortality = cached_mortality_table(a, 20)
        chart = px.line(mortality, x="age", y="q_x", log_y=True,
                        labels={"age": "Whole age y", "q_x": "q_y (log scale)"},
                        title="One-year death probability by whole age")
        st.plotly_chart(chart, width="stretch")
        if source_is_formula:
            st.caption("A straight line on a log scale means death rates grow exponentially with "
                       "age, as in a Makeham-style formula applied directly to q_x. The curve "
                       "bends at young ages, where the constant risk A dominates.")
        else:
            st.caption("The death probabilities from the uploaded table, scaled by the mortality "
                       "factor.")

    with col2:
        survival = cached_survival_curve(float(r["age"]), a)
        chart = px.line(survival, x="age", y="survival",
                        labels={"age": "Age", "survival": "Probability of being alive"},
                        title=f"Survival of {r['employee_id']} from exact age {r['age']:.2f}")
        chart.add_vline(x=r["retirement_age_used"], line_dash="dash",
                        annotation_text=f"Retirement at {r['retirement_age_used']:.2f}"
                        if r["immediate_retirement"] else f"Retirement at {R}")
        st.plotly_chart(chart, width="stretch")
        st.caption(f"Probability of {r['employee_id']} still being alive at each age, from their "
                   "exact age today, using the same calculation as Step 5. The dashed line marks "
                   "retirement.")

    contributions = cached_annuity_contributions(r["retirement_age_used"], a)
    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(go.Bar(x=contributions["payment_age"], y=contributions["contribution"],
                           name="Contribution of each payment"), secondary_y=False)
    chart.add_trace(go.Scatter(x=contributions["payment_age"], y=contributions["running_total"],
                               name="Running total", mode="lines"), secondary_y=True)
    start_text = (f"{r['retirement_age_used']:.2f}" if r["immediate_retirement"] else f"{R}")
    chart.update_layout(title=f"Building the annuity factor at {start_text}, before payment timing: "
                              f"{contributions['running_total'].iloc[-1]:.4f}")
    chart.update_xaxes(title_text="Payment age")
    chart.update_yaxes(title_text="Survival × discount", secondary_y=False)
    chart.update_yaxes(title_text="Running total", secondary_y=True)
    st.plotly_chart(chart, width="stretch")
    st.caption("Each bar is one yearly payment, valued at the start of the pension: its chance of "
               "being paid (survival) times its discount factor. Later payments count less for two "
               f"reasons at once: fewer people are still alive, and the money is discounted for "
               f"longer. The line adds them up, and the payment timing adjustment of "
               f"{r['timing_adjustment']:.4f} then gives the {r['annuity_at_retirement']:.4f} "
               "used in the valuation.")


def pension_results_section(results, valued, total):
    """Show DBO across employees: scatter, sorted table, immediate retirements and concentration."""
    st.header("Results across all employees")
    table = results.merge(valued[["employee_id", "monthly_salary_eur"]], on="employee_id")

    chart = px.scatter(
        table, x="age", y="dbo", size="past_service",
        hover_data={"employee_id": True, "past_service": ":.1f", "yearly_salary": ":,.0f",
                    "dbo": ":,.0f", "age": ":.1f"},
        labels={"age": "Age", "dbo": "DBO (EUR)", "past_service": "Past service (years)",
                "yearly_salary": "Yearly salary", "employee_id": "Employee"},
        title="DBO by age (bubble size = years of service)",
    )
    st.plotly_chart(chart, width="stretch")

    columns = ["employee_id", "age", "past_service", "years_to_retirement", "yearly_salary",
               "earned_pension", "dbo", "service_cost"]
    column_config = {
        "employee_id": st.column_config.TextColumn("Employee"),
        "age": st.column_config.NumberColumn("Age", format="%.1f"),
        "past_service": st.column_config.NumberColumn("Service (years)", format="%.1f"),
        "years_to_retirement": st.column_config.NumberColumn("Years to retirement", format="%.1f"),
        "yearly_salary": st.column_config.NumberColumn("Yearly salary", format="€%,.0f"),
        "earned_pension": st.column_config.NumberColumn("Earned pension / year", format="€%,.0f"),
        "dbo": st.column_config.NumberColumn("DBO", format="€%,.0f"),
        "service_cost": st.column_config.NumberColumn("Service cost", format="€%,.0f"),
    }
    st.dataframe(table[columns].sort_values("dbo", ascending=False), hide_index=True,
                 width="stretch", column_config=column_config)

    # The total stays the exact sum so it ties to the cash flows, duration and checks; only the
    # display is rounded, so the difference is stated rather than hidden
    displayed_sum = table["dbo"].round(0).sum()
    st.caption(f"The total DBO of {eur(total)} is the exact sum of the unrounded employee values. "
               f"Each row is shown to the nearest euro, so the rows as displayed add up to "
               f"{eur(displayed_sum)}. Any small difference is rounding only.")

    immediate = table[table["immediate_retirement"]]
    if not immediate.empty:
        with st.container(border=True):
            st.markdown("**Valued as immediate retirement; please confirm status**")
            st.dataframe(immediate[columns].sort_values("dbo", ascending=False), hide_index=True,
                         width="stretch", column_config=column_config)
            st.caption(f"These {len(immediate)} employee(s), holding {eur(immediate['dbo'].sum())} "
                       "of the total, are at or past the chosen retirement age. They are valued as "
                       "retiring on the valuation date, with current salary and an immediate "
                       "annuity at their exact age, until the client confirms their status.")

    share = top_n_share(results, n=5)
    st.markdown(
        f"The 5 employees with the largest DBO hold **{share:.0f}%** of the total. Older employees "
        "with long service dominate: they have earned more years of pension, their salaries are "
        "higher, and their payments are closer, so they are discounted less and they are more "
        "likely to reach retirement."
    )


def pension_costs_section(results, valued, a, r, total):
    """Show next year's service cost and interest cost, with their formulas and simplifications."""
    st.header("Service cost and interest cost for the next year")
    outputs = cached_standard_outputs(results, valued, a)
    service_cost = outputs["service_cost"]
    i = a["discount_rate"]
    n = r["years_to_retirement"]
    age_text = f"{r['retirement_age_used']:.2f}" if r["immediate_retirement"] else f"{a['retirement_age']}"

    st.info("**Active employees only.** Pensioners are not valued. Employees at or past the "
            "retirement age are treated as retiring on the valuation date, so they earn no further "
            "service and have no service cost.")

    col1, col2 = st.columns(2)
    col1.metric("Current service cost, next year", eur(service_cost))
    col2.metric("Interest cost, next year", eur(outputs["interest_cost"]))

    with st.container(border=True):
        st.markdown("**Current service cost**")
        st.latex(rf"SC = \text{{accrual}} \cdot \min(1,\, n) \cdot S_{{{age_text}}} \cdot "
                 rf"\ddot{{a}}^{{(f)}}_{{{age_text}}} \cdot v^{{n}} \cdot {{}}_np_x")
        st.latex(rf"SC_{{\text{{{r['employee_id']}}}}} = {a['accrual_rate']:g} \cdot "
                 rf"{r['service_next_year']:.2f} \cdot {tex_eur(r['projected_final_salary'])} \cdot "
                 rf"{r['annuity_at_retirement']:.4f} \cdot {r['discount_factor']:.4f} \cdot "
                 rf"{r['survival_to_retirement']:.4f} = {tex_eur(r['service_cost'])}")
        st.caption(
            "The value today of the pension earned by working the next year, on the same projected "
            f"unit credit basis as the DBO. For one more year of service it equals the DBO per year "
            f"of service. Someone less than a year from retirement ({n:.2f} years for "
            f"{r['employee_id']}) earns only the part of the year they will work. The total is the "
            "sum over all employees."
        )

    with st.container(border=True):
        st.markdown("**Interest cost**")
        st.latex(r"IC = i \cdot \left(\text{DBO} + SC\right)")
        st.latex(rf"IC = {i:g} \cdot \left({tex_eur(total)} + {tex_eur(service_cost)}\right) "
                 rf"= {tex_eur(outputs['interest_cost'])}")
        st.caption(
            "One year's interest on what is owed. Simplifications: service cost is treated as "
            "arising at the start of the year, so it earns a full year's interest, and pension "
            f"payments due during the year are not deducted. Those payments are "
            f"{eur(outputs['payments_next_year'])}; if they were paid evenly through the year, the "
            f"interest cost would be about {eur(outputs['interest_on_payments'])} lower."
        )


def pension_cash_flow_section(valued, a, duration_years):
    """Show expected yearly pension payments and link them to duration."""
    st.header("Expected benefit cash flows")
    flows = cached_cash_flows(valued, a)
    average_time = cached_average_payment_time(valued, a)
    adjustment_total = flows["timing_adjustment"].sum()

    chart = go.Figure()
    chart.add_trace(go.Bar(x=flows["year"], y=flows["expected_payment"],
                           name="Expected payments (undiscounted)"))
    chart.add_trace(go.Bar(x=flows["year"], y=flows["timing_adjustment"],
                           name="Payment timing adjustment"))
    chart.add_trace(go.Scatter(x=flows["year"], y=flows["discounted_payment"],
                               name="Discounted to today", mode="lines"))
    chart.update_layout(title="Expected pension payments by calendar year",
                        xaxis_title="Year", yaxis_title="EUR", barmode="relative")
    st.plotly_chart(chart, width="stretch")
    st.caption(
        f"Payments start when the first employees retire and continue for decades. The **payment "
        f"timing adjustment** ({eur(adjustment_total)} undiscounted) is shown separately, dated at "
        f"each retirement date, because the {a['payments_per_year']}-payment approximation is a "
        "correction to the yearly payments and not a list of dated instalments. With the "
        f"adjustment, the discounted line adds up exactly to the total DBO "
        f"({eur(flows['discounted_payment'].sum())}). Weighted by present value, and using each "
        f"payment's exact date, the average payment is {average_time:.1f} years away. Duration "
        f"({duration_years:.1f} years) is close to that average divided by 1 + i "
        f"({average_time / (1 + a['discount_rate']):.1f}), which is why a mostly active workforce "
        "has a long duration."
    )


def pension_extra_life_note(sensitivity, a):
    """Show the mortality factor that adds one year to life expectancy at 65 and how it is found."""
    factor = cached_extra_life_factor(a)
    before = life_expectancy(LIFE_EXPECTANCY_AGE, a)
    after = life_expectancy(LIFE_EXPECTANCY_AGE, dict(a, mortality_factor=factor))
    row = sensitivity.set_index("scenario").loc[LIFE_EXPECTANCY_SCENARIO]
    with st.container(border=True):
        st.markdown("**Life expectancy +1 year at 65**")
        st.latex(r"\text{find } m^{*} \text{ such that } e_{65}(m^{*}) = e_{65}(m) + 1")
        st.latex(rf"e_{{65}}({a['mortality_factor']:g}) = {before:.2f}, \qquad "
                 rf"m^{{*}} = {factor:.4f}, \qquad e_{{65}}(m^{{*}}) = {after:.2f}")
        st.caption(
            f"A mortality factor of {factor:.4f} ({factor / a['mortality_factor']:.1%} of the "
            f"current setting) adds one year to life expectancy at 65 and changes the DBO by "
            f"{row['pct_change']:+.1f}%. It is found by bisection: life expectancy falls as the "
            "factor rises, so each trial halves the range that can contain the answer. The same "
            "factor applies at every age, before and after retirement."
        )


def pension_sensitivity_section(sensitivity, valued, a):
    """Show the sensitivity table, tornado chart, duration estimate and convexity."""
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
    pension_extra_life_note(sensitivity, a)
    st.plotly_chart(sensitivity_chart(sensitivity), width="stretch")

    by_scenario = sensitivity.set_index("scenario")
    lower = by_scenario.loc["discount rate -0.5%"]
    higher = by_scenario.loc["discount rate +0.5%"]
    base_dbo = by_scenario.loc["base", "total_dbo"]
    duration_years = duration(sensitivity)
    down_total, down_pct = cached_rate_move(valued, a, -0.01)
    up_total, up_pct = cached_rate_move(valued, a, 0.01)

    with st.container(border=True):
        st.markdown("**Duration: a first-order estimate for small rate moves**")
        st.latex(r"D \approx \frac{DBO_{i-0.5\%} - DBO_{i+0.5\%}}{2 \cdot DBO \cdot 0.005}")
        st.latex(rf"D \approx \frac{{{tex_eur(lower['total_dbo'])} - {tex_eur(higher['total_dbo'])}}}"
                 rf"{{2 \cdot {tex_eur(base_dbo)} \cdot 0.005}} = {duration_years:.1f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("First-order estimate of a 1-point move", f"±{duration_years:.1f}%")
        col2.metric("Exact effect of a 1-point fall", f"{down_pct:+.1f}%",
                    f"DBO {eur(down_total)}", delta_color="off")
        col3.metric("Exact effect of a 1-point rise", f"{up_pct:+.1f}%",
                    f"DBO {eur(up_total)}", delta_color="off")
        st.caption(
            f"Duration treats the DBO as a straight line in the discount rate, so it only works "
            f"for small moves. A full 1-point move is recalculated exactly above. The exact fall "
            f"({down_pct:+.1f}%) is larger than the estimate and the exact rise ({up_pct:+.1f}%) "
            "is smaller, because the DBO curves against the rate. This curve is convexity."
        )

    with st.container(border=True):
        st.markdown(
            f"**Convexity.** Lowering the discount rate by 0.5 points changes the DBO by "
            f"**{lower['pct_change']:+.1f}%**, and raising it by 0.5 points changes it by "
            f"**{higher['pct_change']:+.1f}%**. Discounting is a curve, $1/(1+i)^t$, not a "
            "straight line, so a rate cut increases far-away payments more than an equal rate "
            "rise reduces them. Duration averages both sides to cancel this out."
        )
    return duration_years


def pension_monte_carlo_section(valued, a):
    """Show the distribution of simulated total DBOs and its percentiles."""
    st.header("Assumption Monte Carlo")
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
    st.caption(
        "Each draw picks a discount rate and a salary growth rate around the chosen values "
        "(standard deviation 0.5 points each) and recalculates the total DBO. The two are drawn "
        "independently, although in reality both are linked through inflation and partly cancel "
        "out, so this range is likely too wide."
    )


def pension_run_record_section(raw, valid, results, queries, outputs, total, duration_years, a):
    """Show the record of this run and offer it, and the full workings, as downloads."""
    st.header("Run record and exports")
    # Taken fresh on every run of the page, so the record says when these numbers were produced
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    record = run_record(raw, valid, outputs, total, duration_years, a, VALUATION_DATE, timestamp)
    assumptions = assumptions_table(a)

    with st.container(border=True):
        # Only the display is text: the downloads below keep the numbers as numbers
        st.dataframe(record.astype({"value": str}), hide_index=True, width="stretch")
        st.caption("A short hash of the employee file as received identifies the exact input data: "
                   "any change to the file gives a different hash.")
        col1, col2 = st.columns(2)
        col1.download_button("Download run record (CSV)", data=record.to_csv(index=False),
                             file_name="pension_run_record.csv", mime="text/csv")
        workbook = excel_workbook(employee_export_table(results, valid), assumptions, record, queries)
        col2.download_button(
            "Download workbook (Excel)", data=workbook, file_name="pension_valuation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("The workbook has four sheets: per-employee results with every intermediate "
                   "step, the assumptions, this run record and the client queries.")


def pension_limitations_section():
    """List the model's limitations and the planned next steps."""
    st.header("Limitations and next steps")
    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.markdown(
            "**Limitations**\n"
            "- Mortality is an illustrative Makeham-style formula applied directly to q_x, with "
            "one unisex table; a real Austrian valuation would use the AVÖ 2018-P tables, which "
            "are licensed, separate for men and women, and not included here.\n"
            "- No turnover: nobody leaves before retirement, which overstates the DBO.\n"
            "- No pension increases after retirement unless a rate is set above.\n"
            "- No survivor benefits and no benefit on death before retirement.\n"
            "- Payment timing is an approximation: the standard formula, extended with an extra "
            "term when pensions increase, and it treats the 14 payments as evenly spaced.\n"
            "- Employees at or past the retirement age are assumed to retire on the valuation "
            "date until the client confirms their status.\n"
            "- Service cost and interest cost are for the next year, for active employees only, "
            "and use the simplifications stated in their section.\n"
            "- Active employees only; pensioners are not valued.\n"
            "- Discount rate and salary growth are drawn independently in the Monte Carlo.\n"
            "- One flat discount rate is used for every payment date.\n"
            "- The sample employee data is synthetic."
        )
    with col2, st.container(border=True):
        st.markdown(
            "**Next steps (from FUTURE.md)**\n"
            "- Use the official AVÖ 2018-P mortality tables.\n"
            "- Model employee turnover before retirement.\n"
            "- Value pensioners already receiving a pension.\n"
            "- Correlate discount rate and salary growth through inflation.\n"
            "- Add Austrian severance (Abfertigung alt) and jubilee payments."
        )


def page_pension():
    """Show the pension valuation as an actuarial walkthrough, from data to DBO."""
    st.title("Pension valuation: from employee data to balance-sheet number")
    st.markdown(
        "A company that promises its employees a pension must show today's value of that promise "
        "as a liability on its balance sheet. This value is the **defined benefit obligation "
        "(DBO)**: what the company owes now for the service employees have already worked, "
        "allowing for salary growth, the chance of survival and the time value of money. This page "
        "walks through how it is calculated for a fictional Austrian company with synthetic data."
    )

    if "employees_raw" not in st.session_state:
        load_employees_into_session(pd.read_csv(EMPLOYEES_FILE), "sample employees")

    # The headline depends on settings further down, and validation depends on the retirement
    # age, so reserve both places now and fill them once the assumptions are known
    headline = st.container()

    pension_promise_section()
    data_area = st.container()
    a, mortality_upload, mortality_status = pension_assumptions_section()

    with data_area:
        valid, queries = pension_data_section(a["retirement_age"])
    if valid.empty:
        st.error("No employees to value with these settings.")
        return

    a = apply_uploaded_mortality(a, mortality_upload, valid, mortality_status)
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
        model_checks_card(cached_model_checks(valid, a))

    selected = pension_method_section(valid, results, a)
    pension_mortality_section(selected, a)
    pension_results_section(results, valid, total)
    pension_costs_section(results, valid, a, selected, total)
    pension_cash_flow_section(valid, a, duration_years)
    pension_sensitivity_section(sensitivity, valid, a)
    pension_monte_carlo_section(valid, a)
    pension_run_record_section(st.session_state["employees_raw"], valid, results, queries, outputs,
                               total, duration_years, a)
    pension_limitations_section()


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
