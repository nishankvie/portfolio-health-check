"""Tests for the pension model. Run from the project folder: python -m pytest tests/test_pension.py -v"""

import pandas as pd
import pytest

from src.pension import (
    ASSUMPTIONS,
    DATA_DIR,
    NOT_VALUED,
    VALUATION_DATE,
    VALUED_IMMEDIATELY,
    annuity_factor,
    death_prob,
    dbo_employee,
    dbo_portfolio,
    employees_below_age,
    expected_cash_flows,
    interest_cost,
    life_expectancy,
    LIFE_EXPECTANCY_SCENARIO,
    mortality_factor_for_extra_life,
    payments_due_next_year,
    run_model_checks,
    standard_outputs,
    sensitivity_table,
    survival_curve,
    survival_prob,
    validate_employees,
    validate_mortality_table,
)

# A table with exactly the formula's values, so it must reproduce the formula's results
FORMULA_AS_TABLE = {age: death_prob(age, ASSUMPTIONS) for age in range(20, 111)}

# The checks must hold at the edges of the page's settings, not only at the default settings
SCENARIOS = {
    "default": {},
    "lowest_rate_highest_growth": {"discount_rate": 0.01, "salary_growth": 0.05},
    "highest_rate_no_growth": {"discount_rate": 0.06, "salary_growth": 0.0},
    "lowest_mortality": {"mortality_factor": 0.8},
    "highest_mortality": {"mortality_factor": 1.2},
    "retire_at_60": {"retirement_age": 60},
    "retire_at_67": {"retirement_age": 67},
    "paid_once_a_year": {"payments_per_year": 1},
    "paid_12_times": {"payments_per_year": 12},
    "increases_2_percent": {"pension_increase": 0.02},
    "increase_equals_discount": {"pension_increase": 0.01, "discount_rate": 0.01},
    "uploaded_table": {"q_table": FORMULA_AS_TABLE},
}


@pytest.fixture(scope="module")
def valid_employees():
    """Return the validated sample employees, read from the saved file without regenerating it."""
    raw = pd.read_csv(DATA_DIR / "employees_raw.csv")
    return validate_employees(raw)[0]


@pytest.fixture(scope="module", params=list(SCENARIOS))
def checks(request, valid_employees):
    """Return the model checks for one scenario, run once per scenario to save time."""
    assumptions = dict(ASSUMPTIONS, **SCENARIOS[request.param])
    return run_model_checks(valid_employees, a=assumptions)


def assert_check_passes(checks, number):
    """Fail with the check's own detail text if the check with this number did not pass."""
    for name, passed, detail in checks:
        if name.startswith(f"{number} "):
            assert passed, f"{name}: {detail}"
            return
    pytest.fail(f"No check numbered {number}")


def test_result_format(checks):
    """Every check must be a (name, True or False, detail text) triple, ready for the page."""
    assert isinstance(checks, list)
    for name, passed, detail in checks:
        assert isinstance(name, str)
        assert isinstance(passed, bool)
        assert isinstance(detail, str)


@pytest.mark.parametrize("number", ["1", "2", "3", "4", "5a", "5b", "5c", "5d", "5e", "5f",
                                    "6", "7", "8", "9", "10", "11", "12", "13"])
def test_model_check(checks, number):
    assert_check_passes(checks, number)


# ---------------------------------------------------------------------------------------------
# Survival over fractional years (uniform distribution of deaths)
# ---------------------------------------------------------------------------------------------

def test_part_year_survival_matches_hand_calculation():
    """From 44.25 to 45 is the rest of one year of age, worked out by hand from q_44."""
    q44 = death_prob(44, ASSUMPTIONS)
    assert survival_prob(44.25, 45) == pytest.approx((1 - q44) / (1 - 0.25 * q44), abs=1e-14)


def test_part_year_to_part_year_matches_hand_calculation():
    """Inside one year of age, survival is the ratio of two points on a straight line."""
    q44 = death_prob(44, ASSUMPTIONS)
    assert survival_prob(44.25, 44.75) == pytest.approx((1 - 0.75 * q44) / (1 - 0.25 * q44), abs=1e-14)


def test_whole_years_are_a_plain_product():
    expected = 1.0
    for age in range(45, 65):
        expected *= 1 - death_prob(age, ASSUMPTIONS)
    assert survival_prob(45, 65) == pytest.approx(expected, abs=1e-14)


def test_splitting_a_period_changes_nothing():
    """The audit found 0.889070 against 0.889152 for this split; they must now be equal."""
    direct = survival_prob(44.73922, 65)
    split = survival_prob(44.73922, 45) * survival_prob(45, 65)
    assert direct == pytest.approx(split, abs=1e-14)


def test_survival_backwards_is_refused():
    with pytest.raises(ValueError):
        survival_prob(66, 65)


def test_chart_uses_the_same_survival_as_the_valuation():
    curve = survival_curve(44.73922)
    at_65 = curve.loc[curve["age"] == 65, "survival"].iloc[0]
    assert at_65 == pytest.approx(survival_prob(44.73922, 65), abs=1e-14)


# ---------------------------------------------------------------------------------------------
# Assumptions, payment timing and immediate retirement
# ---------------------------------------------------------------------------------------------

def test_life_expectancy_is_roughly_20():
    assert 19.5 <= life_expectancy(65) <= 20.5


def test_timing_adjustment_is_in_the_cash_flows(valid_employees):
    results, total = dbo_portfolio(valid_employees)
    flows = expected_cash_flows(valid_employees)
    assert flows["timing_adjustment"].sum() < 0
    assert flows["discounted_payment"].sum() == pytest.approx(total, abs=0.01)


def test_more_instalments_lower_the_annuity():
    once = annuity_factor(65, dict(ASSUMPTIONS, payments_per_year=1))
    fourteen = annuity_factor(65, dict(ASSUMPTIONS, payments_per_year=14))
    assert once - fourteen == pytest.approx(13 / 28, abs=1e-12)


def test_uploaded_table_reproduces_the_formula(valid_employees):
    formula_total = dbo_portfolio(valid_employees)[1]
    table_total = dbo_portfolio(valid_employees, a=dict(ASSUMPTIONS, q_table=FORMULA_AS_TABLE))[1]
    assert table_total == pytest.approx(formula_total, abs=1e-6)


@pytest.mark.parametrize("slider_value", [0.8, 1.0, 1.2])
def test_life_expectancy_scenario_adds_one_year_from_the_slider(valid_employees, slider_value):
    a = dict(ASSUMPTIONS, mortality_factor=slider_value)
    table = sensitivity_table(valid_employees, a=a)
    row = table[table["scenario"] == LIFE_EXPECTANCY_SCENARIO].iloc[0]
    with_extra_life = life_expectancy(65, dict(a, mortality_factor=row["assumption_value"]))
    assert with_extra_life == pytest.approx(life_expectancy(65, a) + 1, abs=1e-6)
    assert row["assumption_value"] < slider_value
    assert row["pct_change"] > 0


def test_old_mortality_scenario_is_gone(valid_employees):
    assert "mortality x 0.9" not in sensitivity_table(valid_employees)["scenario"].values


def test_impossible_life_expectancy_target_is_refused():
    with pytest.raises(ValueError):
        mortality_factor_for_extra_life(100.0)


def test_later_retirement_can_correctly_raise_dbo_when_growth_exceeds_discount(valid_employees):
    """At 5% growth and 1% discount, waiting a year adds more than discounting removes."""
    young = employees_below_age(valid_employees, 60)
    a = dict(ASSUMPTIONS, discount_rate=0.01, salary_growth=0.05)
    at_60 = dbo_portfolio(young, a=dict(a, retirement_age=60))[1]
    at_61 = dbo_portfolio(young, a=dict(a, retirement_age=61))[1]
    assert at_61 > at_60
    name, passed, detail = run_model_checks(valid_employees, a=a)[7]
    assert name.startswith("5d ") and passed and "Not a rule" in detail


def one_employee(birth_date, entry_date="2000-01-01", salary=5000.0):
    """Return a single employee record for hand calculations."""
    return pd.Series({
        "employee_id": "X",
        "birth_date": pd.Timestamp(birth_date),
        "entry_date": pd.Timestamp(entry_date),
        "monthly_salary_eur": salary,
    })


def test_service_cost_is_the_dbo_per_year_of_service():
    result = dbo_employee(one_employee("1980-01-15"))
    assert result["service_next_year"] == 1
    assert result["service_cost"] == pytest.approx(result["dbo"] / result["past_service"], abs=1e-9)


def test_service_cost_is_part_of_a_year_for_someone_retiring_soon():
    result = dbo_employee(one_employee("1962-06-30"))  # 64.5 at the valuation date
    assert 0 < result["years_to_retirement"] < 1
    assert result["service_next_year"] == result["years_to_retirement"]
    expected = result["dbo"] / result["past_service"] * result["years_to_retirement"]
    assert result["service_cost"] == pytest.approx(expected, abs=1e-9)


def test_service_cost_is_zero_past_retirement():
    assert dbo_employee(one_employee("1959-09-30"))["service_cost"] == 0


def test_new_joiner_has_service_cost_but_no_dbo():
    result = dbo_employee(one_employee("1990-01-01", entry_date=VALUATION_DATE))
    assert result["dbo"] == 0
    assert result["service_cost"] > 0


def test_interest_cost_formula():
    assert interest_cost(1000.0, 100.0, dict(ASSUMPTIONS, discount_rate=0.04)) == pytest.approx(44.0)


def test_standard_outputs_add_up(valid_employees):
    results, total = dbo_portfolio(valid_employees)
    outputs = standard_outputs(results, valid_employees)
    assert outputs["service_cost"] == pytest.approx(results["service_cost"].sum())
    assert outputs["interest_cost"] == pytest.approx(0.035 * (total + outputs["service_cost"]))
    assert outputs["payments_next_year"] == pytest.approx(payments_due_next_year(valid_employees))
    assert outputs["payments_next_year"] > 0


def test_employee_past_retirement_is_valued_as_retiring_today():
    row = pd.Series({
        "employee_id": "OLD",
        "birth_date": pd.Timestamp("1959-09-30"),
        "entry_date": pd.Timestamp("1989-09-30"),
        "monthly_salary_eur": 5200.0,
    })
    result = dbo_employee(row)
    assert result["immediate_retirement"]
    assert result["years_to_retirement"] == 0
    assert result["projected_final_salary"] == result["yearly_salary"]
    assert result["discount_factor"] == 1
    assert result["survival_to_retirement"] == 1
    assert result["annuity_at_retirement"] == pytest.approx(annuity_factor(result["age"]), abs=1e-12)
    assert result["dbo"] > 0


# ---------------------------------------------------------------------------------------------
# Validation of employee data
# ---------------------------------------------------------------------------------------------

def employees(**changes):
    """Return a small clean employee file; changes are {column: {row: value}}."""
    table = pd.DataFrame({
        "employee_id": ["A1", "A2", "A3"],
        "birth_date": ["1980-01-15", "1975-06-30", "1990-03-01"],
        "entry_date": ["2010-01-01", "2005-07-01", "2018-09-01"],
        "monthly_salary_eur": [4000.0, 5000.0, 3500.0],
    })
    table["monthly_salary_eur"] = table["monthly_salary_eur"].astype(object)
    for column, values in changes.items():
        for row, value in values.items():
            table.loc[row, column] = value
    return table


def issues_for(query_df, employee_id):
    return query_df.loc[query_df["employee_id"] == employee_id, "issue"].tolist()


def test_clean_file_does_not_crash():
    valid, log_df, query_df = validate_employees(employees())
    assert len(valid) == 3
    assert log_df.empty
    assert query_df.empty


@pytest.mark.parametrize("column, issue", [("birth_date", "missing birth date"),
                                           ("entry_date", "missing entry date")])
def test_blank_date_is_a_query(column, issue):
    valid, _, query_df = validate_employees(employees(**{column: {1: None}}))
    assert issue in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


def test_blank_salary_is_a_query():
    valid, _, query_df = validate_employees(employees(monthly_salary_eur={1: None}))
    assert "missing salary" in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


def test_entry_after_valuation_is_a_query():
    valid, _, query_df = validate_employees(employees(entry_date={1: "2028-06-30"}))
    assert "entry after valuation date" in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


@pytest.mark.parametrize("salary", [0, -3000])
def test_zero_or_negative_salary_is_a_query(salary):
    valid, _, query_df = validate_employees(employees(monthly_salary_eur={1: salary}))
    assert "salary not positive" in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


@pytest.mark.parametrize("text", ["4.200", "4,200", "€4,200", "4,200.50"])
def test_ambiguous_salary_is_a_query_not_a_guess(text):
    valid, _, query_df = validate_employees(employees(monthly_salary_eur={1: text}))
    assert "ambiguous salary" in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


@pytest.mark.parametrize("text, expected", [("€4.200,00", 4200.0), ("4200,50", 4200.5),
                                            ("€4.200", 4200.0), ("5200.0", 5200.0), ("5200", 5200.0)])
def test_unambiguous_text_is_converted(text, expected):
    valid, _, query_df = validate_employees(employees(monthly_salary_eur={1: text}))
    salary = valid.loc[valid["employee_id"] == "A2", "monthly_salary_eur"].iloc[0]
    assert salary == expected
    assert query_df.empty


def test_unreadable_salary_is_a_query():
    valid, _, query_df = validate_employees(employees(monthly_salary_eur={1: "ask HR"}))
    assert "unreadable salary" in issues_for(query_df, "A2")
    assert "A2" not in valid["employee_id"].values


def test_text_salary_duplicate_is_removed_as_a_duplicate():
    """The same person with the salary once as a number and once as text is one person."""
    table = employees()
    extra = table.loc[[1]].copy()
    extra["monthly_salary_eur"] = "€5.000,00"
    valid, log_df, query_df = validate_employees(pd.concat([table, extra], ignore_index=True))
    assert len(valid) == 3
    assert (log_df["issue"] == "exact duplicate").sum() == 1
    assert query_df.empty


def test_same_id_with_different_values_is_a_query():
    table = employees()
    extra = table.loc[[1]].copy()
    extra["monthly_salary_eur"] = 9999.0
    valid, _, query_df = validate_employees(pd.concat([table, extra], ignore_index=True))
    assert issues_for(query_df, "A2") == ["conflicting records"]
    assert "A2" not in valid["employee_id"].values
    assert len(valid) == 2


def test_blank_employee_id_is_a_query():
    valid, _, query_df = validate_employees(employees(employee_id={1: None}))
    assert "missing employee ID" in query_df["issue"].tolist()
    assert len(valid) == 2


def test_validation_uses_the_chosen_retirement_age():
    table = employees(birth_date={1: "1961-03-01"})  # aged 65.8 at the valuation date
    _, _, at_65 = validate_employees(table, VALUATION_DATE, 65)
    _, _, at_67 = validate_employees(table, VALUATION_DATE, 67)
    assert issues_for(at_65, "A2") == ["at or past retirement age"]
    assert issues_for(at_67, "A2") == []


def test_past_retirement_query_does_not_exclude_the_employee():
    table = employees(birth_date={1: "1961-03-01"})
    valid, _, query_df = validate_employees(table, VALUATION_DATE, 65)
    assert "A2" in valid["employee_id"].values
    assert query_df.loc[query_df["employee_id"] == "A2", "effect"].iloc[0] == VALUED_IMMEDIATELY


def test_past_retirement_with_another_problem_is_not_valued():
    table = employees(birth_date={1: "1961-03-01"}, monthly_salary_eur={1: None})
    valid, _, query_df = validate_employees(table, VALUATION_DATE, 65)
    assert "A2" not in valid["employee_id"].values
    assert set(query_df.loc[query_df["employee_id"] == "A2", "effect"]) == {NOT_VALUED}


def test_valid_employees_have_no_missing_values(valid_employees):
    columns = ["employee_id", "birth_date", "entry_date", "monthly_salary_eur"]
    assert not valid_employees[columns].isna().any().any()


# ---------------------------------------------------------------------------------------------
# Validation of an uploaded mortality table
# ---------------------------------------------------------------------------------------------

def mortality_upload(first_age=20, last_age=110):
    """Return a valid upload with a gentle rise in q_x."""
    ages = list(range(first_age, last_age + 1))
    return pd.DataFrame({"age": ages, "q_x": [min(0.001 * 1.08 ** (age - first_age), 1) for age in ages]})


def test_good_mortality_table_is_accepted():
    table, problems = validate_mortality_table(mortality_upload(), 25, 62)
    assert problems == []
    assert table[20] == pytest.approx(0.001)


def test_mortality_table_needs_its_columns():
    table, problems = validate_mortality_table(pd.DataFrame({"age": [20], "qx": [0.1]}), 25, 62)
    assert table is None
    assert "q_x" in problems[0]


def test_mortality_table_with_a_gap_is_rejected():
    upload = mortality_upload()
    table, problems = validate_mortality_table(upload[upload["age"] != 50], 25, 62)
    assert table is None
    assert any("continuous" in problem for problem in problems)


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01])
def test_mortality_table_rates_must_be_probabilities(bad_rate):
    upload = mortality_upload()
    upload.loc[5, "q_x"] = bad_rate
    table, problems = validate_mortality_table(upload, 25, 62)
    assert table is None
    assert any("between 0 and 1" in problem for problem in problems)


def test_mortality_table_must_cover_the_employees():
    table, problems = validate_mortality_table(mortality_upload(first_age=30), 25, 62)
    assert table is None
    assert any("start at age" in problem for problem in problems)
