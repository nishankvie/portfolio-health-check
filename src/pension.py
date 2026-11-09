"""Synthetic employee data for a pension provision example. All employees are fictional."""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import log_change, parse_eur

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

# The pension promise valued in this module (a defined benefit plan):
# from age 65 the employee receives, every year,
#     1% x years of service x final yearly salary
# where yearly salary = monthly salary x 14, because Austrian employees are paid 14 monthly
# salaries a year (12 regular + holiday pay in summer + Christmas pay in winter).
# Example: 30 years of service and a final monthly salary of 5,000 gives
#     0.01 x 30 x 5,000 x 14 = 21,000 euros a year for life.

VALUATION_DATE = "2026-12-31"
REQUIRED_EMPLOYEE_COLUMNS = ["employee_id", "birth_date", "entry_date", "monthly_salary_eur"]
RETIREMENT_AGE = 65
MIN_WORKING_AGE = 15
MIN_AGE = 22
MAX_AGE = 63
MIN_ENTRY_AGE = 20
MIN_SALARY_EUR = 2500
MAX_SALARY_EUR = 8000
DAYS_PER_YEAR = 365.25
PAYMENT_FREQUENCIES = (1, 12, 14)
SALARY_PAYMENTS_PER_YEAR = 14
# An uploaded table must reach this age, or the annuity would stop too early
MIN_TABLE_LAST_AGE = 100
# Survival from this age to 65 is shown on the page, so an uploaded table must cover it
MORTALITY_REFERENCE_AGE = 45
ILLUSTRATIVE_MORTALITY = "Illustrative Makeham-style formula applied directly to q_x"
# What a client query means for the valuation until it is answered
LIFE_EXPECTANCY_AGE = 65
LIFE_EXPECTANCY_SCENARIO = "life expectancy +1 year at 65"
# Close enough that the resulting life expectancy is right to well under a thousandth of a year
BISECTION_TOLERANCE = 1e-8
NOT_VALUED = "Not valued until answered"
VALUED_IMMEDIATELY = "Valued as immediate retirement until confirmed"
LOG_COLUMNS = ["company_id", "month", "column", "issue", "old_value", "new_value"]

# TIME BASIS. The whole model uses one rule:
# - Age, past service, years to retirement, salary growth and discounting use exact years
#   (days between dates / 365.25) at the valuation date.
# - Death probabilities q_y are for whole ages y. Within a year of age, deaths are assumed to
#   be spread evenly (uniform distribution of deaths), so survival over a part-year has one
#   rule and a period can be split anywhere without changing the answer.
# - Yearly pension dates count from the exact retirement date.
# The minimum-working-age query alone uses completed years, because that rule is a legal age.

# Valuation assumptions. ALL ARE ILLUSTRATIVE, chosen to show the method, not to value a real plan.
# Mortality uses a Makeham-style formula applied directly to q_x, calibrated so that unisex life
# expectancy at 65 is roughly 20 years; a real Austrian valuation would use the official
# AVÖ 2018-P tables (separate for men and women). Those are licensed and not included here.
# Turnover (employees leaving before retirement) and death benefits before retirement are
# ignored, which overstates the DBO
ASSUMPTIONS = {
    "discount_rate": 0.035,     # yearly rate used to convert future payments into today's euros
    "salary_growth": 0.025,     # yearly salary increase until retirement
    "retirement_age": RETIREMENT_AGE,
    "accrual_rate": 0.01,       # 1% of final salary earned per year of service
    "makeham_A": 0.0002,        # age-independent death risk (accidents, illness)
    "makeham_B": 0.000032,      # size of the age-related death risk
    "makeham_c": 1.095,         # age-related risk grows by 9.5% per year of age
    "mortality_factor": 1.0,    # scales every death probability, e.g. 0.9 = people live longer
    "max_age": 110,             # nobody is assumed to live beyond this age
    "payments_per_year": 14,    # Austrian pensions are commonly paid 14 times a year
    "pension_increase": 0.0,    # yearly increase of a pension once it is being paid
    "q_table": None,            # optional dictionary of whole age -> q_x, replaces the formula
    "mortality_source": ILLUSTRATIVE_MORTALITY,
}

# Size of the up/down shift in the sensitivity table (0.5 percentage points)
SENSITIVITY_SHIFT = 0.005


def generate_employees(n=40, valuation_date=VALUATION_DATE, seed=42):
    """Return a clean DataFrame of fictional employees with birth, entry date and salary."""
    rng = np.random.default_rng(seed)
    valuation = pd.Timestamp(valuation_date)

    employee_ids = [f"E{i:03d}" for i in range(1, n + 1)]

    ages = rng.uniform(MIN_AGE, MAX_AGE, size=n)
    birth_dates = (valuation - pd.to_timedelta(ages * DAYS_PER_YEAR, unit="D")).floor("D")

    # Joined somewhere between age 20 and one year before the valuation date
    entry_ages = rng.uniform(MIN_ENTRY_AGE, ages - 1)
    entry_dates = (birth_dates + pd.to_timedelta(entry_ages * DAYS_PER_YEAR, unit="D")).floor("D")

    # Salary rises with age (experience), plus individual noise, kept inside the allowed range
    age_share = (ages - MIN_AGE) / (MAX_AGE - MIN_AGE)
    raw_salaries = MIN_SALARY_EUR + age_share * 3500 + rng.normal(0, 700, size=n)
    raw_salaries = np.clip(raw_salaries, MIN_SALARY_EUR, MAX_SALARY_EUR)
    salaries = np.round(raw_salaries / 50) * 50

    employees = pd.DataFrame({
        "employee_id": employee_ids,
        "birth_date": birth_dates,
        "entry_date": entry_dates,
        "monthly_salary_eur": salaries,
    })
    return employees


def austrian_format(value):
    """Return a number as Austrian-style text, e.g. 4200 -> '€4.200,00'."""
    english = f"{value:,.2f}"
    # Swap the separators through a placeholder, so the first swap isn't undone by the second
    austrian = english.replace(",", "#").replace(".", ",").replace("#", ".")
    return "€" + austrian


def make_messy_employees(employees, seed=42):
    """Save the clean table as the truth, plant errors in a copy, save and return it."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(exist_ok=True)
    employees.to_csv(DATA_DIR / "employees_truth.csv", index=False)

    messy = employees.copy()

    # Separate rows for each error, so no row gets two and every error can be counted
    picked = rng.choice(messy.index, size=13, replace=False)
    swap_rows = picked[0:2]
    young_entry_rows = picked[2:4]
    missing_salary_rows = picked[4:7]
    text_salary_rows = picked[7:11]
    duplicate_rows = picked[11:13]

    # The salary column must be able to hold text before we plant it
    messy["monthly_salary_eur"] = messy["monthly_salary_eur"].astype(object)

    # Error: entry date before birth date (the two dates swapped)
    for row in swap_rows:
        birth = messy.loc[row, "birth_date"]
        messy.loc[row, "birth_date"] = messy.loc[row, "entry_date"]
        messy.loc[row, "entry_date"] = birth

    # Error: entry age under 15 (entry year moved back, month and day kept)
    for row in young_entry_rows:
        birth = messy.loc[row, "birth_date"]
        entry = messy.loc[row, "entry_date"]
        wrong_entry_age = int(rng.integers(10, 15))
        years_back = (entry.year - birth.year) - wrong_entry_age
        # DateOffset handles 29 February safely (it moves to the 28th in non-leap years)
        messy.loc[row, "entry_date"] = entry - pd.DateOffset(years=years_back)

    # Error: missing salaries
    messy.loc[missing_salary_rows, "monthly_salary_eur"] = np.nan

    # Error: salaries typed as Austrian text, e.g. "€4.200,00"
    for row in text_salary_rows:
        messy.loc[row, "monthly_salary_eur"] = austrian_format(messy.loc[row, "monthly_salary_eur"])

    # Error: duplicated rows (exact copies appended to the end)
    messy = pd.concat([messy, messy.loc[duplicate_rows]])

    # Error: an extra employee aged 67, already past the retirement age of 65
    birth = pd.Timestamp(VALUATION_DATE) - pd.DateOffset(years=67, months=3)
    too_old = pd.DataFrame({
        "employee_id": [f"E{len(employees) + 1:03d}"],
        "birth_date": [birth],
        "entry_date": [birth + pd.DateOffset(years=30)],
        "monthly_salary_eur": [5200.0],
    })
    messy = pd.concat([messy, too_old], ignore_index=True)

    messy.to_csv(DATA_DIR / "employees_raw.csv", index=False)
    return messy


def age_in_years(birth_date, on_date):
    """Return completed years of age on a date (birthday not yet reached = one year less)."""
    birthday_not_reached = (on_date.month, on_date.day) < (birth_date.month, birth_date.day)
    return on_date.year - birth_date.year - int(birthday_not_reached)


def parse_salary(value):
    """Return (salary as a float or NaN, problem text or None), without ever guessing."""
    if not isinstance(value, str):
        if pd.notna(value) and not np.isfinite(value):
            return np.nan, "unreadable"
        return value, None

    text = value.strip()
    if text == "":
        return np.nan, None
    digits = text.replace("€", "").replace(" ", "")

    # "4.200" could be 4,200 euros or 4.20 euros, and "4,200" has the same problem. A euro
    # sign settles the dot (Austrian thousands), but not a lone comma with three digits after it
    lone_comma = re.fullmatch(r"\d{1,3},\d{3}", digits)
    lone_dot = "€" not in text and re.fullmatch(r"\d{1,3}\.\d{3}", digits)
    # A dot after a comma is English format, which parse_eur would misread
    english_format = "," in digits and "." in digits and digits.rfind(".") > digits.rfind(",")
    if lone_comma or lone_dot or english_format:
        return np.nan, "ambiguous"

    try:
        # parse_eur treats every dot as a thousands separator, which is right for "€4.200,00"
        # but would turn a plain "5200.0" (as written by pandas) into 52000. Only Austrian
        # text (with € or a decimal comma) goes through it; plain numbers use float()
        if "€" in text or "," in text:
            salary = parse_eur(text)
        else:
            salary = float(text)
    except ValueError:
        return np.nan, "unreadable"

    if not np.isfinite(salary):
        return np.nan, "unreadable"
    return salary, None


def salary_queries(salary, problem, original):
    """Return (issue, question) pairs for a salary that can't be used."""
    if problem == "ambiguous":
        return [("ambiguous salary",
                 f"Salary is written as '{original}', which could mean thousands or decimals; "
                 "please confirm the gross monthly salary in euros.")]
    if problem == "unreadable":
        return [("unreadable salary",
                 f"Salary is written as '{original}', which can't be read as a number; please "
                 "provide the gross monthly salary in euros.")]
    if pd.isna(salary):
        return [("missing salary",
                 "Monthly salary is missing; please provide the current gross monthly salary.")]
    if salary <= 0:
        return [("salary not positive",
                 f"Monthly salary is {salary:,.2f}, which is zero or negative; please provide "
                 "the current gross monthly salary.")]
    return []


def date_queries(birth, entry, valuation):
    """Return (issue, question) pairs for blank or impossible birth and entry dates."""
    found = []
    if pd.isna(birth):
        found.append(("missing birth date",
                      "Birth date is blank or unreadable; please provide it as YYYY-MM-DD."))
    if pd.isna(entry):
        found.append(("missing entry date",
                      "Entry date is blank or unreadable; please provide it as YYYY-MM-DD."))

    if pd.notna(entry) and entry > valuation:
        found.append(("entry after valuation date",
                      f"Entry date {entry:%d.%m.%Y} is after the valuation date "
                      f"{valuation:%d.%m.%Y}; please confirm the entry date."))

    if pd.notna(birth) and pd.notna(entry):
        if entry < birth:
            found.append(("entry before birth",
                          f"Entry date {entry:%d.%m.%Y} is before birth date {birth:%d.%m.%Y}; "
                          "dates possibly swapped, please confirm both dates."))
        # elif: with swapped dates the entry age is negative, which would repeat the same problem
        elif age_in_years(birth, entry) < MIN_WORKING_AGE:
            found.append(("entry age under 15",
                          f"Entry on {entry:%d.%m.%Y} means joining at age "
                          f"{age_in_years(birth, entry)}, below the minimum working age of "
                          f"{MIN_WORKING_AGE}; please confirm the entry date."))
    return found


def clean_salaries_and_duplicates(df, log):
    """Return (df with numeric salaries and exact duplicates removed, salary problems, originals)."""
    originals = df["monthly_salary_eur"].copy()
    parsed = [parse_salary(value) for value in originals]
    df["monthly_salary_eur"] = pd.to_numeric(pd.Series([item[0] for item in parsed], index=df.index))
    problems = pd.Series([item[1] for item in parsed], index=df.index, dtype=object)

    # Salaries are numbers by now, so "€4.200,00" and 4200.0 count as the same value. An exact
    # duplicate is the same person listed twice, which would double their liability
    exact_duplicate = df.duplicated()
    for row in df.index[exact_duplicate]:
        log_change(log, df.loc[row, "employee_id"], None, "all", "exact duplicate", None, "dropped")
    df = df[~exact_duplicate].copy()

    # Converting a format is not guessing, so text salaries are fixed, not queried
    for row in df.index:
        old_value = originals[row]
        if isinstance(old_value, str) and ("€" in old_value or "," in old_value) and pd.isna(problems[row]):
            log_change(log, df.loc[row, "employee_id"], None, "monthly_salary_eur",
                       "text number", old_value, df.loc[row, "monthly_salary_eur"])
    return df, problems, originals


def validate_employees(raw_df, valuation_date=VALUATION_DATE, retirement_age=RETIREMENT_AGE):
    """Return (valid employees, change log, client query list) without estimating anything."""
    # Why query instead of estimate: each row is one person's individual pension promise,
    # and there are no neighbouring values to borrow from (unlike a company's previous and
    # next month). A guessed birth date or salary would directly change that person's
    # liability, so only the client can supply the correct value
    missing = [column for column in REQUIRED_EMPLOYEE_COLUMNS if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    valuation = pd.Timestamp(valuation_date)
    df = raw_df.copy()
    log = []

    # errors="coerce" turns a blank or unreadable date into a missing value, which becomes a
    # question for the client instead of rejecting the whole file
    df["birth_date"] = pd.to_datetime(df["birth_date"], format="%Y-%m-%d", errors="coerce")
    df["entry_date"] = pd.to_datetime(df["entry_date"], format="%Y-%m-%d", errors="coerce")
    df, salary_problems, original_salaries = clean_salaries_and_duplicates(df, log)

    queries = []
    blocked_rows = set()

    def add_blocking_query(row, employee_id, issue, question):
        """Record a question whose answer is needed before this row can be valued."""
        blocked_rows.add(row)
        queries.append({"row": row, "employee_id": employee_id, "issue": issue,
                        "question": question, "blocking": True})

    # The same ID with different details is one person with conflicting records, not two people
    for employee_id in df["employee_id"][df["employee_id"].duplicated()].dropna().unique():
        rows = df.index[df["employee_id"] == employee_id]
        add_blocking_query(rows[0], employee_id, "conflicting records",
                           f"Employee {employee_id} appears {len(rows)} times with different "
                           "details; please confirm which record is correct.")
        blocked_rows.update(rows)

    for row in df.index:
        employee_id = df.loc[row, "employee_id"]
        birth = df.loc[row, "birth_date"]

        if pd.isna(employee_id):
            employee_id = f"(row {row + 2} of the file)"
            add_blocking_query(row, employee_id, "missing employee ID",
                               "Employee ID is blank; please identify this employee.")

        found = date_queries(birth, df.loc[row, "entry_date"], valuation)
        found += salary_queries(df.loc[row, "monthly_salary_eur"], salary_problems[row],
                                original_salaries[row])
        for issue, question in found:
            add_blocking_query(row, employee_id, issue, question)

        # Exact years, the same measure the valuation itself uses. This one doesn't block:
        # the person is valued as retiring today while the client confirms their status
        if pd.notna(birth) and exact_years(birth, valuation) >= retirement_age:
            age_now = exact_years(birth, valuation)
            queries.append({
                "row": row, "employee_id": employee_id, "issue": "at or past retirement age",
                "question": f"Employee is {age_now:.2f} at the valuation date, at or past the "
                            f"retirement age of {retirement_age}; please confirm whether they "
                            "are still working, already receiving a pension, or have left.",
                "blocking": False,
            })

    for query in queries:
        is_valued = not query["blocking"] and query["row"] not in blocked_rows
        query["effect"] = VALUED_IMMEDIATELY if is_valued else NOT_VALUED

    query_df = pd.DataFrame(queries, columns=["employee_id", "issue", "question", "effect"])

    # Anyone with a blocking question stays out of the valuation until the client answers
    valid = df.drop(index=list(blocked_rows)).reset_index(drop=True)

    # log_change was written for the portfolio, so rename its columns for employees. Naming the
    # columns keeps them present when the file is so clean that nothing was logged
    log_df = pd.DataFrame(log, columns=LOG_COLUMNS).rename(columns={"company_id": "employee_id"})
    log_df = log_df.drop(columns="month")

    return valid, log_df, query_df


def death_prob(age, a=ASSUMPTIONS):
    """Return the probability of dying within one year from a whole age."""
    if a["q_table"] is not None:
        base_rate = a["q_table"][age]
    else:
        # Makeham-style formula applied directly to q_x: A is constant risk, B * c^age grows
        # exponentially with age. (The textbook Makeham law describes the force of mortality)
        base_rate = a["makeham_A"] + a["makeham_B"] * a["makeham_c"] ** age
    # A probability can't exceed 1, but the formula keeps growing at very high ages
    return min(a["mortality_factor"] * base_rate, 1.0)


def alive_at(age, start_whole_age, a=ASSUMPTIONS):
    """Return the share alive at an exact age, out of 1 alive at a whole starting age."""
    whole_age = int(np.floor(age))
    part_year = age - whole_age

    alive = 1.0
    for year_of_age in range(start_whole_age, whole_age):
        alive *= 1 - death_prob(year_of_age, a)

    # Uniform distribution of deaths: 40% through a year of age, 40% of that year's deaths
    # have happened
    if part_year > 0:
        alive *= 1 - part_year * death_prob(whole_age, a)
    return alive


def survival_prob(from_age, to_age, a=ASSUMPTIONS):
    """Return the probability of being alive at to_age, given alive at from_age (exact ages)."""
    if to_age < from_age:
        raise ValueError(f"to_age ({to_age}) is before from_age ({from_age}).")
    # A ratio of one survivor curve, so splitting a period anywhere gives the same answer
    start_whole_age = int(np.floor(from_age))
    return alive_at(to_age, start_whole_age, a) / alive_at(from_age, start_whole_age, a)


def life_expectancy(age, a=ASSUMPTIONS):
    """Return the complete expectation of life at a whole age, ending at the maximum age."""
    # Under uniform deaths survival is a straight line within each year of age, so the
    # average of the two ends of a year is exact, not an approximation
    expected_years = 0.0
    for k in range(0, a["max_age"] - age):
        expected_years += (survival_prob(age, age + k, a) + survival_prob(age, age + k + 1, a)) / 2
    return expected_years


def timing_shift(a=ASSUMPTIONS):
    """Return (f - 1) / (2f): how much later, in years, instalments arrive on average."""
    payments = a["payments_per_year"]
    return (payments - 1) / (2 * payments)


def annuity_schedule(age, a=ASSUMPTIONS):
    """Return one row per yearly payment date, per 1 euro of starting pension, from an exact age."""
    increase = 1 + a["pension_increase"]
    shift = timing_shift(a)
    schedule = []

    # k = 0 is the payment right at the start (start of the first year), so it isn't discounted
    for k in range(0, int(a["max_age"] - age) + 1):
        survival = survival_prob(age, age + k, a)
        pension_level = increase ** k

        # The instalments of each year arrive later than its start. Added up over all years,
        # the effect telescopes: with no pension increases only the k = 0 term remains, which
        # is the standard -(f - 1) / (2f). Increases leave a small term in each later year
        if k == 0:
            adjustment = -shift
        else:
            adjustment = -shift * survival * (pension_level - increase ** (k - 1))

        schedule.append({
            "k": k,
            "survival": survival,
            "pension_level": pension_level,
            "expected_payment": survival * pension_level,
            "timing_adjustment": adjustment,
        })
    return schedule


def annuity_values(age, a=ASSUMPTIONS):
    """Return (annuity if paid once at the start of each year, annuity with instalments)."""
    v = 1 / (1 + a["discount_rate"])
    before_timing = 0.0
    with_timing = 0.0
    for item in annuity_schedule(age, a):
        discount = v ** item["k"]
        before_timing += discount * item["expected_payment"]
        with_timing += discount * (item["expected_payment"] + item["timing_adjustment"])
    return before_timing, with_timing


def annuity_before_timing(age, a=ASSUMPTIONS):
    """Return today's value of 1 euro a year for life, as if paid once at the start of each year."""
    return annuity_values(age, a)[0]


def annuity_factor(age, a=ASSUMPTIONS):
    """Return the annuity factor used in the valuation, allowing for the payment frequency."""
    return annuity_values(age, a)[1]


def exact_years(start_date, end_date):
    """Return the time between two dates in exact years, the model's one time basis."""
    return (end_date - start_date).days / DAYS_PER_YEAR


def dbo_employee(row, valuation_date=VALUATION_DATE, a=ASSUMPTIONS, known_annuities=None):
    """Return the DBO for one employee (projected unit credit) with every intermediate value."""
    valuation = pd.Timestamp(valuation_date)
    v = 1 / (1 + a["discount_rate"])

    age = exact_years(row["birth_date"], valuation)
    past_service = exact_years(row["entry_date"], valuation)

    # Anyone at or past the retirement age is valued as retiring on the valuation date. All
    # the formulas below then hold with no special cases: n = 0 means no salary growth, no
    # discounting and survival of 1, and the annuity starts at their exact age
    immediate_retirement = age >= a["retirement_age"]
    retirement_age_used = max(age, a["retirement_age"])
    years_to_retirement = retirement_age_used - age

    yearly_salary = row["monthly_salary_eur"] * SALARY_PAYMENTS_PER_YEAR
    projected_final_salary = yearly_salary * (1 + a["salary_growth"]) ** years_to_retirement

    # PUC: service earned SO FAR, valued at the salary expected AT RETIREMENT
    earned_pension = a["accrual_rate"] * past_service * projected_final_salary

    # Everyone retiring at the same age shares one annuity, so it needs calculating only once
    if known_annuities is None:
        known_annuities = {}
    if retirement_age_used not in known_annuities:
        known_annuities[retirement_age_used] = annuity_values(retirement_age_used, a)
    annual_annuity, annuity_at_retirement = known_annuities[retirement_age_used]
    discount_factor = v ** years_to_retirement
    survival_to_retirement = survival_prob(age, retirement_age_used, a)

    dbo = earned_pension * annuity_at_retirement * discount_factor * survival_to_retirement

    # Same PUC basis as the DBO, for the service still to be worked in the next year: a full
    # year, part of one for someone retiring within the year, nothing for someone already past
    service_next_year = min(1.0, years_to_retirement)
    service_cost = (a["accrual_rate"] * service_next_year * projected_final_salary
                    * annuity_at_retirement * discount_factor * survival_to_retirement)

    return {
        "employee_id": row["employee_id"],
        "age": age,
        "past_service": past_service,
        "retirement_age_used": retirement_age_used,
        "immediate_retirement": immediate_retirement,
        "years_to_retirement": years_to_retirement,
        "yearly_salary": yearly_salary,
        "projected_final_salary": projected_final_salary,
        "earned_pension": earned_pension,
        "annual_annuity": annual_annuity,
        "timing_adjustment": annuity_at_retirement - annual_annuity,
        "annuity_at_retirement": annuity_at_retirement,
        "discount_factor": discount_factor,
        "survival_to_retirement": survival_to_retirement,
        "dbo": dbo,
        "service_next_year": service_next_year,
        "service_cost": service_cost,
    }


def dbo_portfolio(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return (one row of DBO workings per employee, total DBO)."""
    known_annuities = {}
    rows = [dbo_employee(row, valuation_date, a, known_annuities) for _, row in valid_df.iterrows()]
    results = pd.DataFrame(rows)
    return results, results["dbo"].sum()


def total_dbo(valid_df, valuation_date, a):
    """Return only the total DBO for one set of assumptions."""
    return dbo_portfolio(valid_df, valuation_date, a)[1]


def interest_cost(dbo, service_cost, a=ASSUMPTIONS):
    """Return one year's interest on the DBO and on service cost valued at the start of the year."""
    return a["discount_rate"] * (dbo + service_cost)


def payments_due_next_year(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return the expected pension payments falling due within one year of the valuation date."""
    payments = payment_rows(valid_df, valuation_date, a)
    return payments.loc[payments["years_from_now"] < 1, "expected_payment"].sum()


def standard_outputs(results, valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return next year's service cost, interest cost and the size of the interest simplification."""
    total = results["dbo"].sum()
    service_cost_total = results["service_cost"].sum()
    payments = payments_due_next_year(valid_df, valuation_date, a)
    return {
        "service_cost": service_cost_total,
        "interest_cost": interest_cost(total, service_cost_total, a),
        "payments_next_year": payments,
        # Payments spread through the year would remove about half a year's interest on them
        "interest_on_payments": a["discount_rate"] * payments / 2,
    }


def mortality_factor_for_extra_life(extra_years, age=LIFE_EXPECTANCY_AGE, a=ASSUMPTIONS):
    """Return the mortality factor that adds extra_years to life expectancy at age, by bisection."""
    target = life_expectancy(age, a) + extra_years
    lower = 0.0
    upper = a["mortality_factor"]
    if life_expectancy(age, dict(a, mortality_factor=lower)) < target:
        raise ValueError(f"Life expectancy at {age} cannot be raised by {extra_years} years.")

    # Life expectancy falls as the factor rises, so each test halves the range that can hold
    # the answer: if life is still too long, the factor must be higher than the midpoint
    while upper - lower > BISECTION_TOLERANCE:
        middle = (lower + upper) / 2
        if life_expectancy(age, dict(a, mortality_factor=middle)) > target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def sensitivity_table(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return the total DBO when one assumption at a time is changed, vs the base."""
    shift = SENSITIVITY_SHIFT
    # (scenario name, assumption to change, new value); None means "change nothing"
    scenarios = [
        ("base", None, None),
        ("discount rate -0.5%", "discount_rate", a["discount_rate"] - shift),
        ("discount rate +0.5%", "discount_rate", a["discount_rate"] + shift),
        ("salary growth +0.5%", "salary_growth", a["salary_growth"] + shift),
        ("salary growth -0.5%", "salary_growth", a["salary_growth"] - shift),
        (LIFE_EXPECTANCY_SCENARIO, "mortality_factor", mortality_factor_for_extra_life(1.0, a=a)),
    ]

    rows = []
    for name, key, new_value in scenarios:
        if key is None:
            changed = dict(a)
        else:
            # dict(a, **{key: value}) copies a and replaces one setting; a itself is untouched
            changed = dict(a, **{key: new_value})
        rows.append({
            "scenario": name,
            "assumption_value": new_value,
            "total_dbo": total_dbo(valid_df, valuation_date, changed),
        })

    table = pd.DataFrame(rows)
    base_dbo = table.loc[table["scenario"] == "base", "total_dbo"].iloc[0]
    table["pct_change"] = (table["total_dbo"] / base_dbo - 1) * 100
    return table


def duration(table):
    """Return the DBO's duration: % change in DBO per 1 percentage point change in discount rate."""
    dbo_by_scenario = table.set_index("scenario")["total_dbo"]
    dbo_lower_rate = dbo_by_scenario["discount rate -0.5%"]
    dbo_higher_rate = dbo_by_scenario["discount rate +0.5%"]
    base_dbo = dbo_by_scenario["base"]
    # Central difference: the slope measured on both sides of the base, divided by the base
    return (dbo_lower_rate - dbo_higher_rate) / (2 * base_dbo * SENSITIVITY_SHIFT)


def dbo_monte_carlo_draws(valid_df, valuation_date=VALUATION_DATE, n_sims=500, seed=42,
                          a=ASSUMPTIONS):
    """Return the simulated total DBO for each random draw of discount rate and salary growth."""
    rng = np.random.default_rng(seed)
    # Drawn independently. In reality both move with inflation, so they tend to rise and fall
    # together and partly cancel out; independent draws therefore overstate the spread
    discount_rates = rng.normal(a["discount_rate"], 0.005, size=n_sims)
    salary_growths = rng.normal(a["salary_growth"], 0.005, size=n_sims)

    totals = []
    for discount_rate, salary_growth in zip(discount_rates, salary_growths):
        changed = dict(a, discount_rate=discount_rate, salary_growth=salary_growth)
        totals.append(total_dbo(valid_df, valuation_date, changed))
    return np.array(totals)


def dbo_monte_carlo(valid_df, valuation_date=VALUATION_DATE, n_sims=500, seed=42, a=ASSUMPTIONS):
    """Return the 10th, 50th and 90th percentile of total DBO under uncertain assumptions."""
    totals = dbo_monte_carlo_draws(valid_df, valuation_date, n_sims, seed, a)
    return dbo_percentiles(totals)


def dbo_percentiles(totals):
    """Return the 10th, 50th and 90th percentile of an array of simulated total DBOs."""
    p10, p50, p90 = np.percentile(totals, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def employee_day_counts(row, valuation_date=VALUATION_DATE):
    """Return (days alive, days employed) at the valuation date, the inputs to age and service."""
    valuation = pd.Timestamp(valuation_date)
    return (valuation - row["birth_date"]).days, (valuation - row["entry_date"]).days


def first_mortality_age(a=ASSUMPTIONS, default_age=20):
    """Return the youngest age the mortality basis can give a death probability for."""
    if a["q_table"] is not None:
        return max(default_age, min(a["q_table"]))
    return default_age


def mortality_table(a=ASSUMPTIONS, from_age=20, to_age=110):
    """Return whole age, one-year death probability q_x, and survival to each age."""
    start_age = max(from_age, first_mortality_age(a, from_age))
    rows = []
    for age in range(start_age, to_age + 1):
        rows.append({
            "age": age,
            "q_x": death_prob(age, a),
            "survival": survival_prob(start_age, age, a),
        })
    return pd.DataFrame(rows)


def survival_curve(exact_age, a=ASSUMPTIONS):
    """Return survival from an exact age to each later whole age, by the valuation's own method."""
    rows = [{"age": exact_age, "survival": 1.0}]
    for age in range(int(np.floor(exact_age)) + 1, a["max_age"] + 1):
        rows.append({"age": age, "survival": survival_prob(exact_age, age, a)})
    return pd.DataFrame(rows)


def validate_mortality_table(table_df, youngest_age, oldest_age):
    """Return (dictionary of age to q_x, list of problems) for an uploaded mortality table."""
    missing = [column for column in ["age", "q_x"] if column not in table_df.columns]
    if missing:
        return None, [f"Missing column(s): {', '.join(missing)}. Expected columns: age, q_x."]

    # to_numeric with coerce turns any text into a missing value, reported below
    ages = pd.to_numeric(table_df["age"], errors="coerce")
    rates = pd.to_numeric(table_df["q_x"], errors="coerce")

    problems = []
    if table_df.empty:
        problems.append("The table has no rows.")
    if ages.isna().any():
        problems.append("Some ages are blank or not numbers.")
    if rates.isna().any():
        problems.append("Some q_x values are blank or not numbers.")
    if problems:
        return None, problems

    if (ages != ages.round()).any():
        problems.append("Ages must be whole numbers.")
    if ages.duplicated().any():
        problems.append("Some ages appear more than once.")
    sorted_ages = ages.sort_values()
    if (sorted_ages.diff().dropna() != 1).any():
        problems.append("Ages must be continuous, with no gaps.")
    if (rates < 0).any() or (rates > 1).any():
        problems.append("Every q_x must be between 0 and 1.")

    first_needed = min(youngest_age, MORTALITY_REFERENCE_AGE)
    if ages.min() > first_needed:
        problems.append(f"The table must start at age {first_needed} or younger.")
    last_needed = max(oldest_age + 1, MIN_TABLE_LAST_AGE)
    if ages.max() < last_needed:
        problems.append(f"The table must reach at least age {last_needed}, or the annuity would "
                        "stop too early.")
    if problems:
        return None, problems

    return {int(age): float(rate) for age, rate in zip(ages, rates)}, []


def employee_age_range(valid_df, valuation_date=VALUATION_DATE):
    """Return the youngest and oldest whole ages among the valid employees."""
    valuation = pd.Timestamp(valuation_date)
    ages = (valuation - valid_df["birth_date"]).dt.days / DAYS_PER_YEAR
    return int(np.floor(ages.min())), int(np.floor(ages.max()))


def annuity_contributions(age, a=ASSUMPTIONS):
    """Return each payment's survival, discount factor and contribution to the annuity factor."""
    v = 1 / (1 + a["discount_rate"])
    rows = []
    running_total = 0.0
    for item in annuity_schedule(age, a):
        discount = v ** item["k"]
        contribution = item["expected_payment"] * discount
        running_total += contribution
        rows.append({
            "payment_age": age + item["k"],
            "survival": item["survival"],
            "discount": discount,
            "pension_level": item["pension_level"],
            "contribution": contribution,
            "running_total": running_total,
        })
    return pd.DataFrame(rows)


def payment_rows(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return every expected payment and timing adjustment per employee, at its exact time."""
    valuation = pd.Timestamp(valuation_date)
    v = 1 / (1 + a["discount_rate"])
    schedules = {}
    known_annuities = {}
    payments = []

    for _, row in valid_df.iterrows():
        result = dbo_employee(row, valuation_date, a, known_annuities)
        start_age = result["retirement_age_used"]
        # Everyone retiring at the same age shares one schedule, so build it only once
        if start_age not in schedules:
            schedules[start_age] = annuity_schedule(start_age, a)

        scale = result["earned_pension"] * result["survival_to_retirement"]
        for item in schedules[start_age]:
            years_from_now = result["years_to_retirement"] + item["k"]
            expected = scale * item["expected_payment"]
            # The approximation isn't a list of dated instalments, so its effect is carried as
            # its own labelled amount. Dating it here keeps the total equal to the DBO
            adjustment = scale * item["timing_adjustment"]
            payment_date = valuation + pd.Timedelta(days=years_from_now * DAYS_PER_YEAR)
            payments.append({
                "year": payment_date.year,
                "years_from_now": years_from_now,
                "expected_payment": expected,
                "timing_adjustment": adjustment,
                "discounted_payment": (expected + adjustment) * v ** years_from_now,
            })
    return pd.DataFrame(payments)


def expected_cash_flows(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return expected pension payments per calendar year, undiscounted and discounted."""
    payments = payment_rows(valid_df, valuation_date, a)
    columns = ["expected_payment", "timing_adjustment", "discounted_payment"]
    by_year = payments.groupby("year")[columns].sum()
    return by_year.reset_index()


def weighted_average_payment_time(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return the average years until payment, weighted by present value, at exact times."""
    payments = payment_rows(valid_df, valuation_date, a)
    weighted_time = (payments["years_from_now"] * payments["discounted_payment"]).sum()
    return weighted_time / payments["discounted_payment"].sum()


def exact_rate_move(valid_df, valuation_date, a, move):
    """Return (total DBO, % change) from recalculating with the discount rate moved by move."""
    base_total = total_dbo(valid_df, valuation_date, a)
    moved_total = total_dbo(valid_df, valuation_date,
                            dict(a, discount_rate=a["discount_rate"] + move))
    return moved_total, (moved_total / base_total - 1) * 100


def top_n_share(results, n=5):
    """Return the share (%) of total DBO held by the n employees with the largest DBO."""
    return results["dbo"].nlargest(n).sum() / results["dbo"].sum() * 100


# ---------------------------------------------------------------------------------------------
# Model checks. Each check returns (name, passed, detail) so the same result can be shown by
# pytest and, later, on the pension page.
# ---------------------------------------------------------------------------------------------

# One cent: far below any displayed amount, but well above floating-point noise on millions
EUR_TOLERANCE = 0.01
# For factors like the annuity, which are numbers near 14 rather than euro amounts
FACTOR_TOLERANCE = 1e-9
# The two duration measures are not the same formula, so they only have to be close
DURATION_GAP_LIMIT_YEARS = 0.25
# A rate step so small that the curve of DBO against rate is effectively a straight line
SMALL_RATE_SHIFT = 0.0001
SMALL_STEP_GAP_LIMIT_YEARS = 0.01
LIFE_EXPECTANCY_TOLERANCE = 1e-6
# Same range as the retirement-age slider on the pension page
RETIREMENT_AGES_CHECKED = range(60, 68)


def employees_below_age(valid_df, age_limit, valuation_date=VALUATION_DATE):
    """Return the employees younger than age_limit at the valuation date, in exact years."""
    valuation = pd.Timestamp(valuation_date)
    age = (valuation - valid_df["birth_date"]).dt.days / DAYS_PER_YEAR
    return valid_df[age < age_limit].reset_index(drop=True)


def cash_flow_duration(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return (present-value-weighted average years to payment, that average divided by 1 + i)."""
    average_time = weighted_average_payment_time(valid_df, valuation_date, a)
    return average_time, average_time / (1 + a["discount_rate"])


def check_cash_flows(valued, valuation_date, a, total):
    """Check 1: the discounted payments and timing adjustment add up to the total DBO."""
    flows = expected_cash_flows(valued, valuation_date, a)
    present_value = flows["discounted_payment"].sum()
    gap = present_value - total
    passed = abs(gap) <= EUR_TOLERANCE
    detail = (f"Present value of cash flows, including the payment timing adjustment, "
              f"€{present_value:,.2f}; total DBO €{total:,.2f}; gap €{gap:.6f}.")
    return "1 Cash flows: present value equals total DBO", bool(passed), detail


def annuity_closed_form(age, a):
    """Return the textbook value of the annuity when nobody dies, with increases and instalments."""
    payments = a["max_age"] - age + 1
    increase_rate = a["pension_increase"]
    growth_over_discount = (1 + increase_rate) / (1 + a["discount_rate"])

    # A geometric series 1 + u + ... + u^(N-1); when u = 1 every one of its N terms is 1
    if abs(growth_over_discount - 1) < FACTOR_TOLERANCE:
        sum_from_0 = float(payments)
        sum_from_1 = float(payments - 1)
    else:
        sum_from_0 = (1 - growth_over_discount ** payments) / (1 - growth_over_discount)
        sum_from_1 = sum_from_0 - 1

    return sum_from_0 - timing_shift(a) * (1 + increase_rate / (1 + increase_rate) * sum_from_1)


def check_annuity_closed_form(a):
    """Check 2: with nobody dying, the annuity equals the textbook annuity-due formula."""
    retirement_age = a["retirement_age"]
    no_deaths = dict(a, mortality_factor=0.0)
    model = annuity_factor(retirement_age, no_deaths)
    closed_form = annuity_closed_form(retirement_age, no_deaths)
    gap = model - closed_form
    passed = abs(gap) <= FACTOR_TOLERANCE
    detail = (f"Model {model:.10f}; closed form {closed_form:.10f}; gap {gap:.2e}. Uses "
              f"{a['payments_per_year']} payments a year and {a['pension_increase']:.2%} pension "
              "increases.")
    return "2 Zero mortality: annuity equals annuity-due formula", bool(passed), detail


def check_zero_service(valued, valuation_date, a):
    """Check 3: employees who start on the valuation date have earned nothing yet."""
    just_joined = valued.copy()
    just_joined["entry_date"] = pd.Timestamp(valuation_date)
    zero_service_total = total_dbo(just_joined, valuation_date, a)
    passed = abs(zero_service_total) <= EUR_TOLERANCE
    detail = f"Total DBO with zero past service for all {len(valued)} employees: €{zero_service_total:.2f}."
    return "3 Zero past service gives DBO 0", bool(passed), detail


def check_accrual_doubling(valued, valuation_date, a, results):
    """Check 4: the pension is proportional to the accrual rate, so the DBO must be too."""
    doubled_a = dict(a, accrual_rate=a["accrual_rate"] * 2)
    doubled = dbo_portfolio(valued, valuation_date, doubled_a)[0]
    largest_gap = (doubled["dbo"] - 2 * results["dbo"]).abs().max()
    passed = largest_gap <= EUR_TOLERANCE
    detail = (f"Largest gap for any employee between doubled-accrual DBO and 2 x DBO: "
              f"€{largest_gap:.6f}. Total goes to €{doubled['dbo'].sum():,.2f}.")
    return "4 Doubling the accrual rate doubles the DBO", bool(passed), detail


def direction_check(name, valued, valuation_date, a, key, new_value, should_rise, total):
    """Return a check that changing one assumption moves the total DBO the expected way."""
    new_total = total_dbo(valued, valuation_date, dict(a, **{key: new_value}))
    if should_rise:
        passed = new_total > total
    else:
        passed = new_total < total
    detail = (f"{key} from {a[key]:g} to {new_value:g}: total DBO €{total:,.0f} -> "
              f"€{new_total:,.0f} ({(new_total / total - 1) * 100:+.2f}%).")
    return name, bool(passed), detail


def falls_at_every_step(totals_by_age):
    """Return True if each total is lower than the total at the previous age."""
    totals = list(totals_by_age.values())
    return all(later < earlier for earlier, later in zip(totals, totals[1:]))


def format_totals_by_age(totals_by_age):
    """Return totals per retirement age as one line of text."""
    return "; ".join(f"{age}: €{total:,.0f}" for age, total in totals_by_age.items())


def retirement_age_direction_result(totals_by_age, a):
    """Return (passed, note): a later retirement age must lower the DBO only if g <= i."""
    if a["salary_growth"] <= a["discount_rate"]:
        return falls_at_every_step(totals_by_age), ""
    # Waiting a year then adds more through salary growth than discounting removes, so the DBO
    # may genuinely rise. The numbers are still reported; a rise here isn't a model error
    note = (" Not a rule at these settings: salary growth is above the discount rate, so a "
            "later retirement age can correctly raise the DBO.")
    return True, note


def check_retirement_age_fixed_group(valid_df, valuation_date, a):
    """Check 5d: for the same people, a later retirement age lowers the DBO."""
    name = "5d Retirement age up: DBO down (employees under every age tested)"
    youngest_age_checked = RETIREMENT_AGES_CHECKED[0]
    # Only people below every age tested, so nobody changes to immediate retirement between steps
    fixed_group = employees_below_age(valid_df, youngest_age_checked, valuation_date)
    if fixed_group.empty:
        return name, False, f"Could not be run: no employee is younger than {youngest_age_checked}."

    totals_by_age = {}
    for age in RETIREMENT_AGES_CHECKED:
        totals_by_age[age] = total_dbo(fixed_group, valuation_date, dict(a, retirement_age=age))
    passed, note = retirement_age_direction_result(totals_by_age, a)
    detail = (f"{len(fixed_group)} employees under {youngest_age_checked}. "
              f"{format_totals_by_age(totals_by_age)}.{note}")
    return name, passed, detail


def check_retirement_age_page_total(valid_df, valuation_date, a):
    """Check 5e: the total the page reports also falls as the retirement age rises."""
    totals_by_age = {}
    immediate_by_age = {}
    for age in RETIREMENT_AGES_CHECKED:
        results, total = dbo_portfolio(valid_df, valuation_date, dict(a, retirement_age=age))
        totals_by_age[age] = total
        immediate_by_age[age] = int(results["immediate_retirement"].sum())
    detail = (f"All {len(valid_df)} valid employees stay in at every age. "
              f"{format_totals_by_age(totals_by_age)}. Valued as immediate retirement: "
              f"{immediate_by_age}.")
    passed, note = retirement_age_direction_result(totals_by_age, a)
    return "5e Retirement age up: page total down", passed, detail + note


def check_sum_of_employees(valued, valuation_date, a, total):
    """Check 6: recalculating each employee on their own adds up to the reported total."""
    employee_dbos = []
    for _, row in valued.iterrows():
        employee_dbos.append(dbo_employee(row, valuation_date, a)["dbo"])
    employee_dbos = np.array(employee_dbos)

    # pandas sum() skips missing values, which would hide an employee with no DBO
    missing = int(np.isnan(employee_dbos).sum())
    gap = employee_dbos.sum() - total
    passed = missing == 0 and abs(gap) <= EUR_TOLERANCE
    rounding_gap = np.round(employee_dbos).sum() - round(total)
    detail = (f"Sum of {len(employee_dbos)} employee DBOs minus total: €{gap:.6f}; employees with "
              f"no DBO: {missing}. Separately, the whole-euro figures shown on screen add up to "
              f"€{rounding_gap:+,.0f} against the rounded total, which is rounding only.")
    return "6 Sum of employee DBOs equals the total", bool(passed), detail


def small_step_duration(valued, valuation_date, a):
    """Return the sensitivity-based duration using a very small change in the discount rate."""
    shift = SMALL_RATE_SHIFT
    total_lower = total_dbo(valued, valuation_date,
                            dict(a, discount_rate=a["discount_rate"] - shift))
    total_higher = total_dbo(valued, valuation_date,
                             dict(a, discount_rate=a["discount_rate"] + shift))
    base = total_dbo(valued, valuation_date, a)
    return (total_lower - total_higher) / (2 * base * shift)


def check_duration(valued, valuation_date, a):
    """Check 7: duration from the sensitivities is close to duration from the cash flows."""
    from_sensitivity = duration(sensitivity_table(valued, valuation_date, a))
    average_time, from_cash_flows = cash_flow_duration(valued, valuation_date, a)
    small_step = small_step_duration(valued, valuation_date, a)
    gap = from_sensitivity - from_cash_flows
    small_step_gap = small_step - from_cash_flows

    passed = abs(gap) <= DURATION_GAP_LIMIT_YEARS and abs(small_step_gap) <= SMALL_STEP_GAP_LIMIT_YEARS
    detail = (f"From +/-0.5% sensitivities: {from_sensitivity:.4f}. From discounted cash flows: "
              f"{average_time:.4f} / {1 + a['discount_rate']:g} = {from_cash_flows:.4f}. "
              f"Gap {gap:+.4f} years. With a +/-{SMALL_RATE_SHIFT:.2%} step the sensitivity "
              f"measure is {small_step:.4f}, gap {small_step_gap:+.4f}.")
    if passed:
        detail += (" Almost all of the gap therefore comes from the size of the 0.5% step (the "
                   "DBO curves against the rate), not from a calculation difference.")
    return "7 Duration: sensitivities close to cash flows", bool(passed), detail


def count_survival_problems(path):
    """Return (values outside 0 to 1, steps that fail to decrease) for one survival path."""
    path = np.array(path)
    outside = int(((path < 0) | (path > 1)).sum())
    steps = np.diff(path)
    # Once survival has reached exactly 0 it cannot fall any further, so that isn't a failure
    not_falling = int(((steps >= 0) & (path[:-1] > 0)).sum())
    return outside, not_falling


def check_survival(valued, valuation_date, a):
    """Check 8: every survival probability is between 0 and 1 and falls as age rises."""
    max_age = a["max_age"]
    start_age = first_mortality_age(a)
    death_probs = [death_prob(age, a) for age in range(start_age, max_age + 1)]
    bad_death_probs = sum(1 for q in death_probs if q < 0 or q > 1)

    # The whole-age path, as on the mortality chart
    paths = [[survival_prob(start_age, age, a) for age in range(start_age, max_age + 1)]]

    # Each employee's own path from their exact age, in half-years to include part-year steps
    valuation = pd.Timestamp(valuation_date)
    for _, row in valued.iterrows():
        age = exact_years(row["birth_date"], valuation)
        steps = int((max_age - age) * 2)
        paths.append([survival_prob(age, age + step / 2, a) for step in range(0, steps + 1)])

    outside = 0
    not_falling = 0
    tested = 0
    for path in paths:
        path_outside, path_not_falling = count_survival_problems(path)
        outside += path_outside
        not_falling += path_not_falling
        tested += len(path)

    passed = bad_death_probs == 0 and outside == 0 and not_falling == 0
    detail = (f"{tested:,} survival probabilities on {len(paths)} paths: {outside} outside 0 to 1, "
              f"{not_falling} steps that did not fall. {len(death_probs)} death probabilities: "
              f"{bad_death_probs} outside 0 to 1.")
    return "8 Survival probabilities valid and decreasing", passed, detail


def check_survival_self_consistent(valued, valuation_date, a):
    """Check 9: splitting a period at any point gives the same survival as not splitting it."""
    valuation = pd.Timestamp(valuation_date)
    largest_gap = 0.0
    comparisons = 0
    for _, row in valued.iterrows():
        age = exact_years(row["birth_date"], valuation)
        end_age = min(age + 20.5, a["max_age"])
        if end_age <= age:
            continue
        direct = survival_prob(age, end_age, a)
        # The next birthday and a midpoint, which usually falls part-way through a year
        for split_age in [min(np.floor(age) + 1, end_age), (age + end_age) / 2]:
            split = survival_prob(age, split_age, a) * survival_prob(split_age, end_age, a)
            largest_gap = max(largest_gap, abs(direct - split))
            comparisons += 1

    passed = bool(largest_gap <= FACTOR_TOLERANCE)
    detail = f"{comparisons} split periods compared; largest difference {largest_gap:.2e}."
    return "9 Survival: splitting a period changes nothing", passed, detail


def check_payment_frequency(a):
    """Check 10: with no increases, instalments change the annuity by exactly (f - 1) / (2f)."""
    retirement_age = a["retirement_age"]
    level = dict(a, pension_increase=0.0)
    once_a_year = annuity_factor(retirement_age, dict(level, payments_per_year=1))

    largest_gap = 0.0
    results = []
    for payments in PAYMENT_FREQUENCIES:
        model = annuity_factor(retirement_age, dict(level, payments_per_year=payments))
        expected = once_a_year - (payments - 1) / (2 * payments)
        largest_gap = max(largest_gap, abs(model - expected))
        results.append(f"{payments} a year: {model:.4f}")

    passed = bool(largest_gap <= FACTOR_TOLERANCE)
    detail = f"{'; '.join(results)}. Largest gap from the formula {largest_gap:.2e}."
    return "10 Payment frequency matches the standard formula", passed, detail


def check_service_cost(valued, valuation_date, a, results):
    """Check 11: service cost equals the extra DBO from more service, for the time still worked."""
    # 1,461 days is exactly 4 x 365.25, so the extra service is exactly 4 years
    longer_service = valued.copy()
    longer_service["entry_date"] = longer_service["entry_date"] - pd.Timedelta(days=1461)
    longer = dbo_portfolio(longer_service, valuation_date, a)[0]

    extra_dbo_per_year = (longer["dbo"] - results["dbo"]) / 4
    expected = extra_dbo_per_year * results["service_next_year"]
    largest_gap = (expected - results["service_cost"]).abs().max()
    passed = largest_gap <= EUR_TOLERANCE
    detail = (f"Service cost €{results['service_cost'].sum():,.2f}; largest gap for any employee "
              f"from the DBO recalculated with more service: €{largest_gap:.6f}.")
    return "11 Service cost equals the DBO per year of service", bool(passed), detail


def check_roll_forward(a, results):
    """Check 12: DBO + service cost + interest equals the expected DBO one year later."""
    name = "12 Roll-forward: DBO + service cost + interest"
    v = 1 / (1 + a["discount_rate"])
    # Someone retiring within the year starts being paid during it, which this identity excludes
    at_least_a_year = results[results["years_to_retirement"] >= 1]
    if at_least_a_year.empty:
        return name, False, "Could not be run: nobody is at least one year from retirement."

    largest_gap = 0.0
    rolled_total = 0.0
    for _, r in at_least_a_year.iterrows():
        age_next_year = r["age"] + 1
        pension_next_year = (a["accrual_rate"] * (r["past_service"] + 1)
                             * r["projected_final_salary"])
        dbo_next_year = (pension_next_year * r["annuity_at_retirement"]
                         * v ** (r["years_to_retirement"] - 1)
                         * survival_prob(age_next_year, r["retirement_age_used"], a))
        expected_next_year = survival_prob(r["age"], age_next_year, a) * dbo_next_year
        rolled = r["dbo"] + r["service_cost"] + interest_cost(r["dbo"], r["service_cost"], a)
        largest_gap = max(largest_gap, abs(expected_next_year - rolled))
        rolled_total += rolled

    passed = bool(largest_gap <= EUR_TOLERANCE)
    detail = (f"{len(at_least_a_year)} employees at least a year from retirement: DBO + service "
              f"cost + interest = €{rolled_total:,.2f}, and the largest gap for any employee from "
              f"their expected DBO one year later is €{largest_gap:.6f}.")
    return name, passed, detail


def check_extra_life(a):
    """Check 13: the bisection's mortality factor really adds one year of life expectancy."""
    factor = mortality_factor_for_extra_life(1.0, a=a)
    before = life_expectancy(LIFE_EXPECTANCY_AGE, a)
    after = life_expectancy(LIFE_EXPECTANCY_AGE, dict(a, mortality_factor=factor))
    gap = after - (before + 1)
    passed = bool(abs(gap) <= LIFE_EXPECTANCY_TOLERANCE)
    detail = (f"Mortality factor {a['mortality_factor']:g} -> {factor:.6f}: life expectancy at "
              f"{LIFE_EXPECTANCY_AGE} from {before:.4f} to {after:.4f}; gap from +1 year {gap:.2e}.")
    return "13 Life expectancy +1 year: factor adds exactly one year", passed, detail


def run_model_checks(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return a list of (check name, passed, detail) for one set of data and assumptions."""
    if valid_df.empty:
        raise ValueError("No employees to check with these settings.")
    results, total = dbo_portfolio(valid_df, valuation_date, a)
    shift = SENSITIVITY_SHIFT

    return [
        check_cash_flows(valid_df, valuation_date, a, total),
        check_annuity_closed_form(a),
        check_zero_service(valid_df, valuation_date, a),
        check_accrual_doubling(valid_df, valuation_date, a, results),
        direction_check("5a Discount rate up: DBO down", valid_df, valuation_date, a,
                        "discount_rate", a["discount_rate"] + shift, False, total),
        direction_check("5b Salary growth up: DBO up", valid_df, valuation_date, a,
                        "salary_growth", a["salary_growth"] + shift, True, total),
        direction_check("5c Mortality factor up: DBO down", valid_df, valuation_date, a,
                        "mortality_factor", a["mortality_factor"] * 1.1, False, total),
        check_retirement_age_fixed_group(valid_df, valuation_date, a),
        check_retirement_age_page_total(valid_df, valuation_date, a),
        direction_check("5f Pension increase up: DBO up", valid_df, valuation_date, a,
                        "pension_increase", a["pension_increase"] + shift, True, total),
        check_sum_of_employees(valid_df, valuation_date, a, total),
        check_duration(valid_df, valuation_date, a),
        check_survival(valid_df, valuation_date, a),
        check_survival_self_consistent(valid_df, valuation_date, a),
        check_payment_frequency(a),
        check_service_cost(valid_df, valuation_date, a, results),
        check_roll_forward(a, results),
        check_extra_life(a),
    ]


# ---------------------------------------------------------------------------------------------
# Run record and exports
# ---------------------------------------------------------------------------------------------

def short_hash(text):
    """Return the first 12 characters of a SHA-256 hash, enough to tell two inputs apart."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def input_data_hash(raw_df):
    """Return a short hash of the employee file as it was received, before any cleaning."""
    return short_hash(raw_df.to_csv(index=False))


def mortality_table_hash(a=ASSUMPTIONS):
    """Return a short hash of an uploaded mortality table, or a note if the formula is in use."""
    if a["q_table"] is None:
        return "not applicable (formula in use)"
    return short_hash(repr(sorted(a["q_table"].items())))


def assumptions_table(a=ASSUMPTIONS):
    """Return every assumption of a valuation as rows of (assumption, value, note)."""
    formula_in_use = a["q_table"] is None
    not_used = "not used (table uploaded)"
    rows = [
        ("discount_rate", a["discount_rate"], "Yearly rate that converts future payments to today"),
        ("salary_growth", a["salary_growth"], "Yearly salary increase until retirement"),
        ("retirement_age", a["retirement_age"], "Older employees are valued as retiring today"),
        ("accrual_rate", a["accrual_rate"], "Share of final salary earned per year of service"),
        ("salary_payments_per_year", SALARY_PAYMENTS_PER_YEAR, "Monthly salary x this = yearly salary"),
        ("payments_per_year", a["payments_per_year"], "Instalments of the pension per year"),
        ("pension_increase", a["pension_increase"], "Yearly increase once the pension is being paid"),
        ("mortality_source", a["mortality_source"], "Illustrative unless a table was uploaded"),
        ("mortality_table_hash", mortality_table_hash(a), "Identifies an uploaded table"),
        ("mortality_factor", a["mortality_factor"], "Scales every death probability"),
        ("makeham_A", a["makeham_A"] if formula_in_use else not_used, "Constant risk in the formula"),
        ("makeham_B", a["makeham_B"] if formula_in_use else not_used, "Age-related scale in the formula"),
        ("makeham_c", a["makeham_c"] if formula_in_use else not_used, "Yearly growth of the age-related risk"),
        ("max_age", a["max_age"], "Nobody is assumed to live beyond this age"),
        ("days_per_year", DAYS_PER_YEAR, "Exact years = days between dates / this"),
    ]
    return pd.DataFrame(rows, columns=["assumption", "value", "note"])


def run_record(raw_df, valid_df, outputs, total, duration_years, a, valuation_date, timestamp):
    """Return the record of one valuation run as rows of (item, value)."""
    rows = [("valuation_date", str(valuation_date)), ("run_timestamp", timestamp)]
    for _, row in assumptions_table(a).iterrows():
        rows.append((row["assumption"], row["value"]))
    rows += [
        ("input_data_hash", input_data_hash(raw_df)),
        ("valid_employees", len(valid_df)),
        ("total_dbo", float(total)),
        ("service_cost_next_year", float(outputs["service_cost"])),
        ("interest_cost_next_year", float(outputs["interest_cost"])),
        ("duration_years", float(duration_years)),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def employee_export_table(results, valid_df):
    """Return each employee's input data followed by every intermediate step of the valuation."""
    inputs = valid_df[REQUIRED_EMPLOYEE_COLUMNS]
    return inputs.merge(results, on="employee_id")


def format_sheet(sheet):
    """Keep client text as text, freeze the header row and widen columns to fit their contents."""
    sheet.freeze_panes = "A2"
    for column in sheet.columns:
        longest = 0
        for cell in column:
            # A client value starting with "=" must stay text and never run as a formula
            if cell.data_type == "f":
                cell.data_type = "s"
            longest = max(longest, len(str(cell.value)) if cell.value is not None else 0)
        sheet.column_dimensions[column[0].column_letter].width = min(longest + 2, 70)


def excel_workbook(employee_table, assumptions, record, queries):
    """Return an Excel workbook as bytes, with one sheet each for the four tables."""
    sheets = {
        "Employees": employee_table,
        "Assumptions": assumptions,
        "Run record": record,
        "Client queries": queries,
    }
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
            format_sheet(writer.sheets[name])
    return buffer.getvalue()


if __name__ == "__main__":
    employees = generate_employees()
    make_messy_employees(employees)
    raw = pd.read_csv(DATA_DIR / "employees_raw.csv")
    valid, log_df, query_df = validate_employees(raw)

    table = sensitivity_table(valid)
    print(f"Sensitivity of total DBO ({len(valid)} valid employees, illustrative assumptions):")
    for _, row in table.iterrows():
        value = "" if pd.isna(row["assumption_value"]) else f"{row['assumption_value']:.3f}"
        print(f"  {row['scenario']:<22} {value:>6}  €{row['total_dbo']:>11,.0f}  "
              f"{row['pct_change']:+6.1f}%")

    print(f"\nDuration: {duration(table):.1f} years")

    percentiles = dbo_monte_carlo(valid)
    print("\nMonte Carlo of total DBO (500 draws of discount rate and salary growth):")
    print(f"  P10: €{percentiles['p10']:,.0f}")
    print(f"  P50: €{percentiles['p50']:,.0f}")
    print(f"  P90: €{percentiles['p90']:,.0f}")
