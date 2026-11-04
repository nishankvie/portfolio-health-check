"""Synthetic employee data for a pension provision example. All employees are fictional."""

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

# Valuation assumptions. ALL ARE ILLUSTRATIVE, chosen to show the method, not to value a real plan.
# Mortality uses a simple Makeham formula; a real Austrian valuation would use the official
# AVÖ 2018-P tables (separate for men and women). Turnover (employees leaving before 65) and
# death benefits before retirement are ignored, which overstates the DBO
ASSUMPTIONS = {
    "discount_rate": 0.035,     # yearly rate used to convert future payments into today's euros
    "salary_growth": 0.025,     # yearly salary increase until retirement
    "retirement_age": RETIREMENT_AGE,
    "accrual_rate": 0.01,       # 1% of final salary earned per year of service
    "makeham_A": 0.0002,        # age-independent death risk (accidents, illness)
    "makeham_B": 0.000035,      # size of the age-related death risk
    "makeham_c": 1.095,         # age-related risk grows by 9.5% per year of age
    "mortality_factor": 1.0,    # scales every death probability, e.g. 0.9 = people live longer
    "max_age": 110,             # nobody is assumed to live beyond this age
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
    """Return a salary as a float, parsing Austrian text like '€4.200,00' with parse_eur."""
    if not isinstance(value, str):
        return value
    # parse_eur treats every dot as a thousands separator, which is right for "€4.200,00"
    # but would turn a plain "5200.0" (as written by pandas) into 52000. Only Austrian text
    # (with € or a decimal comma) goes through it; plain numbers use float()
    if "€" in value or "," in value:
        return parse_eur(value)
    return float(value)


def validate_employees(raw_df, valuation_date=VALUATION_DATE):
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

    df["birth_date"] = pd.to_datetime(df["birth_date"], format="%Y-%m-%d")
    df["entry_date"] = pd.to_datetime(df["entry_date"], format="%Y-%m-%d")

    # Exact duplicates: the same person listed twice would double their liability
    exact_duplicate = df.duplicated()
    for row in df.index[exact_duplicate]:
        log_change(log, df.loc[row, "employee_id"], None, "all", "exact duplicate", None, "dropped")
    df = df[~exact_duplicate]

    # Text salaries: converting a format is not guessing, so this is fixed, not queried
    for row in df.index:
        old_value = df.loc[row, "monthly_salary_eur"]
        if isinstance(old_value, str) and ("€" in old_value or "," in old_value):
            log_change(log, df.loc[row, "employee_id"], None, "monthly_salary_eur",
                       "text number", old_value, parse_salary(old_value))
    df["monthly_salary_eur"] = df["monthly_salary_eur"].apply(parse_salary).astype(float)

    # Plausibility checks: every failure becomes a question for the client
    queries = []
    for row in df.index:
        employee_id = df.loc[row, "employee_id"]
        birth = df.loc[row, "birth_date"]
        entry = df.loc[row, "entry_date"]

        if entry < birth:
            queries.append({
                "employee_id": employee_id,
                "issue": "entry before birth",
                "question": f"Entry date {entry:%d.%m.%Y} is before birth date {birth:%d.%m.%Y}; "
                            "dates possibly swapped, please confirm both dates.",
            })
        # elif: with swapped dates the entry age is negative, which would repeat the same problem
        elif age_in_years(birth, entry) < MIN_WORKING_AGE:
            queries.append({
                "employee_id": employee_id,
                "issue": "entry age under 15",
                "question": f"Entry on {entry:%d.%m.%Y} means joining at age "
                            f"{age_in_years(birth, entry)}, below the minimum working age of "
                            f"{MIN_WORKING_AGE}; please confirm the entry date.",
            })

        if pd.isna(df.loc[row, "monthly_salary_eur"]):
            queries.append({
                "employee_id": employee_id,
                "issue": "missing salary",
                "question": "Monthly salary is missing; please provide the current gross "
                            "monthly salary.",
            })

        age_now = age_in_years(birth, valuation)
        if age_now >= RETIREMENT_AGE:
            queries.append({
                "employee_id": employee_id,
                "issue": "age 65 or over",
                "question": f"Employee is {age_now} at the valuation date, past the retirement "
                            f"age of {RETIREMENT_AGE}; please confirm whether they are still "
                            "working, already receiving a pension, or have left.",
            })

    query_df = pd.DataFrame(queries, columns=["employee_id", "issue", "question"])

    # Anyone with an open question stays out of the valuation until the client answers
    valid = df[~df["employee_id"].isin(query_df["employee_id"])].reset_index(drop=True)

    # log_change was written for the portfolio, so rename its columns for employees
    log_df = pd.DataFrame(log).rename(columns={"company_id": "employee_id"}).drop(columns="month")

    return valid, log_df, query_df


def death_prob(age, a=ASSUMPTIONS):
    """Return the probability of dying within one year at a given age (Makeham)."""
    # Makeham: q = A + B * c^age. A is constant risk, B * c^age grows exponentially with age
    q = a["mortality_factor"] * (a["makeham_A"] + a["makeham_B"] * a["makeham_c"] ** age)
    # A probability can't exceed 1, but the formula keeps growing at very high ages
    return min(q, 1.0)


def survival_prob(from_age, to_age, a=ASSUMPTIONS):
    """Return the probability of being alive at to_age, given alive at from_age."""
    survival = 1.0
    years = to_age - from_age
    whole_years = int(years)

    # Survive each whole year in turn: probabilities of independent steps multiply
    for k in range(whole_years):
        survival *= 1 - death_prob(from_age + k, a)

    # A leftover part-year (e.g. 0.7 of a year) uses the yearly survival to that power
    fraction = years - whole_years
    if fraction > 0:
        survival *= (1 - death_prob(from_age + whole_years, a)) ** fraction

    return survival


def annuity_factor(age, a=ASSUMPTIONS):
    """Return today's value of 1 euro a year for life, paid at the start of each year."""
    v = 1 / (1 + a["discount_rate"])
    factor = 0.0
    # k = 0 is the payment right now (start of the first year), so it isn't discounted
    for k in range(0, a["max_age"] - age + 1):
        factor += v ** k * survival_prob(age, age + k, a)
    return factor


def dbo_employee(row, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return the DBO for one employee (projected unit credit) with every intermediate value."""
    valuation = pd.Timestamp(valuation_date)
    v = 1 / (1 + a["discount_rate"])

    # Decimal years, because service and discounting accrue continuously, not on birthdays
    age = (valuation - row["birth_date"]).days / DAYS_PER_YEAR
    past_service = (valuation - row["entry_date"]).days / DAYS_PER_YEAR
    years_to_retirement = a["retirement_age"] - age

    yearly_salary = row["monthly_salary_eur"] * 14
    projected_final_salary = yearly_salary * (1 + a["salary_growth"]) ** years_to_retirement

    # PUC: service earned SO FAR, valued at the salary expected AT RETIREMENT
    earned_pension = a["accrual_rate"] * past_service * projected_final_salary

    annuity_at_retirement = annuity_factor(a["retirement_age"], a)
    discount_factor = v ** years_to_retirement
    survival_to_retirement = survival_prob(age, a["retirement_age"], a)

    dbo = earned_pension * annuity_at_retirement * discount_factor * survival_to_retirement

    return {
        "employee_id": row["employee_id"],
        "age": age,
        "past_service": past_service,
        "years_to_retirement": years_to_retirement,
        "yearly_salary": yearly_salary,
        "projected_final_salary": projected_final_salary,
        "earned_pension": earned_pension,
        "annuity_at_retirement": annuity_at_retirement,
        "discount_factor": discount_factor,
        "survival_to_retirement": survival_to_retirement,
        "dbo": dbo,
    }


def dbo_portfolio(valid_df, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return (one row of DBO workings per employee, total DBO)."""
    rows = [dbo_employee(row, valuation_date, a) for _, row in valid_df.iterrows()]
    results = pd.DataFrame(rows)
    return results, results["dbo"].sum()


def explain_employee(row, valuation_date=VALUATION_DATE, a=ASSUMPTIONS):
    """Return every step of one employee's DBO calculation as (label, value text) pairs."""
    r = dbo_employee(row, valuation_date, a)
    retirement_age = a["retirement_age"]
    # :g drops float noise, so 1 + 0.035 shows as 1.035 rather than 1.0350000000000001
    return [
        ("Employee", r["employee_id"]),
        ("Valuation date", f"{pd.Timestamp(valuation_date):%d.%m.%Y}"),
        ("Age", f"{r['age']:.2f} years"),
        ("Past service", f"{r['past_service']:.2f} years"),
        ("Years to retirement", f"{retirement_age} - {r['age']:.2f} = "
                                f"{r['years_to_retirement']:.2f}"),
        ("Yearly salary", f"€{row['monthly_salary_eur']:,.0f} x 14 = €{r['yearly_salary']:,.0f}"),
        ("Projected final salary", f"€{r['yearly_salary']:,.0f} x "
                                   f"{1 + a['salary_growth']:g}^{r['years_to_retirement']:.2f} = "
                                   f"€{r['projected_final_salary']:,.0f}"),
        ("Earned pension per year", f"{a['accrual_rate']:g} x {r['past_service']:.2f} x "
                                    f"€{r['projected_final_salary']:,.0f} = "
                                    f"€{r['earned_pension']:,.0f}"),
        (f"Annuity factor at {retirement_age}", f"{r['annuity_at_retirement']:.4f}"),
        ("Discount factor", f"1 / {1 + a['discount_rate']:g}^{r['years_to_retirement']:.2f} = "
                            f"{r['discount_factor']:.4f}"),
        (f"Survival to {retirement_age}", f"{r['survival_to_retirement']:.4f}"),
        ("DBO", f"€{r['earned_pension']:,.0f} x {r['annuity_at_retirement']:.4f} x "
                f"{r['discount_factor']:.4f} x {r['survival_to_retirement']:.4f} = "
                f"€{r['dbo']:,.0f}"),
    ]


def total_dbo(valid_df, valuation_date, a):
    """Return only the total DBO for one set of assumptions."""
    return dbo_portfolio(valid_df, valuation_date, a)[1]


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
        ("mortality x 0.9", "mortality_factor", 0.9),
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


def dbo_monte_carlo(valid_df, valuation_date=VALUATION_DATE, n_sims=500, seed=42, a=ASSUMPTIONS):
    """Return the 10th, 50th and 90th percentile of total DBO under uncertain assumptions."""
    rng = np.random.default_rng(seed)
    # Drawn independently. In reality both move with inflation, so they tend to rise and fall
    # together and partly cancel out; independent draws therefore overstate the spread
    discount_rates = rng.normal(a["discount_rate"], 0.005, size=n_sims)
    salary_growths = rng.normal(a["salary_growth"], 0.005, size=n_sims)

    totals = []
    for discount_rate, salary_growth in zip(discount_rates, salary_growths):
        changed = dict(a, discount_rate=discount_rate, salary_growth=salary_growth)
        totals.append(total_dbo(valid_df, valuation_date, changed))

    p10, p50, p90 = np.percentile(totals, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


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
