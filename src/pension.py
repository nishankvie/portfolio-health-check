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
    "max_age": 110,             # nobody is assumed to live beyond this age
}


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
    q = a["makeham_A"] + a["makeham_B"] * a["makeham_c"] ** age
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
    """Print every step of one employee's DBO calculation with labels."""
    r = dbo_employee(row, valuation_date, a)
    print(f"DBO calculation for {r['employee_id']} at {pd.Timestamp(valuation_date):%d.%m.%Y}")
    print(f"  Age:                     {r['age']:.2f} years")
    print(f"  Past service:            {r['past_service']:.2f} years")
    print(f"  Years to retirement:     {a['retirement_age']} - {r['age']:.2f} = "
          f"{r['years_to_retirement']:.2f}")
    print(f"  Yearly salary:           €{row['monthly_salary_eur']:,.0f} x 14 = "
          f"€{r['yearly_salary']:,.0f}")
    print(f"  Projected final salary:  €{r['yearly_salary']:,.0f} x "
          f"{1 + a['salary_growth']}^{r['years_to_retirement']:.2f} = "
          f"€{r['projected_final_salary']:,.0f}")
    print(f"  Earned pension per year: {a['accrual_rate']} x {r['past_service']:.2f} x "
          f"€{r['projected_final_salary']:,.0f} = €{r['earned_pension']:,.0f}")
    print(f"  Annuity factor at {a['retirement_age']}:    {r['annuity_at_retirement']:.4f}")
    print(f"  Discount factor:         1 / {1 + a['discount_rate']}^"
          f"{r['years_to_retirement']:.2f} = {r['discount_factor']:.4f}")
    print(f"  Survival to {a['retirement_age']}:          {r['survival_to_retirement']:.4f}")
    print(f"  DBO:                     €{r['earned_pension']:,.0f} x "
          f"{r['annuity_at_retirement']:.4f} x {r['discount_factor']:.4f} x "
          f"{r['survival_to_retirement']:.4f} = €{r['dbo']:,.0f}")


if __name__ == "__main__":
    employees = generate_employees()
    make_messy_employees(employees)
    raw = pd.read_csv(DATA_DIR / "employees_raw.csv")

    valid, log_df, query_df = validate_employees(raw)

    results, total_dbo = dbo_portfolio(valid, VALUATION_DATE, ASSUMPTIONS)
    print(f"Total DBO for {len(results)} valid employees: €{total_dbo:,.0f}")
    print(f"({len(query_df)} employees are waiting for client answers and are not included)")

    largest = results.sort_values("dbo", ascending=False).head(5)
    print("\n5 largest DBOs:")
    for _, result in largest.iterrows():
        print(f"  {result['employee_id']}  age {result['age']:.1f}  "
              f"service {result['past_service']:.1f} years  "
              f"salary €{result['yearly_salary']:,.0f}/year  DBO €{result['dbo']:,.0f}")

    # Pick the valid employee whose age is closest to 45
    closest_to_45 = (results["age"] - 45).abs().idxmin()
    employee_id = results.loc[closest_to_45, "employee_id"]
    example_row = valid[valid["employee_id"] == employee_id].iloc[0]
    print()
    explain_employee(example_row, VALUATION_DATE, ASSUMPTIONS)
