"""Synthetic employee data for a pension provision example. All employees are fictional."""

from pathlib import Path

import numpy as np
import pandas as pd

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
MIN_AGE = 22
MAX_AGE = 63
MIN_ENTRY_AGE = 20
MIN_SALARY_EUR = 2500
MAX_SALARY_EUR = 8000
DAYS_PER_YEAR = 365.25


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


if __name__ == "__main__":
    employees = generate_employees()
    messy = make_messy_employees(employees)

    print(f"Rows in truth: {len(employees)}")
    print(f"Rows in raw:   {len(messy)}")

    # Show rows with a visible problem: entry before birth, or a salary that isn't a number
    salary_is_number = messy["monthly_salary_eur"].apply(lambda value: isinstance(value, float))
    has_problem = (messy["entry_date"] < messy["birth_date"]) | ~salary_is_number
    has_problem = has_problem | messy["monthly_salary_eur"].isna()
    print("\n5 example messy rows:")
    print(messy[has_problem].head().to_string(index=False))
