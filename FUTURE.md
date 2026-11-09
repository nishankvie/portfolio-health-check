# Future improvements

Ideas collected while building the project, grouped by area. Each one says what it would add and why.

## Data generation

- **Funding rounds:** startups currently never raise new money, so cash only goes down. Adding rounds would make the data more realistic and enable recovery tracking.
- **Companies joining mid-window:** all companies are currently invested before the reporting window starts. New investments mid-window would have fewer months of data.
- **Companies stop spending at zero cash:** companies at €0 currently keep reporting costs, which is impossible in reality. They should shut down or raise a round.

## Data cleaning

- **ID validation against a master list:** check company_id against the fund's company list (CRM) and flag rows where the ID and name disagree, instead of assuming IDs are always correct.
- **Grid starts at each company's investment date:** needed once companies can join mid-window, so the cleaner doesn't invent months before the investment.
- **Cash identity check:** verify net burn using cash movement (cash this month = cash last month − net burn), to find out which field is wrong when the accounting check fails.

## Metrics and flags

- **Window limit from data length:** the maximum window is currently fixed at 12, which assumes 24 months of data. It should be derived from the data (months available ÷ 2).
- **Flag stability:** only change a flag after 2 months in a row, to avoid flickering and alert fatigue.
- **Recovery detection:** highlight companies that went red or amber and recovered (needs funding rounds first).
- **Last-round valuations:** value companies at their last funding round price, as real funds usually do, instead of illustrative revenue multiples.
- **Dilution:** reduce the fund's ownership when companies raise new rounds; stake values are currently overstated.
- **IRR:** add the internal rate of return, which accounts for when money was invested and returned.

## Monte Carlo

- **Fat tails:** use Student-t or bootstrapped historical changes instead of normal noise, to capture extreme months.
- **Jumps in all scenarios:** sudden events (losing a big client) are currently only in the downturn scenario.
- **Correlation between companies:** model shared shocks so a downturn hits all companies together, not only in the downturn scenario.

## Machine learning

- **More data:** only 4 companies ever failed, so results are fragile. Real portfolio history would be needed to trust the models.
- **More features:** burn multiple (needs handling of missing values), headcount growth, sector.
- **Threshold chosen by cost:** set the threshold based on the cost of a missed failure versus a false alarm.

## Pension module

- **Official mortality tables:** the page accepts an uploaded table. A real valuation needs a licensed table such as AVÖ 2018-P, which is not included.
- **Separate tables for men and women:** the model uses one table for everyone.
- **Employee turnover:** model people leaving before retirement.
- **Survivor and death benefits:** model pensions for survivors and benefits on death before retirement.
- **Pensioners:** value people already receiving a pension. Employees past the retirement age are currently treated as retiring on the valuation date.
- **Yield curve:** use a different discount rate for each payment date instead of one flat rate.
- **Actual payment dates:** the 14 payments are treated as evenly spaced, but the two extra payments fall around June and November.
- **Interest cost:** deduct the pensions paid during the year.
- **Correlated assumptions:** discount rate and salary growth are linked through inflation. Drawing them independently makes the Monte Carlo range too wide.
- **Austrian severance (Abfertigung alt) and jubilee payments:** other obligations often valued by Austrian actuaries.
