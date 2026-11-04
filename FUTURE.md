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

- **Official mortality tables:** use AVÖ 2018-P instead of the illustrative Makeham formula.
- **Employee turnover:** model people leaving before retirement.
- **Service cost and interest cost:** the yearly change in DBO required for IAS 19 reporting.
- **Pensioners:** value people already receiving a pension, instead of only active employees.
- **Correlated assumptions:** discount rate and salary growth are linked through inflation; drawing them independently makes the Monte Carlo range too wide.
- **Austrian severance (Abfertigung alt) and jubilee payments:** other common obligations valued by Austrian actuaries.
