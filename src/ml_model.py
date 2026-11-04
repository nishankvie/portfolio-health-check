"""Build a dataset for predicting which companies will run out of cash (synthetic data)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# "src." because this file is run as a module from the project folder (python3 -m src.ml_model),
# the same way app.py will import it
from src.simulate import SCENARIOS, simulate_company

# Resolve data/ from this file's location so the script works from any folder
DATA_DIR = Path(__file__).parent.parent / "data"

FEATURES = ["runway_capped", "mom_growth", "burn_trend_pct", "is_profitable"]

# Beyond 3 years the exact runway hardly changes the risk, and the model can't use infinity
RUNWAY_CAP_MONTHS = 36


def runs_out_within(company_cash, as_of_month, horizon):
    """Return 1 if cash is 0 in any month after as_of_month up to as_of_month + horizon, else 0."""
    end_month = as_of_month + pd.DateOffset(months=horizon)
    in_window = (company_cash["month"] > as_of_month) & (company_cash["month"] <= end_month)
    ran_out = (company_cash.loc[in_window, "cash_eur"] == 0).any()
    return int(ran_out)


def add_features(rolling_rows):
    """Return the rows with the model features added (shared by training and scoring)."""
    df = rolling_rows[["company_id", "as_of_month", "latest_cash", "runway_months",
                       "mom_growth", "burn_trend_pct", "avg_burn"]].copy()
    df["runway_capped"] = df["runway_months"].clip(upper=RUNWAY_CAP_MONTHS)
    df["is_profitable"] = (df["avg_burn"] <= 0).astype(int)
    return df


def build_dataset(rolling_df, clean_df, horizon=6):
    """Return one row per company and month with features and a 'runs out soon' label."""
    # Features: all come from rolling_df, which only used data up to as_of_month
    df = add_features(rolling_df)

    # Label: the ONLY column allowed to look ahead, because it is the outcome we want to
    # predict. If a feature looked ahead, the model would learn from information it
    # won't have when used for real
    labels = []
    for _, row in df.iterrows():
        company_cash = clean_df[clean_df["company_id"] == row["company_id"]]
        labels.append(runs_out_within(company_cash, row["as_of_month"], horizon))
    df["label"] = labels

    # Already out of cash: there's nothing left to predict, and keeping these rows would
    # teach the model the trivial rule "cash is 0 now, so cash is 0 later"
    already_out = df["latest_cash"] == 0

    # Label unknown: the full horizon isn't in the data yet. Counting these as 0 would
    # wrongly say "survived" for companies that might run out next month
    last_month = clean_df["month"].max()
    window_incomplete = df["as_of_month"] + pd.DateOffset(months=horizon) > last_month

    df = df[~already_out & ~window_incomplete]

    # Missing features: logistic regression can't handle NaN, and guessing values would
    # make up information
    df = df.dropna(subset=FEATURES)

    return df[["company_id", "as_of_month"] + FEATURES + ["label"]].reset_index(drop=True)


def add_rule_flags(dataset, rolling_df):
    """Add the rule-based flag and two 0/1 detector columns: red, and red or amber."""
    flags = rolling_df[["company_id", "as_of_month", "flag"]]
    df = dataset.merge(flags, on=["company_id", "as_of_month"], how="left")
    df["rule_red"] = (df["flag"] == "red").astype(int)
    df["rule_red_amber"] = df["flag"].isin(["red", "amber"]).astype(int)
    return df


def make_model():
    """Return a fresh, untrained pipeline: scale the features, then logistic regression."""
    # class_weight="balanced": only about 1 row in 9 is a failure, so without it the model
    # can score well by mostly predicting "survives". Balancing makes each missed failure
    # count as much as all the survivors together, so the model has to find failures
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def make_forest():
    """Return a fresh, untrained random forest."""
    # max_depth=3: each tree may ask at most 3 questions. With only 4 failing companies, deeper
    # trees would memorise those exact companies (overfitting) instead of general patterns.
    # Trees split on thresholds, so unlike logistic regression no scaling is needed
    return RandomForestClassifier(
        n_estimators=200, max_depth=3, class_weight="balanced", random_state=42
    )


def cross_validated_proba(model, dataset):
    """Return out-of-sample probabilities from leave-one-company-out cross-validation."""
    X = dataset[FEATURES]
    y = dataset["label"]
    groups = dataset["company_id"]

    # Leave one whole company out per fold: the same company's months are near-copies of
    # each other, so if some of its months were in training the model would recognise the
    # company instead of learning general warning signs. This way every prediction is for
    # a company the model has never seen
    probabilities = cross_val_predict(
        model, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba"
    )
    # predict_proba returns [P(label 0), P(label 1)] per row; we want P(runs out)
    return probabilities[:, 1]


def train_logistic_cv(dataset):
    """Return an out-of-sample logistic regression probability for every row."""
    return cross_validated_proba(make_model(), dataset)


def train_forest_cv(dataset):
    """Return an out-of-sample random forest probability for every row."""
    return cross_validated_proba(make_forest(), dataset)


def evaluate(y_true, y_pred, name):
    """Return precision, recall and confusion-matrix counts for one detector."""
    # labels=[0, 1] fixes the order, so ravel() always gives tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # A detector that never says "yes" has no precision, which is undefined rather than 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "detector": name,
        "precision": precision,
        "recall": recall,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def fit_final_logistic(dataset):
    """Fit the pipeline on all rows and return one coefficient per (scaled) feature."""
    model = make_model()
    model.fit(dataset[FEATURES], dataset["label"])
    coefficients = model.named_steps["model"].coef_[0]
    return pd.Series(coefficients, index=FEATURES)


def forest_importance(dataset):
    """Fit the forest on all rows and return how much each feature was used."""
    forest = make_forest()
    forest.fit(dataset[FEATURES], dataset["label"])
    # Importance = how much a feature's splits reduced the mix of classes, averaged over all
    # trees. It shows what the model relied on, NOT what causes companies to run out
    return pd.Series(forest.feature_importances_, index=FEATURES)


def threshold_table(y_true, proba, thresholds=(0.3, 0.5, 0.7)):
    """Return precision, recall, TP and FP for each probability threshold."""
    rows = []
    for threshold in thresholds:
        predictions = (proba >= threshold).astype(int)
        result = evaluate(y_true, predictions, name="")
        rows.append({
            "threshold": threshold,
            "precision": result["precision"],
            "recall": result["recall"],
            "TP": result["TP"],
            "FP": result["FP"],
        })
    return pd.DataFrame(rows)


def risk_today(rolling_df, clean_df, horizon=6):
    """Return each company's ML risk and Monte Carlo probability of running out in 6 months."""
    dataset = build_dataset(rolling_df, clean_df, horizon)
    model = make_model()
    model.fit(dataset[FEATURES], dataset["label"])

    # Latest month only, companies that still have cash: the ones we can still act on.
    # These rows have no label yet (their future hasn't happened), which is the point
    latest_month = rolling_df["as_of_month"].max()
    today = add_features(rolling_df[rolling_df["as_of_month"] == latest_month])
    today = today[today["latest_cash"] > 0].dropna(subset=FEATURES)

    result = today[["company_id"]].copy()
    result["ml_risk"] = model.predict_proba(today[FEATURES])[:, 1]

    # Same seed per company as simulate_portfolio (42 + position in the sorted company list)
    all_companies = sorted(clean_df["company_id"].unique())
    mc_probs = []
    for company_id in result["company_id"]:
        company = clean_df[clean_df["company_id"] == company_id]
        seed = 42 + all_companies.index(company_id)
        simulation = simulate_company(company, horizon=6, seed=seed, scenario=SCENARIOS["base"])
        mc_probs.append(simulation["prob_out_of_cash"])
    result["mc_prob_6m"] = mc_probs

    return result.reset_index(drop=True)


if __name__ == "__main__":
    rolling = pd.read_csv(
        DATA_DIR / "rolling_metrics.csv", parse_dates=["as_of_month"], date_format="%Y-%m-%d"
    )
    clean = pd.read_csv(
        DATA_DIR / "clean_portfolio.csv", parse_dates=["month"], date_format="%Y-%m-%d"
    )

    dataset = build_dataset(rolling, clean)
    dataset.to_csv(DATA_DIR / "ml_dataset.csv", index=False)
    dataset = add_rule_flags(dataset, rolling)

    logistic_proba = train_logistic_cv(dataset)
    forest_proba = train_forest_cv(dataset)
    y = dataset["label"]

    results = pd.DataFrame([
        evaluate(y, dataset["rule_red"], "rule_red"),
        evaluate(y, dataset["rule_red_amber"], "rule_red_amber"),
        evaluate(y, (logistic_proba >= 0.5).astype(int), "logistic_regression"),
        evaluate(y, (forest_proba >= 0.5).astype(int), "random_forest"),
    ])
    print(f"Detectors on {len(dataset)} rows ({y.sum()} that ran out within 6 months):")
    print(results.round({"precision": 2, "recall": 2}).to_string(index=False))

    coefficients = fit_final_logistic(dataset)
    by_size = coefficients.abs().sort_values(ascending=False).index
    print("\nLogistic regression coefficients (scaled features, all rows):")
    print(coefficients[by_size].round(3).to_string())

    importances = forest_importance(dataset).sort_values(ascending=False)
    print("\nRandom forest feature importances (all rows):")
    print(importances.round(3).to_string())

    for name, proba in [("Logistic regression", logistic_proba), ("Random forest", forest_proba)]:
        print(f"\nThresholds, {name}:")
        print(threshold_table(y, proba).round({"precision": 2, "recall": 2}).to_string(index=False))

    today = risk_today(rolling, clean)
    today = today.sort_values("ml_risk", ascending=False)
    today["ml_risk"] = today["ml_risk"].map(lambda p: f"{p:.1%}")
    today["mc_prob_6m"] = today["mc_prob_6m"].map(lambda p: f"{p:.1%}")
    latest_month = rolling["as_of_month"].max()
    print(f"\nRisk today ({latest_month:%Y-%m}), companies with cash:")
    print(today.to_string(index=False))
