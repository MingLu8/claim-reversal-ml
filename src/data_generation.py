import pandas as pd
import numpy as np

def generate_claims(n=10000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame()

    # Claim-level features
    df["ingredient_cost"] = np.random.gamma(2, 50, n)
    df["copay_amount"] = np.random.choice([0, 5, 10, 20, 40], n)
    df["days_supply"] = np.random.choice([30, 60, 90], n)
    df["refill_number"] = np.random.randint(0, 5, n)

    # Member-level
    df["member_tenure_days"] = np.random.randint(30, 2000, n)
    df["prior_reversal_count"] = np.random.poisson(0.5, n)

    df["pharmacy_type"] = np.random.choice(["chain", "independent"], n)
    df["drug_tier"] = np.random.choice(["generic", "preferred_brand", "non_preferred"], n)
    # Pharmacy-level
    df["pharmacy_historical_reversal_rate"] = np.random.uniform(0.01, 0.2, n)

    # Create probabilistic label (realistic logic)
    logit = (
        -3
        + 0.02 * df["copay_amount"]
        + 0.5 * df["pharmacy_historical_reversal_rate"]
        + 0.3 * df["prior_reversal_count"]
        - 0.0005 * df["member_tenure_days"]
    )

    prob = 1 / (1 + np.exp(-logit))

    df["reversed"] = np.random.binomial(1, prob)

    return df