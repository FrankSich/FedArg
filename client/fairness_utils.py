# client/fairness_util.py

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def analyze_diagnosis_fairness_by_sponsor(
    df: pd.DataFrame,
    sponsor_col="Sponsor",
    diagnosis_col="Diagnoses",
):
    """
    Analyze diagnosis distribution per sponsor.
    """

    sponsor_counts = df[sponsor_col].value_counts()
    sponsor_ratio = (sponsor_counts / sponsor_counts.sum()).round(3)

    summary = pd.DataFrame({
        "count": sponsor_counts,
        "ratio": sponsor_ratio
    })

    # Sponsor x Diagnoses
    matrix = (
        df.groupby([sponsor_col, diagnosis_col])
          .size()
          .unstack(fill_value=0)
    )

    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0).round(3)

    # Flag dominance
    summary["max_diagnosis_share"] = matrix_pct.max(axis=1)
    summary["biased"] = summary["max_diagnosis_share"] > 0.60

    return summary, matrix, matrix_pct


def smote_diagnoses_within_sponsor(
    df: pd.DataFrame,
    sponsor_col="Sponsor",
    diagnosis_col="Diagnoses",
):
    """
    Apply SMOTE to balance Diagnoses *within each Sponsor*.
    """

    result = []

    feature_cols = [
        c for c in df.columns
        if c not in [sponsor_col, diagnosis_col]
    ]

    df_enc = df.copy()
    encoders = {}

    # Encode categorical features
    for col in feature_cols:
        if df_enc[col].dtype == "object":
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    # Encode diagnoses
    diag_encoder = LabelEncoder()
    df_enc[diagnosis_col] = diag_encoder.fit_transform(
        df_enc[diagnosis_col].astype(str)
    )

    for sponsor, group in df_enc.groupby(sponsor_col):
        X = group[feature_cols]
        y = group[diagnosis_col]

        if y.nunique() < 2 or len(group) < 6:
            result.append(group)
            continue

        smote = SMOTE(
            random_state=42,
            k_neighbors=min(3, y.value_counts().min() - 1)
        )

        try:
            X_res, y_res = smote.fit_resample(X, y)
            tmp = pd.DataFrame(X_res, columns=feature_cols)
            tmp[diagnosis_col] = y_res
            tmp[sponsor_col] = sponsor
            result.append(tmp)
        except Exception:
            result.append(group)

    balanced = pd.concat(result)

    # Decode diagnoses back
    balanced[diagnosis_col] = diag_encoder.inverse_transform(
        balanced[diagnosis_col].astype(int)
    )

    return balanced.reset_index(drop=True)
