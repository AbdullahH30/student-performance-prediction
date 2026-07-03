"""
Generate a realistic synthetic Student Performance Dataset.
This script produces a CSV file that mirrors real-world student data
based on published educational research distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000


def generate_dataset(n: int = N) -> pd.DataFrame:
    """
    Generate a synthetic student performance dataset.

    Args:
        n: Number of student records to generate.

    Returns:
        DataFrame with all student features and performance label.
    """
    # ── Demographics ────────────────────────────────────────────────────────
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    age = np.random.randint(15, 19, size=n)

    school_type = np.random.choice(
        ["Public", "Private"], size=n, p=[0.65, 0.35]
    )
    distance_from_home = np.random.choice(
        ["Near", "Moderate", "Far"], size=n, p=[0.45, 0.35, 0.20]
    )

    # ── Socio-economic / Family ──────────────────────────────────────────────
    parental_education = np.random.choice(
        ["None", "High School", "College", "Postgraduate"],
        size=n,
        p=[0.10, 0.35, 0.40, 0.15],
    )
    family_support = np.random.choice(
        ["Low", "Medium", "High"], size=n, p=[0.20, 0.45, 0.35]
    )
    internet_access = np.random.choice(["Yes", "No"], size=n, p=[0.75, 0.25])

    # ── Academic Behaviour ───────────────────────────────────────────────────
    study_hours = np.clip(np.random.normal(5, 2, n), 0, 12).round(1)
    attendance = np.clip(np.random.normal(80, 12, n), 30, 100).round(1)
    absences = np.clip(np.random.normal(6, 4, n), 0, 30).round(0).astype(int)
    previous_grades = np.clip(np.random.normal(68, 15, n), 30, 100).round(1)

    # ── Lifestyle ────────────────────────────────────────────────────────────
    sleep_hours = np.clip(np.random.normal(7, 1.2, n), 4, 10).round(1)
    extracurricular = np.random.choice(["Yes", "No"], size=n, p=[0.55, 0.45])
    tutoring = np.random.choice(["Yes", "No"], size=n, p=[0.40, 0.60])

    motivation_level = np.random.choice(
        ["Low", "Medium", "High"], size=n, p=[0.25, 0.45, 0.30]
    )

    # ── Composite score (drives target, reflects real research) ─────────────
    score = (
        0.30 * (study_hours / 12)
        + 0.25 * (attendance / 100)
        + 0.20 * (previous_grades / 100)
        - 0.10 * (absences / 30)
        + 0.05 * (sleep_hours / 10)
        + 0.05 * (parental_education == "Postgraduate").astype(float)
        + 0.03 * (family_support == "High").astype(float)
        + 0.02 * (internet_access == "Yes").astype(float)
    )
    noise = np.random.normal(0, 0.05, n)
    score = np.clip(score + noise, 0, 1)

    # Motivation boost
    score[motivation_level == "High"] += 0.08
    score[motivation_level == "Low"] -= 0.08
    score = np.clip(score, 0, 1)

    # ── Target variable ──────────────────────────────────────────────────────
    performance = pd.cut(
        score,
        bins=[0, 0.40, 0.68, 1.01],
        labels=["Low", "Medium", "High"],
    )

    df = pd.DataFrame(
        {
            "gender": gender,
            "age": age,
            "school_type": school_type,
            "distance_from_home": distance_from_home,
            "parental_education": parental_education,
            "family_support": family_support,
            "internet_access": internet_access,
            "study_hours": study_hours,
            "attendance": attendance,
            "absences": absences,
            "previous_grades": previous_grades,
            "sleep_hours": sleep_hours,
            "extracurricular": extracurricular,
            "tutoring": tutoring,
            "motivation_level": motivation_level,
            "performance": performance,
        }
    )

    # Introduce ~3 % missing values in selected columns (realistic noise)
    for col in ["sleep_hours", "family_support", "parental_education"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "data" / "student_performance.csv"
    output_path.parent.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"✅  Dataset saved → {output_path}  ({len(df)} rows, {df.shape[1]} cols)")
    print(df["performance"].value_counts())
