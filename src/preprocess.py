import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

QUESTIONS = [
    "When you face a tough decision, what do you do first?",
    "Your friend is in a crisis. How do you respond?",
    "You must choose between a safe path and a risky one with bigger rewards. You:",
]

ANSWER_CHOICES = {
    "Q1": [
        "List pros and cons on paper",           # 0 → Logical
        "Go with what feels right immediately",  # 1 → Emotional
        "Sleep on it and see how I feel later",  # 2 → Balanced
    ],
    "Q2": [
        "Give them a clear, step-by-step plan",  # 0 → Logical
        "Sit with them and share their pain",    # 1 → Emotional
        "Listen first, then offer options",      # 2 → Balanced
    ],
    "Q3": [
        "Analyze data and probabilities",        # 0 → Logical
        "Follow my gut — life is short",         # 1 → Emotional
        "Weigh both sides carefully before deciding", # 2 → Balanced
    ],
}

# Score map: answer index → (logical_pts, emotional_pts, balanced_pts)
SCORE_MAP = {
    0: (2, 0, 0),   # Logical answer
    1: (0, 2, 0),   # Emotional answer
    2: (0, 0, 2),   # Balanced answer
}


def generate_synthetic_data(n=1200, noise=0.12, random_state=42):
    """Generate synthetic training data with controlled noise for high accuracy."""
    rng = np.random.default_rng(random_state)
    records = []

    styles = ["Logical Thinker", "Emotional Thinker", "Balanced Thinker"]
    # Each style strongly prefers its own answer (index 0,1,2)
    prefs = {
        "Logical Thinker":   [0, 0, 0],   # always picks answer 0
        "Emotional Thinker": [1, 1, 1],   # always picks answer 1
        "Balanced Thinker":  [2, 2, 2],   # always picks answer 2
    }

    for _ in range(n):
        style = rng.choice(styles)
        base  = prefs[style]
        answers = []
        for b in base:
            if rng.random() < noise:
                # random wrong answer (adds realism)
                wrong = [x for x in [0,1,2] if x != b]
                answers.append(int(rng.choice(wrong)))
            else:
                answers.append(b)

        logical   = sum(1 for a in answers if a == 0)
        emotional = sum(1 for a in answers if a == 1)
        balanced  = sum(1 for a in answers if a == 2)
        records.append([logical, emotional, balanced, style])

    df = pd.DataFrame(records, columns=["Logical_score","Emotional_score","Balanced_score","Target"])
    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Target"])
    return df, le


def compute_scores_from_raw(answers: list):
    """answers: list of 3 ints (0,1,2)"""
    logical   = sum(1 for a in answers if a == 0)
    emotional = sum(1 for a in answers if a == 1)
    balanced  = sum(1 for a in answers if a == 2)
    return {"logical": logical, "emotional": emotional, "balanced": balanced}