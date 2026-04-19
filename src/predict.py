import os, pickle, pandas as pd

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "labels.pkl")

INSIGHTS = {
    "Logical Thinker": {
        "icon": "🧠",
        "tagline": "You process the world through reason and structure.",
        "strengths": [
            "Exceptional at breaking complex problems into clear steps",
            "Calm and reliable under pressure",
            "Decisions are well-justified and defensible",
        ],
        "blind_spots": [
            "May undervalue emotions in people-centred situations",
            "Can overthink and delay when speed matters",
        ],
        "recommendation": "Practice making one small decision daily using only your gut — intuition is a skill you can build.",
        "color": "#5c6b4a",
    },
    "Emotional Thinker": {
        "icon": "🌸",
        "tagline": "You navigate life through empathy and feeling.",
        "strengths": [
            "Deeply attuned to people and relationships",
            "Courageous and fast in uncertain, human situations",
            "Creative — you see paths others miss",
        ],
        "blind_spots": [
            "Decisions can shift with your mood or energy",
            "Complex trade-offs may feel overwhelming",
        ],
        "recommendation": "Before big decisions, write down one data point or fact that supports each option — it grounds your instincts.",
        "color": "#8b5e52",
    },
    "Balanced Thinker": {
        "icon": "🌿",
        "tagline": "You move between logic and heart with ease.",
        "strengths": [
            "Trusted by both analytical and emotional people",
            "Handles ambiguity with steadiness",
            "Fair, nuanced, and rarely reactive",
        ],
        "blind_spots": [
            "Can appear indecisive in high-pressure moments",
            "May avoid taking a strong stand when it's needed",
        ],
        "recommendation": "When others need a leader, step forward — your balance is rare and people naturally look to you.",
        "color": "#4a7c6b",
    },
}


def _rule_based(logical, emotional, balanced):
    scores = {
        "Logical Thinker": logical,
        "Emotional Thinker": emotional,
        "Balanced Thinker": balanced,
    }
    return max(scores, key=scores.get)


def predict_style(logical: int, emotional: int, balanced: int) -> dict:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(LABELS_PATH, "rb") as f:
            le = pickle.load(f)

        X = pd.DataFrame(
            [[logical, emotional, balanced]],
            columns=["Logical_score", "Emotional_score", "Balanced_score"]
        )
        pred_code = model.predict(X)[0]
        style = le.inverse_transform([pred_code])[0]

        # Confidence probabilities
        proba = model.predict_proba(X)[0]
        classes = le.inverse_transform(model.classes_)
        confidence = {c: round(float(p) * 100) for c, p in zip(classes, proba)}

    except Exception:
        style = _rule_based(logical, emotional, balanced)
        confidence = {
            "Logical Thinker": logical * 33,
            "Emotional Thinker": emotional * 33,
            "Balanced Thinker": balanced * 34,
        }

    insight = INSIGHTS[style]

    return {
        "style": style,
        "icon": insight["icon"],
        "tagline": insight["tagline"],
        "strengths": insight["strengths"],
        "blind_spots": insight["blind_spots"],
        "recommendation": insight["recommendation"],
        "color": insight["color"],
        "confidence": confidence,
        "scores": {"logical": logical, "emotional": emotional, "balanced": balanced},
    }