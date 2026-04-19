import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request
from preprocess import QUESTIONS, ANSWER_CHOICES, compute_scores_from_raw
from predict import predict_style

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        questions=QUESTIONS,
        answer_choices=ANSWER_CHOICES,
        enumerate=enumerate,
        total=len(QUESTIONS),
    )


@app.route("/predict", methods=["POST"])
def predict():
    answers = [int(request.form.get(f"q{i}", 0)) for i in range(1, 4)]
    scores  = compute_scores_from_raw(answers)
    result  = predict_style(scores["logical"], scores["emotional"], scores["balanced"])

    result["logical_pct"]   = round(scores["logical"]   / 3 * 100)
    result["emotional_pct"] = round(scores["emotional"]  / 3 * 100)
    result["balanced_pct"]  = round(scores["balanced"]   / 3 * 100)

    return render_template("result.html", r=result)


if __name__ == "__main__":
    app.run(debug=True)