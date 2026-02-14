import os
from flask import Flask, request, render_template
from gcs_loader import load_model

app = Flask(__name__, template_folder="templates")

model = load_model()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    probability = max(model.predict_proba([features])[0])
    result = "Malignant" if prediction == 1 else "Benign"
    return render_template(
        "index.html",
        prediction=result,
        probability=probability
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
