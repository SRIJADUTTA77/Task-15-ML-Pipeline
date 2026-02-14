# src/app.py
import os
import logging
from flask import Flask, request, render_template
from gcs_loader import load_model_from_gcs

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates")

model = None  # DO NOT load at import time

@app.before_first_request
def init_model():
    global model
    if model is None:
        logging.info("Initializing ML model...")
        model = load_model_from_gcs()

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
    logging.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
