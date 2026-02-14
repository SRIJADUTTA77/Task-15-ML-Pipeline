import os
import logging
from flask import Flask, request, render_template
from gcs_loader import load_model_from_gcs

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates")

# ENV VARS
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_BLOB = os.environ.get("MODEL_BLOB")

if not MODEL_BUCKET or not MODEL_BLOB:
    raise RuntimeError("MODEL_BUCKET or MODEL_BLOB not set")

logging.info(f"Loading model from gs://{MODEL_BUCKET}/{MODEL_BLOB}")

model = load_model_from_gcs(
    bucket_name=MODEL_BUCKET,
    blob_name=MODEL_BLOB
)

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
        probability=round(probability, 4),
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    logging.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
