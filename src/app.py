import os
from flask import Flask, request, render_template
from gcs_loader import load_model_from_gcs

app = Flask(__name__, template_folder="templates")

model = None

@app.before_first_request
def load_model():
    global model
    bucket = os.environ.get("MODEL_BUCKET")
    blob = os.environ.get("MODEL_BLOB")

    if not bucket or not blob:
        raise RuntimeError("MODEL_BUCKET or MODEL_BLOB not set")

    model = load_model_from_gcs(bucket, blob)


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
