# src/gcs_loader.py
import os
import tempfile
from google.cloud import storage
import joblib
import logging

def load_model_from_gcs():
    bucket_name = os.environ.get("MODEL_BUCKET")
    blob_name = os.environ.get("MODEL_BLOB")

    if not bucket_name or not blob_name:
        raise RuntimeError("MODEL_BUCKET or MODEL_BLOB env vars not set")

    logging.info(f"Loading model from gs://{bucket_name}/{blob_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        model = joblib.load(tmp.name)

    logging.info("Model loaded successfully")
    return model
