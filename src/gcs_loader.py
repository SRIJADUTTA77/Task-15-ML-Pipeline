import os
import tempfile
from google.cloud import storage
import joblib

_model = None  # cache

def load_model():
    global _model
    if _model is not None:
        return _model

    bucket_name = os.environ.get("MODEL_BUCKET")
    blob_name = os.environ.get("MODEL_BLOB")

    if not bucket_name or not blob_name:
        raise RuntimeError("MODEL_BUCKET or MODEL_BLOB not set")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        _model = joblib.load(tmp.name)

    return _model
