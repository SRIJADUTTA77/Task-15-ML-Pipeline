import os
from google.cloud import storage
import joblib

def load_model_from_gcs(
    bucket_name: str,
    blob_name: str,
    local_path: str = "/tmp/model.pkl"
):
    """
    Securely download and load a model from GCS.
    Uses Cloud Run service account automatically.
    """

    if not os.path.exists(local_path):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.download_to_filename(local_path)

    return joblib.load(local_path)
