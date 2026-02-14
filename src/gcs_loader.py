import tempfile
import joblib
from google.cloud import storage

def load_model_from_gcs(bucket_name: str, blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile() as temp:
        blob.download_to_filename(temp.name)
        model = joblib.load(temp.name)

    return model
