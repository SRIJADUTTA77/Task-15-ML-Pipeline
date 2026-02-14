import joblib
import tempfile
from google.cloud import storage


def load_model_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile() as temp:
        blob.download_to_filename(temp.name)
        model = joblib.load(temp.name)

    return model
