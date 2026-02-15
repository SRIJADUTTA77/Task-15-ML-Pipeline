# Breast Cancer ML Pipeline – CI/CD with GitHub Actions & Google Cloud Run

## Project Overview

This project implements an end-to-end **Machine Learning inference pipeline** for Breast Cancer prediction, deployed on **Google Cloud Run** with **CI/CD auto-deployment via GitHub Actions**. The design emphasizes **low cost (free-tier friendly)**, secure model loading from **Google Cloud Storage (GCS)**, and production-ready DevOps practices.

---

## Architecture Summary

* **Model**: Scikit-learn pipeline (`.pkl`) stored in GCS
* **API**: Flask application
* **Containerization**: Docker
* **Deployment Target**: Google Cloud Run (fully managed)
* **CI/CD**: GitHub Actions → Cloud Run
* **Authentication**: Workload Identity Federation (no long-lived keys)

---

## Stage 1 – Task-15 (Completed)

**Objective:** Build and containerize an ML inference service

### What Was Implemented

* Flask-based inference API (`src/app.py`)
* ML pipeline creation and serialization
* Dockerfile with Cloud Run–compatible configuration
* Runtime port binding using `$PORT`

### Key Files

* `src/app.py`
* `src/ml_pipeline.py`
* `requirements.txt`
* `Dockerfile`

### Validation

* Local Docker run successful
* API responds correctly on `/` and `/predict`

**Status:** ✅ Completed

---

## Stage 2 – Deployment on Google Cloud (Low / Near-Free Cost)

**Objective:** Deploy securely with minimal cost

### Google Cloud Services Used

| Service           | Usage             | Cost Strategy                 |
| ----------------- | ----------------- | ----------------------------- |
| Cloud Run         | Inference hosting | Free tier (2M requests/month) |
| Cloud Storage     | Model storage     | Standard, single small object |
| Artifact Registry | Container images  | Minimal usage                 |
| Cloud Build       | Image build       | Free tier                     |

### Secure Model Loading (Runtime)

* Model stored in GCS
* Loaded at container startup using Application Default Credentials
* No model committed to GitHub

### Environment Variables

```bash
MODEL_BUCKET=run-sources-breast-cancer-ml-app-2026-us-central1
MODEL_BLOB=models/breast_cancer_pipeline.pkl
```

### Cost Notes

* Idle Cloud Run services incur **zero compute cost**
* Storage cost for model (<10 KB) is negligible

**Status:** ✅ Completed

---

## Stage 3 – CI/CD Auto Deployment via GitHub Actions

**Objective:** Fully automated build & deploy on every push to `main`

### CI/CD Flow

```text
Git Push → GitHub Actions → Cloud Build → Cloud Run Deploy
```

### Workflow Highlights

* Trigger: `push` to `main`
* Auth: Workload Identity Federation (OIDC)
* No service account keys stored in GitHub
* Automatic container build and deploy

### Workflow File

`.github/workflows/deploy.yml`

### Verified Capabilities

* Auto redeploy on code changes
* Health-check aware deployment
* Correct PORT binding
* GCS model access verified

**Status:** ✅ Completed & Verified

---

## Final Implementation (Current State)

At this stage, the system is **production-grade for a portfolio / academic / demo ML deployment**.

### What You Have Achieved

* ✔ End-to-end ML inference service
* ✔ Secure, scalable, serverless deployment
* ✔ Zero-maintenance CI/CD pipeline
* ✔ Near-zero operational cost

---

## Recommended Next Enhancements (Optional)

**Expert Suggestions**

1. **Model Versioning**

   * Store models as `models/v1/`, `v2/` in GCS

2. **Traffic Splitting**

   * Canary deploy new revisions in Cloud Run

3. **Monitoring**

   * Enable Cloud Run metrics & alerts

4. **Authentication**

   * Protect `/predict` with IAM or API Gateway

5. **Frontend UI**

   * Add lightweight Streamlit or HTML form

---

## Service URL (Live)

```text
https://breast-cancer-ml-service-409419731461.us-central1.run.app
```

---

## Conclusion

This implementation demonstrates **industry-aligned MLOps and DevOps maturity**:

* Clean separation of code, model, and infra
* Secure cloud-native practices
* CI/CD automation comparable to real-world ML platforms


