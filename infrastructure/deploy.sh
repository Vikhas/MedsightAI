#!/bin/bash
# ============================================================================
# MediSight — Cloud Run Deployment Script
# ============================================================================
# Usage: ./deploy.sh [PROJECT_ID] [REGION]
# ============================================================================

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="${2:-us-central1}"
SERVICE_NAME="medisight"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=========================================="
echo "  MediSight — Deploying to Cloud Run"
echo "=========================================="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Service:  ${SERVICE_NAME}"
echo "=========================================="

# 1. Ensure required APIs are enabled
echo ""
echo "→ Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

# 2. Build the container image
echo ""
echo "→ Building container image..."
cd "$(dirname "$0")/.."
gcloud builds submit \
    --tag "${IMAGE_NAME}" \
    --project="${PROJECT_ID}" \
    --quiet

# 3. Deploy to Cloud Run
echo ""
echo "→ Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --project="${PROJECT_ID}" \
    --allow-unauthenticated \
    --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY:-}" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 3600 \
    --max-instances 10 \
    --min-instances 0 \
    --port 8080 \
    --quiet

# 4. Get the service URL
echo ""
echo "→ Deployment complete!"
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format='value(status.url)')

echo ""
echo "=========================================="
echo "  ✅ MediSight is live!"
echo "  🌐 URL: ${SERVICE_URL}"
echo "=========================================="
