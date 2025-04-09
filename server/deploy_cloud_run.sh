#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables (replace with your actual values)
PROJECT_ID="$(gcloud config get-value project)" # Or hardcode your project ID
PROJECT_NUMBER="$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')"
SERVICE_NAME="gofanco-mcp-server"       # Name for your Cloud Run service
REGION="us-central1"                  # Google Cloud region
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" # Docker image name

# Required environment variables for the Cloud Run service
# Option 1: Set directly in the command (less secure)
# ES_URL="https://your-es-proxy-url"
# BEARER_TOKEN="your-bearer-token"
# CLOUD_ID="your-cloud-id"

# Option 2: Recommended - Use Secret Manager for sensitive values
# Create secrets in Google Secret Manager first (e.g., es-bearer-token, es-cloud-id)
ES_URL_SECRET="projects/${PROJECT_NUMBER}/secrets/ES_URL:latest" # Name of your ES_URL secret
BEARER_TOKEN_SECRET="projects/${PROJECT_NUMBER}/secrets/ES_BEARER_TOKEN:latest" # Name of your BEARER_TOKEN secret
CLOUD_ID_SECRET="projects/${PROJECT_NUMBER}/secrets/ES_CLOUD_ID:latest"       # Name of your CLOUD_ID secret

# Enable required Google Cloud services
echo "Enabling necessary Google Cloud services..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Build the Docker image using Cloud Build
echo "Building Docker image with Cloud Build..."
gcloud builds submit --tag ${IMAGE_NAME}

echo "Image built successfully: ${IMAGE_NAME}"

# Deploy to Cloud Run
echo "Deploying service ${SERVICE_NAME} to Cloud Run in region ${REGION}..."

# Deployment command using Secret Manager for environment variables
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets=ES_URL=${ES_URL_SECRET},ES_BEARER_TOKEN=${BEARER_TOKEN_SECRET},ES_CLOUD_ID=${CLOUD_ID_SECRET} \
  --quiet

# # Deployment command setting environment variables directly (less secure)
# gcloud run deploy ${SERVICE_NAME} \
#   --image ${IMAGE_NAME} \
#   --region ${REGION} \
#   --platform managed \
#   --allow-unauthenticated \
#   --set-env-vars=ES_URL=${ES_URL},ES_BEARER_TOKEN=${BEARER_TOKEN},ES_CLOUD_ID=${CLOUD_ID} \
#   --quiet

echo "Deployment complete."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Service URL: ${SERVICE_URL}"

echo "You can now test your service:"
echo "curl ${SERVICE_URL}/health"
echo "curl -X POST ${SERVICE_URL}/recommend -H \"Content-Type: application/json\" -d '{\"query\": \"4K HDMI splitter\"}'" 
