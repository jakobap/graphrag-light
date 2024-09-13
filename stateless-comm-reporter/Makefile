ifneq (,$(wildcard ./.env))
    include .env
    export
endif

config:
	gcloud config set project ${GCP_PROJECT_ID}
	gcloud config set ai/region ${GCP_REGION}
	gcloud config set run/region ${GCP_REGION}
	gcloud config set artifacts/location ${GCP_REGION}

init:
	gcloud services enable {firestore,storage,aiplatform,compute,run,cloudbuild,artifactregistry}.googleapis.com

build:
	gcloud builds submit . \
		--tag $$(gcloud config get-value artifacts/location)-docker.pkg.dev/${GCP_PROJECT_ID}/graph-rag-repo/stateless-comm-reporter-image:latest
		
deploy:
	gcloud run deploy stateless-comm-reporter-service \
		--image europe-west1-docker.pkg.dev/${GCP_PROJECT_ID}/graph-rag-repo/stateless-comm-reporter-image:latest \
		--region europe-west1 \
		--allow-unauthenticated

traffic:
	gcloud run services update-traffic stateless-comm-reporter-service --to-latest

all: config init build deploy traffic