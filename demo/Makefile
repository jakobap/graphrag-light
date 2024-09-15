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

# New command to copy and replace the graph2nosql repository
update-graph2nosql:
	rm -rf graph2nosql # Remove the existing directory if it exists
	cp -r ../graph2nosql/. graph2nosql # Copy the updated repository

# New command to copy and replace the graph2nosql repository
update-graphrag-lite:
	rm -rf graphrag_lite # Remove the existing directory if it exists
	cp -r ../graphrag_lite/. graphrag_lite # Copy the updated repository

build:
	gcloud builds submit . \
		--tag $$(gcloud config get-value artifacts/location)-docker.pkg.dev/${GCP_PROJECT_ID}/graph-rag-repo/graphrag-lite-demo-build:latest
		
deploy:
	gcloud run deploy graphrag-lite-demo-service \
		--image europe-west1-docker.pkg.dev/${GCP_PROJECT_ID}/graph-rag-repo/graphrag-lite-demo-build:latest \
		--region europe-west1 \
		--allow-unauthenticated

traffic:
	gcloud run services update-traffic graphrag-lite-demo-service --to-latest

all: config init update-graph2nosql update-graphrag-lite build deploy traffic