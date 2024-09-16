# graphrag-light

Welcome to graphrag-light. An easy to start, GCP native, graphrag implementation. 

![Graph Retrieval Augmented Generation (Graph RAG) Architecture](./graphrag-gcp.png)


## Installation Steps

1. Run in terminal to create a python venv:
```
python3 -m venv .venv-graphrag-lite 
```

2. graphrag-lite uses [graph2nosql](https://github.com/jakobap/graph2nosql), a simple python interface to store and manage knowledge graphs in your NoSQL DB of choice (*no* graphdb needed, you heard correctly!).

Add graph2nosql into your project folder for local installation. No pip distribution available (yet).
```
git clone https://github.com/jakobap/graph2nosql.git
```

3. Install graphrag-lite dependencies
```
pip install -r ./graphrag_lite/requirements.txt
```

4. This repo implements graphrag as paralelized architecture on GCP. Large parts of the indexing and query steps are processed by two stateless microservices. These are defined in `stateless-comm-reporter` and `stateless-context-processor`. Both directories contain Makefiles to automate the build and deployment. 

To deploy:
* Generate one or multiple service account key and place it/them in `stateless-comm-reporter` and `stateless-context-processor`.
* Setup an .env file with the following environment variables and place it in `stateless-comm-reporter` and `stateless-context-processor`:

```
GCP_CREDENTIAL_FILE=""
GCP_PROJECT_ID=""
GCP_PROJECT_NUMBER =""
GCP_REGION =""
GCP_MULTIREGION = ""
DOCUMENT_AI_PROCESSOR_ID =""
DOCUMENT_AI_PROCESSOR_VERSION = ""
LANGFUSE_SECRET_KEY = 
LANGFUSE_PUBLIC_KEY = 
LANGFUSE_HOST=
FIRESTORE_DB_ID=""
NODE_COLL_ID=""
COMM_COLL_ID=""
EDGES_COLL_ID=""
SCHEDULER_PUBSUB_ID=""
COMMUNITY_WL_PUBSUB=""
QUERY_FS_DB_ID=""
QUERY_FS_INT__RESPONSE_COLL=""
RAW_PDFS_BUCKET_NAME=""
```
* Run 
```
make all
```