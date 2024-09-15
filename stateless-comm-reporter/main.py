from http import client
import json
import logging
import traceback
import os
import ast

# from LLMSession import LLMSession
import prompts

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values

import firebase_admin
from firebase_admin import firestore

import google.auth

from graph2nosql.graph2nosql.graph2nosql import NoSQLKnowledgeGraph
from graph2nosql.databases import firestore_kg
from graph2nosql.datamodel import data_model

from graphrag_lite.LLMSession import LLMSession
from graphrag_lite.GraphExtractor import GraphExtractor

app = FastAPI()


@observe()
def generate_response(c, kg: NoSQLKnowledgeGraph):

    print(f"Unwrapped Community Record: {c}")

    extractor = GraphExtractor(graph_db=kg)
    comm_data = extractor.async_generate_comm_report(c=c)
    return comm_data


@app.get("/helloworld")
def helloworld():
    return {"message": "Hello World"}


@app.post("/receive_community_request")
async def trigger_analysis(request: Request):
    """Receive and parse Pub/Sub messages."""
    secrets = dotenv_values(".env")

    os.environ["LANGFUSE_SECRET_KEY"] = str(
        secrets["LANGFUSE_SECRET_KEY"])
    os.environ["LANGFUSE_PUBLIC_KEY"] = str(
        secrets["LANGFUSE_PUBLIC_KEY"])
    os.environ["LANGFUSE_HOST"] = str(
        secrets["LANGFUSE_HOST"])

    fskg = firestore_kg.FirestoreKG(
        gcp_project_id=str(secrets["GCP_PROJECT_ID"]),
        gcp_credential_file=str(secrets["GCP_CREDENTIAL_FILE"]),
        firestore_db_id=str(secrets["FIRESTORE_DB_ID"]),
        node_collection_id=str(secrets["NODE_COLL_ID"]),
        edges_collection_id=str(secrets["EDGES_COLL_ID"]),
        community_collection_id=str(secrets["COMM_COLL_ID"])
    )

    try:
        payload = await request.body()
        message_dict = json.loads(payload.decode())
        print(
            f"Received envelope at /receive_analysis_request: {message_dict}")

        community_record = ast.literal_eval(message_dict["community_record"])
        print(community_record)

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        return JSONResponse(content={"message": f"Error parsing request: {e}"}, status_code=400)

    print(f"Received Pub/Sub message for Analysis: {message_dict}")

    try:
        comm_report = generate_response(
            c=community_record,
            kg=fskg
        )

        fskg.store_community(community=comm_report)

        print("comm report done")
        langfuse_context.flush()
        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during comm reporting: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)
