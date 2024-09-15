from http import client
import json
import logging
import traceback
import os
import ast

from LLMSession import LLMSession
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


app = FastAPI()


@observe()
def generate_response(c, kg: NoSQLKnowledgeGraph):

    print(f"Unwrapped Community Record: {c}")

    langfuse_context.update_current_trace(
        name="Async Community Report Generation",
        public=False
    )

    llm = LLMSession(system_message=prompts.COMMUNITY_REPORT_SYSTEM,
                     model_name="gemini-1.5-pro-001")

    comm_nodes = []
    comm_edges = []

    for n in c:
        node = kg.get_node(n)
        node_edges_to = [{"edge_source_entity": kg.get_edge(source_uid=node.node_uid,
                                                            target_uid=e),
                          "edge_target_entity": kg.get_edge(source_uid=node.node_uid,
                                                            target_uid=e),
                          "edge_description": kg.get_edge(source_uid=node.node_uid,
                                                          target_uid=e)} for e in node.edges_to]

        node_edges_from = [{"edge_source_entity": kg.get_edge(source_uid=e,
                                                              target_uid=node.node_uid),
                            "edge_target_entity": kg.get_edge(source_uid=e,
                                                              target_uid=node.node_uid),
                            "edge_description": kg.get_edge(source_uid=e,
                                                            target_uid=node.node_uid)} for e in node.edges_from]

        node_data = {"entity_id": node.node_title,
                     "entity_type": node.node_type,
                     "entity_description": node.node_description}

        comm_nodes.append(node_data)
        comm_edges.extend(node_edges_to)
        comm_edges.extend(node_edges_from)

        response_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "summary": {
                    "type": "string"
                },
                "rating": {
                    "type": "int"
                },
                "rating_explanation": {
                    "type": "string"
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string"
                            },
                            "explanation": {
                                "type": "string"
                            }
                        },
                        # Ensure both fields are present in each finding
                        "required": ["summary", "explanation"]
                    }
                }
            },
            # List required fields at the top level
            "required": ["title", "summary", "rating", "rating_explanation", "findings"]
        }

    comm_report = llm.generate(client_query_string=prompts.COMMUNITY_REPORT_QUERY.format(
        entities=comm_nodes,
        relationships=comm_edges,
        response_mime_type="application/json",
        response_schema=response_schema
    ))

    comm_report_dict = llm.parse_json_response(comm_report)

    if comm_report_dict == {}:
        comm_data = data_model.CommunityData(title=str(c),
                                             summary="",
                                             rating=0,
                                             rating_explanation="",
                                             findings=[{}],
                                             community_nodes=c)
    else:
        comm_data = data_model.CommunityData(title=comm_report_dict["title"],
                                             summary=comm_report_dict["summary"],
                                             rating=comm_report_dict["rating"],
                                             rating_explanation=comm_report_dict["rating_explanation"],
                                             findings=comm_report_dict["findings"],
                                             community_nodes=c)

    langfuse_context.flush()
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
        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during comm reporting: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)
