from http import client
import json
import logging
import traceback
import os

from LLMSession import LLMSession
import prompts

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values

import firebase_admin
from firebase_admin import firestore

import google.auth

try:
    from graph2nosql.graph2nosql.graph2nosql import NoSQLKnowledgeGraph   
    from graph2nosql.databases import firestore_kg
except:
    from ..graph2nosql.graph2nosql.graph2nosql import NoSQLKnowledgeGraph   
    from ..graph2nosql.databases import firestore_kg
    from ..graph2nosql.datamodel import data_model


app = FastAPI()


@observe()
def generate_response(c, kg: NoSQLKnowledgeGraph):

    try:
        langfuse_context.update_current_trace(
            name="Community Intermediate Query Gen",
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

                kg.store_community(community=comm_data)

        langfuse_context.flush()
        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during report generation for community: {c} with error: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)

def store_in_fs(response: str, user_query: str, community_report: dict) -> None:
    """
    Stores the LLM response in Firestore.

    Args:
        response (str): The JSON formatted LLM response.
        user_query (str): The original user query.
        community_report (dict): The community report used for the response.
    """
    secrets = dotenv_values(".env")
    gcp_credentials, project_id = google.auth.load_credentials_from_file(
        str(secrets["GCP_CREDENTIAL_FILE"]))

    db = firestore.Client(project=project_id,  # type: ignore
                          credentials=gcp_credentials,
                          database=str(secrets["QUERY_FS_DB_ID"]))

    # Extract community title for structuring data
    community_title = community_report.get("title", "Unknown Community")

    # Parse the JSON response
    try:
        # response = response.replace("'", '"')
        response_dict = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"Error {e} while decoding JSON response: {response}")
        return  # Exit early on error

    # Prepare data for Firestore
    refreshed_data = {
        "community": community_title,
        "response": response_dict.get("response", ""),
        "score": response_dict.get("score", 0)
    }

    # Get a reference to the document (using user_query as key)
    doc_ref = db.collection(secrets["QUERY_FS_INT__RESPONSE_COLL"]).document(
        user_query)  # Use collection ID from .env

    # Get the existing data if any
    doc_snapshot = doc_ref.get()
    existing_data = doc_snapshot.to_dict() if doc_snapshot.exists else {}

    existing_data[community_title] = refreshed_data

    # Update the document with the new list
    doc_ref.set(existing_data, merge=True)

    logging.info(f"Stored response for query '{user_query}' and community '{
                 community_title}' in Firestore.")
    print("saving in fs done")


@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}


@app.post("/receive_community_request")
async def trigger_analysis(request: Request):
    """Receive and parse Pub/Sub messages."""
    secrets = dotenv_values(".env")

    os.environ["LANGFUSE_SECRET_KEY"] = str(
        secrets["LANGFUSE_SECRET_KEY"])
    os.environ["LANGFUSE_PUBLIC_KEY"] = str(
        secrets["LANGFUSE_PUBLIC_KEY"])
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

    fskg = firestore_kg.FirestoreKG(
        gcp_project_id=str(secrets["project_id"]),
        gcp_credential_file=str(secrets["firestore_credential_file"]),
        firestore_db_id=str(secrets["database_id"]),
        node_collection_id=str(secrets["node_coll_id"]),
        edges_collection_id=str(secrets["edges_coll_id"]),
        community_collection_id=str(secrets["community_coll_id"])
    )

    try:
        payload = await request.body()
        message_dict = json.loads(payload.decode())
        logging.debug(
            f"Received envelope at /receive_analysis_request: {message_dict}")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        return JSONResponse(content={"message": f"Error parsing request: {e}"}, status_code=400)

    print(f"Received Pub/Sub message for Analysis: {message_dict}")

    try:
        response_json = generate_response(
            c=message_dict["community_record"],
            kg=fskg
        )

        store_in_fs(response=str(response_json),
                    user_query=message_dict["user_query"], community_report=message_dict["community_report"])

        print("analysis done")
        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during file analysis: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)


if __name__ == "__main__":

#     sample_query = "Who are some of the most notable individuals of us american politics in the 21century"
#     sample_msg = {'community_report': {'title': 'Joe Biden, Kamala Harris, and the Democratic Party', 'community_nodes': ['KAMALA HARRIS', 'DEMOCRATIC PARTY', 'JOE BIDEN'], 'summary': "This community centers around the political figures of Joe Biden and Kamala Harris, and their relationship to the Democratic Party in the context of the 2024 US Presidential election. Joe Biden, the previous president, withdrew from the race, leading Kamala Harris to become the party's nominee.", 'document_id': None, 'community_uid': None, 'community_embedding': [], 'rating': 8.5, 'rating_explanation': 'The impact severity is high due to the significant influence these individuals and the Democratic Party hold in the US political landscape, particularly during a presidential election year.', 'findings': [{'explanation': "Joe Biden, the 46th US President, withdrew from the 2024 presidential race due to age, health, and political reasons. This decision significantly impacted the Democratic Party's trajectory for the election. [Data: Entities ('JOE BIDEN'), ('DEMOCRATIC PARTY')]", 'summary': "Joe Biden's Withdrawal and its Impact"}, {'explanation': "Following Biden's withdrawal, Kamala Harris, the current US Vice President, became the Democratic Party's nominee for president. This shift in candidacy places Harris in a prominent position within the party and the election. [Data: Entities ('KAMALA HARRIS'), ('DEMOCRATIC PARTY')]", 'summary': "Kamala Harris's Nomination"}, {'explanation': "The Democratic Party, as one of the two major political parties in the US, plays a crucial role in the presidential election. The party's dynamics and decisions significantly impact the country's political landscape. [Data: Entities ('DEMOCRATIC PARTY')]", 'summary': "The Democratic Party's Role"}]}, 'user_query': 'Who are some of the most notable individuals of us american politics in the 21century '}

#     response = generate_response(client_query=sample_query, community_report=sample_msg["community_report"])
#     store_in_fs(response=response, user_query=sample_query, community_report=sample_msg["community_report"])

#     sample_msg ={'community_report': {'title': 'Count Bernadotte and the UN', 'community_nodes': ['COUNT BERNADOTTE', 'UN'], 'summary': "This community centers around the historical figure of Count Bernadotte, a Swedish UN envoy to Palestine who was murdered in 1948. The relationship highlights Bernadotte's diplomatic role within the UN's involvement in the Palestine situation during that era.", 'document_id': None, 'community_uid': None, 'community_embedding': [], 'rating': 2.0, 'rating_explanation': 'The impact severity rating is low because the events surrounding this community are historical and do not pose an ongoing threat.', 'findings': [{'explanation': "Count Bernadotte played a significant historical role as a Swedish UN envoy to Palestine. His assassination in 1948 underscores the volatile political climate of the region at the time.  [Data: Entities ('COUNT BERNADOTTE'), Relationships ('COUNT BERNADOTTE_to_UN')]", 'summary': "Count Bernadotte's Diplomatic Role"}, {'explanation': "The UN's involvement in Palestine during this period, as evidenced by Bernadotte's envoy role, highlights the organization's early engagement in seeking resolutions to the Arab-Israeli conflict. [Data: Entities ('UN'), Relationships ('COUNT BERNADOTTE_to_UN')]", 'summary': 'UN Involvement in Palestine'}]}, 'user_query': 'Who are some of the most notable individuals of us american politics in the 21century '}
#     response = generate_response(client_query=sample_query, community_report=sample_msg["community_report"])
#     store_in_fs(response=response, user_query=sample_query, community_report=sample_msg["community_report"])
#     print("Hello World!")
