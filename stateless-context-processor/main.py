import json
import logging
import traceback
import os

from LLMSession import LLMSession

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values, load_dotenv

import firebase_admin
from firebase_admin import firestore

import google.auth


app = FastAPI()

"""Prompt templates for global search pipe."""

MAP_SYSTEM_PROMPT = """
---Role---
You are an expert agent answering questions based on context that is organized as a knowledge graph.
You will be provided with exactly one community report extracted from that same knowledge graph.


---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the given community report.

You should use the data provided in the community description below as the only context for generating the response.
If you don't know the answer or if the input community description does not contain sufficient information to provide an answer respond "The user question cannot be answered based on the given community context.".

Your response should always contain following elements:
- Query based response: A comprehensive and truthful response to the given user query, solely based on the provided context.
- Importance Score: An integer score between 0-10 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{"response": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}}
"""

MAP_QUERY_PROMPT = """
---Context Community Report---
{context_community_report}

---User Question---
{user_question}

---JSON Response---
The json response formatted as follows:
{{"response": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}}

response: 
"""

@observe()
def generate_response(client_query: str, community_report: dict):

    load_dotenv(".env")

    llm = LLMSession(
        system_message=MAP_SYSTEM_PROMPT,
        model_name="gemini-1.5-pro-001"
    )

    response_schema = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
            },
            "score": {
                "type": "integer",
            },
        },
        "required": ["response", "score"],
    }

    query_prompt = MAP_QUERY_PROMPT.format(
        context_community_report=community_report, user_question=client_query)

    response = llm.generate(client_query_string=query_prompt,
                 response_schema=response_schema,
                 response_mime_type="application/json")
    
    print(f"Response for Community: {community_report["title"]} & Query: {client_query}: {response}")

    langfuse_context.flush()
    return response


def store_in_fs(response: str, user_query: str, community_report: dict) -> None:
    """
    Stores the LLM response in Firestore.

    Args:
        response (str): The JSON formatted LLM response.
        user_query (str): The original user query.
        community_report (dict): The community report used for the response.
    """
    gcp_credentials, project_id = google.auth.load_credentials_from_file(str(os.getenv("GCP_CREDENTIAL_FILE")))

    db = firestore.Client(project=os.getenv("GCP_PROJECT_ID"),  # type: ignore
                           credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),  # Use correct env variable
                           database=os.getenv("QUERY_FS_DB_ID"))  # Use database ID from .env

    # Extract community title for structuring data
    community_title = community_report.get("title", "Unknown Community")

    # Parse the JSON response
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {e}")
        return  # Exit early on error

    # Prepare data for Firestore
    data = {
        "community": community_title,
        "response": response_dict.get("response", ""),
        "score": response_dict.get("score", 0)
    }

    # Get a reference to the document (using user_query as key)
    doc_ref = db.collection(os.getenv("QUERY_FS_INT__RESPONSE_COLL")).document(user_query)  # Use collection ID from .env

    # Get the existing data if any
    doc_snapshot = doc_ref.get()
    existing_data = doc_snapshot.to_dict() if doc_snapshot.exists else {}

    # Append the new data to the existing list or create a new list
    responses_list = existing_data.get("responses", [])
    responses_list.append(data)

    # Update the document with the new list
    doc_ref.set({"responses": responses_list}, merge=True)

    logging.info(f"Stored response for query '{user_query}' and community '{community_title}' in Firestore.")
    print("saving in fs done")

    return None

@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}


@app.post("/receive_analysis_request")
async def trigger_analysis(request: Request):
    """Receive and parse Pub/Sub messages."""
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
            client_query=message_dict["user_query"],
            community_report=message_dict["community_report"]
        )

        store_in_fs(response=response_json, user_query=message_dict["user_query"], community_report=message_dict["community_report"])

        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during file analysis: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)
