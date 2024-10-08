from http import client
import json
import logging
import traceback
import os

from LLMSession import LLMSession

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values

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

"""

MAP_QUERY_PROMPT = """
---Context Community Report---
{context_community_report}

---User Question---
{user_question}

Call extract_json_schema with the question response and context relevance score.

---Question Response & Context Relevance Score---
"""

@observe()
def generate_response(client_query: str, community_report: dict):

    langfuse_context.update_current_trace(
        name="Community Intermediate Query Gen",
        session_id=client_query,
        public=False
    )

    llm = LLMSession(
        system_message=MAP_SYSTEM_PROMPT,
        model_name="gemini-1.5-pro-001"
    )

    # llm_flash = LLMSession(
    #     system_message=MAP_SYSTEM_PROMPT,
    #     model_name="gemini-1.5-flash-001"
    # )

    response_schema = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The response to the user question as raw string.",
            },
            "score": {
                "type": "number",
                "description": "The relevance score of the given community report context towards answering the user question [0.0, 10.0]",
            },
        },
        "required": ["response", "score"],
    }

    query_prompt = MAP_QUERY_PROMPT.format(
        context_community_report=community_report, user_question=client_query)

    response = llm.generate(client_query_string=query_prompt,
                 response_schema=response_schema,
                 response_mime_type="application/json")

    # response = llm_flash.function_call_gen(client_query_string=query_prompt,
    #                                  response_schema=response_schema)
    
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
    secrets = dotenv_values(".env")
    gcp_credentials, project_id = google.auth.load_credentials_from_file(str(secrets["GCP_CREDENTIAL_FILE"]))

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
    doc_ref = db.collection(secrets["QUERY_FS_INT__RESPONSE_COLL"]).document(user_query)  # Use collection ID from .env

    # Get the existing data if any
    doc_snapshot = doc_ref.get()
    existing_data = doc_snapshot.to_dict() if doc_snapshot.exists else {}

    existing_data[community_title] = refreshed_data

    # Update the document with the new list
    doc_ref.set(existing_data, merge=True)

    logging.info(f"Stored response for query '{user_query}' and community '{community_title}' in Firestore.")
    print("saving in fs done")


@app.get("/helloworld")
async def helloworld():
    return {"message": "Hello World"}


@app.post("/receive_analysis_request")
async def trigger_analysis(request: Request):
    """Receive and parse Pub/Sub messages."""
    secrets = dotenv_values(".env")

    os.environ["LANGFUSE_SECRET_KEY"] = str(
            secrets["LANGFUSE_SECRET_KEY"])
    os.environ["LANGFUSE_PUBLIC_KEY"] = str(
            secrets["LANGFUSE_PUBLIC_KEY"])
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

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

        print("analysis done")
        return JSONResponse(content={"message": "File analysis completed successfully!"}, status_code=200)
    except Exception as e:
        msg = f"Something went wrong during file analysis: {e}"
        logging.error(msg)
        traceback.print_exc()
        return JSONResponse(content={"message": msg}, status_code=500)


# if __name__ == "__main__":

#     sample_query = "Who are some of the most notable individuals of us american politics in the 21century"
#     sample_msg = {'community_report': {'title': 'Joe Biden, Kamala Harris, and the Democratic Party', 'community_nodes': ['KAMALA HARRIS', 'DEMOCRATIC PARTY', 'JOE BIDEN'], 'summary': "This community centers around the political figures of Joe Biden and Kamala Harris, and their relationship to the Democratic Party in the context of the 2024 US Presidential election. Joe Biden, the previous president, withdrew from the race, leading Kamala Harris to become the party's nominee.", 'document_id': None, 'community_uid': None, 'community_embedding': [], 'rating': 8.5, 'rating_explanation': 'The impact severity is high due to the significant influence these individuals and the Democratic Party hold in the US political landscape, particularly during a presidential election year.', 'findings': [{'explanation': "Joe Biden, the 46th US President, withdrew from the 2024 presidential race due to age, health, and political reasons. This decision significantly impacted the Democratic Party's trajectory for the election. [Data: Entities ('JOE BIDEN'), ('DEMOCRATIC PARTY')]", 'summary': "Joe Biden's Withdrawal and its Impact"}, {'explanation': "Following Biden's withdrawal, Kamala Harris, the current US Vice President, became the Democratic Party's nominee for president. This shift in candidacy places Harris in a prominent position within the party and the election. [Data: Entities ('KAMALA HARRIS'), ('DEMOCRATIC PARTY')]", 'summary': "Kamala Harris's Nomination"}, {'explanation': "The Democratic Party, as one of the two major political parties in the US, plays a crucial role in the presidential election. The party's dynamics and decisions significantly impact the country's political landscape. [Data: Entities ('DEMOCRATIC PARTY')]", 'summary': "The Democratic Party's Role"}]}, 'user_query': 'Who are some of the most notable individuals of us american politics in the 21century '}

#     response = generate_response(client_query=sample_query, community_report=sample_msg["community_report"]) 
#     store_in_fs(response=response, user_query=sample_query, community_report=sample_msg["community_report"]) 

#     sample_msg ={'community_report': {'title': 'Count Bernadotte and the UN', 'community_nodes': ['COUNT BERNADOTTE', 'UN'], 'summary': "This community centers around the historical figure of Count Bernadotte, a Swedish UN envoy to Palestine who was murdered in 1948. The relationship highlights Bernadotte's diplomatic role within the UN's involvement in the Palestine situation during that era.", 'document_id': None, 'community_uid': None, 'community_embedding': [], 'rating': 2.0, 'rating_explanation': 'The impact severity rating is low because the events surrounding this community are historical and do not pose an ongoing threat.', 'findings': [{'explanation': "Count Bernadotte played a significant historical role as a Swedish UN envoy to Palestine. His assassination in 1948 underscores the volatile political climate of the region at the time.  [Data: Entities ('COUNT BERNADOTTE'), Relationships ('COUNT BERNADOTTE_to_UN')]", 'summary': "Count Bernadotte's Diplomatic Role"}, {'explanation': "The UN's involvement in Palestine during this period, as evidenced by Bernadotte's envoy role, highlights the organization's early engagement in seeking resolutions to the Arab-Israeli conflict. [Data: Entities ('UN'), Relationships ('COUNT BERNADOTTE_to_UN')]", 'summary': 'UN Involvement in Palestine'}]}, 'user_query': 'Who are some of the most notable individuals of us american politics in the 21century '}
#     response = generate_response(client_query=sample_query, community_report=sample_msg["community_report"]) 
#     store_in_fs(response=response, user_query=sample_query, community_report=sample_msg["community_report"]) 
#     print("Hello World!")
