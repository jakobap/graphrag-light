from dataclasses import dataclass
from pyparsing import abstractmethod
from typing import Any
import json
from dotenv import dotenv_values
import time

import google.auth
from google.cloud import pubsub_v1

from nosql_kg.firestore_kg import FirestoreKG
from nosql_kg import data_model


@dataclass
class CommunityAnswerRequest:
    """
    Dataclass describing a Workload request made to the message queue for async generation of community based intermediate user query answers.
    """

    community_report: data_model.CommunityData
    user_query: str

    def __to_dict__(self):
        return {
            "community_report": self.community_report.__to_dict__(),
            "user_query": self.user_query
        }


class KGraphGlobalQuery:
    def __init__(self) -> None:
        # initialized with info on mq, knowledge graph, shared nosql state
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # orchestration method taking natural language user query to produce and return final answer to client

        # pair user query with existing community reports

        # send pairs to pubsub queue for work scheduling

        # periodically query shared state to check for processing compeltion

        # get intermediate responses

        # based on helpfulness build final context

        # generate & return final response based on final context

        pass

    @abstractmethod
    def _send_to_mq(self, message: CommunityAnswerRequest):
        # method to send to message queue for async work scheduling
        pass

    @abstractmethod
    def _get_comm_reports(self):
        # method to get all community reports from kg
        pass

    @abstractmethod
    def _check_shared_state(self, user_query:str, max_attempts: int = 10, sleep_time: int = 2):
        # method to check shared state for query result
        # method to query shared state for intermediate responses for one given user query
        pass

    def _context_builder(self):
        # given a user query pulls community reports and sends (query, community) objects for distributed LLM inference
        pass

    def _build_final_context(self):
        # method to generate final context based on helpfulness of top intermediate responses

        # final context includes community info for most helpful based on int responses
        # final context might also include node descriptions of the community members
        pass


class GlobalQueryGCP(KGraphGlobalQuery):
    def __init__(self, secrets: dict, fskg: FirestoreKG) -> None:
        super().__init__()

        self.secrets = secrets

        self.gcp_credentials, self.project_id = google.auth.load_credentials_from_file(
            str(self.secrets["GCP_CREDENTIAL_FILE"]))
        
        self.fskg = fskg

    def _send_to_mq(self, message: CommunityAnswerRequest) -> None:
        """Publishes one message to a Pub/Sub topic."""
        publisher = pubsub_v1.PublisherClient(credentials=self.gcp_credentials)
        topic_path = publisher.topic_path(
            str(self.secrets["GCP_PROJECT_ID"]), str(self.secrets["SCHEDULER_PUBSUB_ID"]))

        # Encode message into bytes
        # message_data = message.encode('utf-8')

        # Publish the message
        mes = json.dumps(message.__to_dict__())
        print(mes)
        # print(json.loads(mes))
        future = publisher.publish(
            topic_path, mes.encode("utf-8"))

        # (Optional) Wait for the publish future to resolve
        message_id = future.result()
        # print(f"Published message ID: {message_id} with payload: {message} to topic: {topic_path}")
        return None

    def _get_comm_reports(self) -> list[data_model.CommunityData]:
        """
        Get Full List of communtiy reports from firebase Knowledge Graph Store
        """
        comm_coll = str(self.secrets["COMM_COLL_ID"])
        docs = self.fskg.db.collection(comm_coll)
        return [data_model.CommunityData.__from_dict__(doc.to_dict()) for doc in docs.stream()]

    def _check_shared_state(self, user_query:str, max_attempts: int = 5, sleep_time: int = 5) -> dict:
        """
        Periodically checks for the existence of a document with user_query as id. 

        Args:
            user_query (str): The ID of the document to look for.
            max_attempts (int, optional): Maximum number of attempts to check. Defaults to 10.
            sleep_time (int, optional): Time to sleep between attempts in seconds. Defaults to 2.

        Returns:
            Dict: The document data if found, otherwise raises timeout error.
        """
        for attempt in range(max_attempts):
            doc_ref = self.fskg.db.collection(str(self.secrets["INTERMEDIATE_ANSWER_COLLECTION"])).document(user_query)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                return doc_snapshot.to_dict()
            else:
                print(f"Attempt {attempt+1}/{max_attempts}: Document not found, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            
        raise TimeoutError(f"Document with ID '{user_query}' not found after {max_attempts} attempts.")


if __name__ == "__main__":
    secrets = dotenv_values(".env")

    firestore_credential_file = str(secrets["GCP_CREDENTIAL_FILE"])
    project_id = str(secrets["GCP_PROJECT_ID"])
    database_id = str(secrets["FIRESTORE_DB_ID"])
    node_coll_id = str(secrets["NODE_COLL_ID"])
    edges_coll_id = str(secrets["EDGES_COLL_ID"])
    community_coll_id = str(secrets["COMM_COLL_ID"])

    fskg = FirestoreKG(
        gcp_project_id=project_id,
        gcp_credential_file=firestore_credential_file,
        firestore_db_id=database_id,
        node_collection_id=node_coll_id,
        edges_collection_id=edges_coll_id,
        community_collection_id=community_coll_id
    )

    global_query = GlobalQueryGCP(secrets, fskg)

    comm_reports = global_query._get_comm_reports()

    test_request = CommunityAnswerRequest(
        community_report=comm_reports[0],
        user_query="Which city has the most bridges?"
    )

    global_query._send_to_mq(message=test_request)

    print("Hello World!")
