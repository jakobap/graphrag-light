from dataclasses import dataclass
from pyparsing import abstractmethod
from typing import Any
import json
from dotenv import dotenv_values
import time
from operator import attrgetter

import google.auth
from google.cloud import pubsub_v1

import firebase_admin
from firebase_admin import firestore

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


@dataclass
class IntermediateCommRespose:
    community: str
    response: str
    score: int

    def to_dict(self):
        return {
            "community": self.community,
            "response": self.response,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            community=str(data.get("community")),
            response=str(data.get("response")),
            score=int(data.get("score", 0))
            )


class KGraphGlobalQuery:
    def __init__(self) -> None:
        # initialized with info on mq, knowledge graph, shared nosql state
        pass

    def __call__(self, user_query: str) -> str:
        # orchestration method taking natural language user query to produce and return final answer to client
        comm_report_list = self._get_comm_reports()

        # pair user query with existing community reports
        query_msg_list = self._context_builder(
            user_query=user_query, comm_report_list=comm_report_list)

        # send pairs to pubsub queue for work scheduling
        # for msg in query_msg_list:
        #     self._send_to_mq(message=msg)

        # periodically query shared state to check for processing compeltion & get intermediate responses
        intermediate_response_list = self._check_shared_state(
            user_query=user_query)

        # based on helpfulness build final context
        sorted_final_responses = self._filter_and_sort_responses(intermediate_response_list=intermediate_response_list)

        # get full community reports for the selected communities
        comm_report_list = [fskg.get_community(r.community) for r in sorted_final_responses]


        # generate & return final response based on final context community repors and nodes.


        return ""

    @abstractmethod
    def _send_to_mq(self, message: CommunityAnswerRequest):
        # method to send to message queue for async work scheduling
        pass

    @abstractmethod
    def _get_comm_reports(self) -> list[data_model.CommunityData]:
        # method to get all community reports from kg
        pass

    @abstractmethod
    def _check_shared_state(self, user_query: str,
                            max_attempts: int = 6,
                            sleep_time: int = 10) -> list[IntermediateCommRespose]:
        # method to check shared state for query result
        # method to query shared state for intermediate responses for one given user query
        pass

    def _context_builder(self, user_query: str, comm_report_list: list[data_model.CommunityData]) -> list[CommunityAnswerRequest]:
        # given a user query pulls community reports and sends (query, community) objects for distributed LLM inference
        comm_answer_request_list = [CommunityAnswerRequest(
            community_report=c, user_query=user_query) for c in comm_report_list]
        return comm_answer_request_list

    def _build_final_context(self, user_query: str, report: data_model.CommunityData):
        # method to generate final context based on helpfulness of top intermediate responses
        # final context includes community info for most helpful based on int responses
        # final context might also include node descriptions of the community members

        return ""

    def _filter_and_sort_responses(self,
                                   intermediate_response_list: list[IntermediateCommRespose],
                                   relevance_threshhold: int = 0,
                                   max_responses: int = 10):
        """
        Filters out responses with score 0 and sorts the remaining responses 
        by score value in descending order.

        Args:
            intermediate_response_list: A list of IntermediateCommRespose objects.
            relevance_threshhold: Minimum relevance score to be included in final context.
            max_responses: Maximum number of intermediate resposes to be included in final context.

        Returns:
            A list of IntermediateCommRespose objects with score > 0, sorted by 
            score in descending order.
        """
        # Filter out responses with score 0
        filtered_responses = [
            response for response in intermediate_response_list if response.score > relevance_threshhold
        ]
        # Sort the remaining responses by score in descending order
        sorted_responses = sorted(
            filtered_responses, key=attrgetter('score'), reverse=True
        )
        return sorted_responses[:max_responses]

class GlobalQueryGCP(KGraphGlobalQuery):
    def __init__(self, secrets: dict, fskg: FirestoreKG) -> None:
        super().__init__()

        self.secrets = secrets

        self.gcp_credentials, self.project_id = google.auth.load_credentials_from_file(
            str(self.secrets["GCP_CREDENTIAL_FILE"]))

        self.fskg = fskg

        if not firebase_admin._apps:
            credentials = firebase_admin.credentials.Certificate(
                str(self.secrets["GCP_CREDENTIAL_FILE"])
            )
            app = firebase_admin.initialize_app(credentials)

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

    def _check_shared_state(self, user_query: str,
                            max_attempts: int = 6,
                            sleep_time: int = 10) -> list[IntermediateCommRespose]:
        """
        Periodically checks for the existence of a document with user_query as id. 

        Args:
            user_query (str): The ID of the document to look for.
            max_attempts (int, optional): Maximum number of attempts to check. Defaults to 10.
            sleep_time (int, optional): Time to sleep between attempts in seconds. Defaults to 2.

        Returns:
            Dict: The document data if found, otherwise raises timeout error.
        """
        query_db = firestore.Client(project=self.project_id,  # type: ignore
                              credentials=self.gcp_credentials,
                              database=str(self.secrets["QUERY_FS_DB_ID"]))
        
        comm_list = fskg.list_communities()

        time.sleep(5)
        for attempt in range(max_attempts):
            doc_ref = query_db.collection(
                str(self.secrets["QUERY_FS_INT__RESPONSE_COLL"])).document(user_query)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                num_stored_responses = len(doc_snapshot.to_dict().keys())
                if num_stored_responses == len(comm_list):
                    responses = doc_snapshot.to_dict()
                    comms = responses.keys()
                    return [IntermediateCommRespose.from_dict(responses[c]) for c in comms]
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

    # comm_reports = global_query._get_comm_reports()

    global_query(user_query="Who won the nobel peace prize 2023?")

    # for i in range(len(comm_reports)):
    #     test_request = CommunityAnswerRequest(
    #         community_report=comm_reports[i],
    #         user_query="Who won the nobel peace prize 2023?"
    #     )

    #     global_query._send_to_mq(message=test_request)

    print("Hello World!")
