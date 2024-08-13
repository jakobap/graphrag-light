from abc import ABC, abstractmethod


class MQ(ABC):
    @abstractmethod
    def send_to_mq(message) -> None:
        """Method to submit a workitem/message to your mq."""
        pass



from google.cloud import pubsub_v1
from dotenv import dotenv_values
import google.auth
import json


class PubSubMQ(MQ):
    def __init__(self, pubsub_topic_id: str) -> None:
        super().__init__()
        self.secrets = dotenv_values(".env")
        self.gcp_credentials, self.project_id = google.auth.load_credentials_from_file(
            str(self.secrets["GCP_CREDENTIAL_FILE"]))
        
        self.pubsub_topic_id = pubsub_topic_id
    
    def send_to_mq(self, message: dict) -> None:
        """Publishes one message to a Pub/Sub topic."""
        publisher = pubsub_v1.PublisherClient(credentials=self.gcp_credentials)
        topic_path = publisher.topic_path(
            str(self.secrets["GCP_PROJECT_ID"]), self.pubsub_topic_id)

        # Publish the message
        mes = json.dumps(message)
        future = publisher.publish(
            topic_path, mes.encode("utf-8"))

        # (Optional) Wait for the publish future to resolve
        message_id = future.result()
        # print(f"Published message ID: {message_id} with payload: {message} to topic: {topic_path}")
        return None