from graph2nosql import NoSQLKnowledgeGraph
from data_model import NodeData, EdgeData, CommunityData

from typing import Dict, List

import firebase_admin
from firebase_admin import firestore
import google.auth


class FirestoreKG(NoSQLKnowledgeGraph):
    def __init__(self,
                 gcp_project_id: str,
                 gcp_credential_file: str,
                 firestore_db_id: str,
                 node_collection_id: str,
                 community_collection_id: str) -> None:
        """
        Initializes the FirestoreKG object.

        Args:
            project_id (str): The Google Cloud project ID.
            database_id (str): The ID of the Firestore database.
            collection_name (str): The name of the collection to store the KG.
        """
        super().__init__()

        if not firebase_admin._apps:
            credentials = firebase_admin.credentials.Certificate(
                firestore_credential_file
            )
            app = firebase_admin.initialize_app(credentials)


        self.credentials, self.project_id = google.auth.load_credentials_from_file(
            gcp_credential_file)

        self.db = firestore.Client(project=gcp_project_id, # type: ignore
                                   credentials=self.credentials,
                                   database=firestore_db_id) 

        self.gcp_project_id = gcp_project_id
        self.database_id = firestore_db_id
        self.node_coll_id = node_collection_id
        self.community_coll_id = community_collection_id

    def add_node(self, node_uid: str, node_data: NodeData) -> None:
        """Adds an node to the knowledge graph."""
        pass

    def get_node(self, node_uid: str) -> NodeData:
        """Retrieves an node from the knowledge graph."""
        doc_ref = self.db.collection(self.node_coll_id).document(node_uid)
        doc_snapshot = doc_ref.get()

        if doc_snapshot.exists:
            try:
                node_data = NodeData(**doc_snapshot.to_dict())
                return node_data
            except TypeError as e:
                raise ValueError(
                    f"Error: Data fetched for node_uid '{node_uid}' does not match the NodeData format. Details: {e}"
                ) from e
        else:
            raise KeyError(f"Error: No node found with node_uid: {node_uid}")

    
    def update_node(self, node_uid: str, node_data: NodeData) -> None:
        """Updates an existing node in the knowledge graph."""
        pass
    
    def remove_node(self, node_uid: str) -> None:
        """Removes an node from the knowledge graph."""
        pass
    
    def add_edge(self, source_uid: str, target_uid: str, edge_data: EdgeData) -> None:
        """Adds an edge (relationship) between two entities in the knowledge graph."""
        pass
    
    def get_edge(self, source_uid: str, target_uid: str, relationship: str) -> EdgeData:
        """Retrieves an edge between two entities."""
        pass

    def update_edge(self, source_uid: str, target_uid: str, edge_data: EdgeData) -> None:
        """Updates an existing edge in the knowledge graph."""
        pass

    def remove_edge(self, source_uid: str, target_uid: str) -> None:
        """Removes an edge between two entities."""
        pass

    def list_communities(self, community_uid: str) -> List[CommunityData]:
        """Lists all communities for the given network."""
        pass

    def _update_communities(self) -> None:
        """Computes Network communities and updates datastore respectively."""
        pass


if __name__ == "__main__":
    import os
    from dotenv import dotenv_values

    os.chdir(os.path.dirname(os.path.abspath(__file__))) 

    secrets = dotenv_values(".env")

    firestore_credential_file = str(secrets["GCP_CREDENTIAL_FILE"])
    project_id = str(secrets["GCP_PROJECT_ID"])
    database_id = str(secrets["FIRESTORE_DB_ID"])
    node_coll_id = str(secrets["NODE_COLL_ID"])
    community_coll_id = str(secrets["COMM_COLL_ID"])

    fskg = FirestoreKG(
        gcp_project_id=project_id,
        gcp_credential_file=firestore_credential_file,
        firestore_db_id=database_id,
        node_collection_id=node_coll_id,
        community_collection_id=community_coll_id
    )

    node_info = fskg.get_node(node_uid="test_node")
    
    print("Hello World!")