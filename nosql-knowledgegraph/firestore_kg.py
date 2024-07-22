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
        doc_ref = self.db.collection(self.node_coll_id).document(node_uid)

        # Check if a node with the same node_uid already exists
        if doc_ref.get().exists:
            raise ValueError(f"Error: Node with node_uid '{node_uid}' already exists.")

        # Convert NodeData to a dictionary for Firestore storage
        try:
            node_data_dict = node_data.__dict__
        except TypeError as e:
            raise ValueError(
                f"Error: Provided node_data for node_uid '{node_uid}' cannot be converted to a dictionary. Details: {e}"
            ) from e

        # Set the document ID to match the node_uid
        try:
            doc_ref.set(node_data_dict)
        except ValueError as e:
            raise ValueError(
                f"Error: Could not add node with node_uid '{node_uid}' to Firestore. Details: {e}"
            ) from e
        
        # Update references in other nodes
        for other_node_uid in node_data.edges_to:
            try:
                other_node_data = self.get_node(other_node_uid)
                other_node_data.edges_from = tuple(set(other_node_data.edges_from) | {node_uid})  # Add to edges_from
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

        for other_node_uid in node_data.edges_from:
            try:
                other_node_data = self.get_node(other_node_uid)
                other_node_data.edges_to = tuple(set(other_node_data.edges_to) | {node_uid})  # Add to edges_to
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

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
        doc_ref = self.db.collection(self.node_coll_id).document(node_uid)

        # Check if the node exists
        if not doc_ref.get().exists:
            raise KeyError(f"Error: Node with node_uid '{node_uid}' does not exist.")

        # Convert NodeData to a dictionary for Firestore storage
        try:
            node_data_dict = node_data.__dict__
        except TypeError as e:
            raise ValueError(
                f"Error: Provided node_data for node_uid '{node_uid}' cannot be converted to a dictionary. Details: {e}"
            ) from e

        # Update the document
        try:
            doc_ref.update(node_data_dict)
        except ValueError as e:
            raise ValueError(
                f"Error: Could not update node with node_uid '{node_uid}' in Firestore. Details: {e}"
            ) from e
    
    def remove_node(self, node_uid: str) -> None:
        """
        Removes an node from the knowledge graph.
        Also removed all edges to and from the node to be removed from all other nodes.
        """
        doc_ref = self.db.collection(self.node_coll_id).document(node_uid)

        # Check if the node exists
        if not doc_ref.get().exists:
            raise KeyError(f"Error: Node with node_uid '{node_uid}' does not exist.")

        # 1. Get the node data to find its connections
        node_data = self.get_node(node_uid)

        # 2. Remove connections TO this node from other nodes
        for other_node_uid in node_data.edges_from:
            try:
                other_node_data = self.get_node(other_node_uid)
                other_node_data.edges_to = tuple(
                    edge for edge in other_node_data.edges_to if edge != node_uid
                )
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

        # 3. Remove connections FROM this node to other nodes
        for other_node_uid in node_data.edges_to:
            try:
                other_node_data = self.get_node(other_node_uid)
                other_node_data.edges_from = tuple(
                    edge for edge in other_node_data.edges_from if edge != node_uid
                )
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

        # 4. Finally, remove the node itself
        doc_ref.delete()

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

    # Test get_node method
    node_info = fskg.get_node(node_uid="test_node")

    # Test add_node method
    test_node_data = NodeData(
        node_uid="test_node_add",
        node_title="Added Test Node",
        node_type="Added Test Node Type",
        node_description="This is a test node",
        node_degree=0,
        document_id="test_doc_id",
        edges_to=("test_node",),
        edges_from=(),
        embedding=(0.1, 0.2, 0.3), 
    )

    try:
        fskg.add_node(node_uid="test_node_add", node_data=test_node_data)
        print("Test add_node: PASSED")
    except Exception as e:
        print(f"Test add_node: FAILED - {e}")

    # Test update_node method
    test_updated_node_data = NodeData(
        node_uid="test_node_add",
        node_title="Added Test Node",
        node_type="Added Test Node Type",
        node_description="This is the updated test node description",
        node_degree=0,
        document_id="test_doc_id",
        edges_to=("test_node",),
        edges_from=(),
        embedding=(0.1, 0.2, 0.3), 
    )

    try:
        fskg.update_node(node_uid="test_node_add", node_data=test_updated_node_data)
        print("Test update_node: PASSED")
    except Exception as e:
        print(f"Test update_node: FAILED - {e}")

    # Get both nodes to check updates
    node_info = fskg.get_node(node_uid="test_node")
    node_info = fskg.get_node(node_uid="test_node_add")

    # Test remove_node method
    try:
        fskg.remove_node(node_uid="test_node_add")
        print("Test remove_node: PASSED")
    except Exception as e:
        print(f"Test remove_node: FAILED - {e}")

    # Get both nodes to deletion & corresponding edge update in remaining
    node_info = fskg.get_node(node_uid="test_node")
    node_info = fskg.get_node(node_uid="test_node_add")

    print("Hello World!")