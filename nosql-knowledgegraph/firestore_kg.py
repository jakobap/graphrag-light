from matplotlib.pylab import source
from graph2nosql import NoSQLKnowledgeGraph
from data_model import NodeData, EdgeData, CommunityData

from typing import Dict, List
import datetime

import firebase_admin
from firebase_admin import firestore
import google.auth

import networkx as nx
import matplotlib.pyplot as plt


class FirestoreKG(NoSQLKnowledgeGraph):
    def __init__(self,
                 gcp_project_id: str,
                 gcp_credential_file: str,
                 firestore_db_id: str,
                 node_collection_id: str,
                 edges_collection_id: str,
                 community_collection_id: str
                 ) -> None:
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
                gcp_credential_file
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
        self.edges_coll_id = edges_collection_id
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
                other_node_data.edges_from = list(set(other_node_data.edges_from) | {node_uid})  # Add to edges_from
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

        for other_node_uid in node_data.edges_from:
            try:
                other_node_data = self.get_node(other_node_uid)
                other_node_data.edges_to = list(set(other_node_data.edges_from) | {node_uid})  # Add to edges_to
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
                other_node_data.edges_to = list(
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
                other_node_data.edges_from = list(
                    edge for edge in other_node_data.edges_from if edge != node_uid
                )
                self.update_node(other_node_uid, other_node_data)
            except KeyError:
                # If the other node doesn't exist, just continue
                continue

        # 4. Finally, remove the node itself
        doc_ref.delete()

    def add_edge(self, edge_data: EdgeData, directed: bool = True) -> None:
        """
        Adds an edge (relationship) between two entities in the knowledge graph.

        Args:
            source_uid (str): The UID of the source node.
            target_uid (str): The UID of the target node.
            edge_data (EdgeData): The edge data to be added.
            directed (bool, optional): Whether the edge is directed. Defaults to True.
        """

        # Check if source and target nodes exist
        if not self.get_node(edge_data.source_uid):
            raise KeyError(f"Error: Source node with node_uid '{edge_data.source_uid}' does not exist.")
        if not self.get_node(edge_data.target_uid):
            raise KeyError(f"Error: Target node with node_uid '{edge_data.target_uid}' does not exist.")

        # Type checking for edge_data
        if not isinstance(edge_data, EdgeData):
            raise TypeError(f"Error: edge_data must be of type EdgeData, not {type(edge_data)}")

        edge_uid = self._generate_edge_uid(edge_data.source_uid, edge_data.target_uid)

        try:
            source_node_data = self.get_node(edge_data.source_uid)
            target_node_data = self.get_node(edge_data.target_uid)

            source_node_data.edges_to = list(set(source_node_data.edges_to) | {edge_data.target_uid})
            self.update_node(edge_data.source_uid, source_node_data)

            # Add the edge to the target node's edges_from
            target_node_data.edges_from = list(set(target_node_data.edges_from) | {edge_data.source_uid})
            self.update_node(edge_data.target_uid, target_node_data)

            # Add the edge to the edges collection
            self._update_egde_coll(edge_uid=edge_uid,
                       target_uid=edge_data.target_uid,
                       source_uid=edge_data.source_uid,
                       description=edge_data.description)

            if not directed:  # If undirected, add the reverse edge as well
                target_node_data.edges_to = list(set(target_node_data.edges_to) | {edge_data.source_uid})
                self.update_node(edge_data.target_uid, target_node_data)

                # Since it's undirected, also add source_uid to target_node_data.edges_from
                source_node_data.edges_from = list(set(source_node_data.edges_from) | {edge_data.target_uid})
                self.update_node(edge_data.source_uid, source_node_data)

                # Add the reverse edge to the edges collection
                self._update_egde_coll(edge_uid=edge_uid,
                                       target_uid=edge_data.source_uid,
                                       source_uid=edge_data.target_uid,
                                       description=edge_data.description)
                                       

        except ValueError as e:
            raise ValueError(
                f"Error: Could not add edge from '{edge_data.source_uid}' to '{edge_data.target_uid}'. Details: {e}"
            ) from e
    
    def get_edge(self, source_uid: str, target_uid: str) -> EdgeData:
        """Retrieves an edge between two entities from the edges collection."""
        edge_uid = self._generate_edge_uid(source_uid, target_uid)
        edge_doc_ref = self.db.collection(self.edges_coll_id).document(edge_uid)
        doc_snapshot = edge_doc_ref.get()

        if doc_snapshot.exists:
            try:
                edge_data = EdgeData(**doc_snapshot.to_dict())
                return edge_data
            except TypeError as e:
                raise ValueError(
                    f"Error: Data fetched for edge_uid '{edge_uid}' does not match the EdgeData format. Details: {e}"
                ) from e
        else:
            raise KeyError(f"Error: No edge found with edge_uid: {edge_uid}")

    def update_edge(self, edge_data: EdgeData) -> None:
        """Updates an existing edge in the knowledge graph."""

        # 1. Validate input and check if the edge exists
        if not isinstance(edge_data, EdgeData):
            raise TypeError(f"Error: edge_data must be of type EdgeData, not {type(edge_data)}")

        edge_uid = self._generate_edge_uid(edge_data.source_uid, edge_data.target_uid)
        
        if not self.db.collection(self.edges_coll_id).document(edge_uid).get().exists:
            raise KeyError(f"Error: Edge with edge_uid '{edge_uid}' does not exist.")

        # 2. Update the edge document in the EDGES collection
        try:
            self._update_egde_coll(
                edge_uid=edge_uid,
                target_uid=edge_data.target_uid,
                source_uid=edge_data.source_uid,
                description=edge_data.description
            )
        except Exception as e:
            raise Exception(f"Error updating edge in edges collection: {e}") from e

        # 3. Update edge references in the NODES collection
        try:
            # 3a. Update source node
            source_node_data = self.get_node(edge_data.source_uid)
            # Ensure the target_uid is present in edges_to
            if edge_data.target_uid not in source_node_data.edges_to:
                source_node_data.edges_to = list(set(source_node_data.edges_to) | {edge_data.target_uid})
                self.update_node(edge_data.source_uid, source_node_data)

            # 3b. Update target node
            target_node_data = self.get_node(edge_data.target_uid)
            # Ensure the source_uid is present in edges_from
            if edge_data.source_uid not in target_node_data.edges_from:
                target_node_data.edges_from = list(set(target_node_data.edges_from) | {edge_data.source_uid})
                self.update_node(edge_data.target_uid, target_node_data)

        except Exception as e:
            raise Exception(f"Error updating edge references in nodes: {e}") from e

    def remove_edge(self, source_uid: str, target_uid: str, directed: bool = True) -> None:
        """
        Removes an edge between two entities in the knowledge graph. 
        It tries to remove references from both node and edge collections.

        Args:
            edge_uid (str): The UID of the edge.
            source_uid (str): The UID of the source node.
            target_uid (str): The UID of the target node.
            directed (bool, optional): Whether the edge is directed. Defaults to True.
        """

        edge_uid = self._generate_edge_uid(source_uid, target_uid)

        try:
            # --- 1. Remove edge references from the NODES collection ---
            try:
                source_node_data = self.get_node(source_uid)
                source_node_data.edges_to = list(edge for edge in source_node_data.edges_to if edge != target_uid)
                self.update_node(source_uid, source_node_data)

                if not directed: 
                    source_node_data.edges_from = list(edge for edge in source_node_data.edges_from if edge != target_uid)
                    self.update_node(source_uid, source_node_data)
            except KeyError:
                # Source node not found, move on to the next step
                pass 

            try:
                target_node_data = self.get_node(target_uid)
                target_node_data.edges_from = list(edge for edge in target_node_data.edges_from if edge != source_uid)
                self.update_node(target_uid, target_node_data)

                if not directed: 
                    target_node_data.edges_to = list(edge for edge in target_node_data.edges_to if edge != source_uid)
                    self.update_node(target_uid, target_node_data)
            except KeyError:
                # Target node not found, move on to the next step
                pass

            # --- 2. Remove edge document from the EDGES collection ---
            edge_doc_ref = self.db.collection(self.edges_coll_id).document(edge_uid)
            edge_doc_ref.delete()

        except Exception as e:
            raise Exception(f"Error: Could not remove edge. Details: {e}") from e

    def build_networkx(self):
        """Get the NetworkX representation of the full graph."""

        graph = nx.Graph()  # Initialize an undirected NetworkX graph

        # 1. Add Nodes to the NetworkX Graph
        nodes_ref = self.db.collection(self.node_coll_id).stream()
        for doc in nodes_ref:
            node_data = doc.to_dict()
            graph.add_node(doc.id, **node_data)

        # 2. Add Edges to the NetworkX Graph
        edges_ref = self.db.collection(self.edges_coll_id).stream()
        for doc in edges_ref:
            edge_data = doc.to_dict()
            source_uid = edge_data['source_uid']
            target_uid = edge_data['target_uid']
            # Consider adding edge attributes if needed (e.g., 'description')
            graph.add_edge(source_uid, target_uid)

        self.networkx = graph

    def list_communities(self) -> List[CommunityData]:
        """Lists all communities for the given network."""
        return []

    def _update_egde_coll(self, edge_uid: str, source_uid: str, target_uid: str, description: str) -> None:
        """Update edge record in the edges collection."""
        edge_doc_ref = self.db.collection(self.edges_coll_id).document(edge_uid)
        edge_data_dict = {
            "edge_uid": edge_uid,
            "source_uid": source_uid,
            "target_uid": target_uid,
            "description": description
        }
        edge_doc_ref.set(edge_data_dict)    

    def get_louvain_communities(self) -> list:
        """Computes Network communities and returns result as list."""
        # 1. Build (or update) the NetworkX graph
        self.build_networkx()

        # 2. Apply Louvain algorithm
        if self.networkx is not None:
            louvain_comm_list = nx.algorithms.community.louvain_communities(self.networkx)
            return louvain_comm_list # type: ignore
        else:
            raise ValueError("Error: NetworkX graph is not initialized. Call build_networkx() first.")
    
    def _generate_edge_uid(self, source_uid: str, target_uid: str):
        return f"{source_uid}_to_{target_uid}"

    def visualize_graph(self, filename: str= f"graph_{datetime.datetime.now()}.png") -> None:
        """Visualizes the provided networkx graph using matplotlib.

        Args:
            graph (nx.Graph): The graph to visualize.
        """
        self.build_networkx()
        
        if self.networkx is not None:
            # Create a larger figure for better visualization
            plt.figure(figsize=(12, 12))

            # Use a spring layout for a more visually appealing arrangement
            pos = nx.spring_layout(self.networkx, k=0.3, iterations=50)

            # Draw nodes with different colors based on entity type
            entity_types = set(data["node_type"] for _, data in self.networkx.nodes(data=True))
            color_map = plt.cm.get_cmap("tab10", len(entity_types))
            for i, entity_type in enumerate(entity_types):
                nodes = [n for n, d in self.networkx.nodes(
                    data=True) if d["node_type"] == entity_type]
                nx.draw_networkx_nodes(
                    self.networkx,
                    pos,
                    nodelist=nodes,
                    node_color=[color_map(i)],  # type: ignore
                    label=entity_type,
                    node_size=[10 + 50 * self.networkx.degree(n) for n in nodes] # type: ignore
                )

            # Draw edges with labels
            nx.draw_networkx_edges(self.networkx, pos, width=0.5, alpha=0.5)
            # edge_labels = nx.get_edge_attributes(graph, "description")
            # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

            # Add node labels with descriptions
            node_labels = {
                node: node
                for node, data in self.networkx.nodes(data=True)
            }
            nx.draw_networkx_labels(self.networkx, pos, labels=node_labels, font_size=8)

            plt.title("Extracted Knowledge Graph")
            plt.axis("off")  # Turn off the axis

            # Add a legend for node colors
            plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=entity_type,
                    markersize=10, markerfacecolor=color_map(i)) for i, entity_type in enumerate(entity_types)])
            
            plt.savefig(filename)
        
        else:
            raise ValueError("Error: NetworkX graph is not initialized. Call build_networkx() first.")

if __name__ == "__main__":
    import os
    from dotenv import dotenv_values

    os.chdir(os.path.dirname(os.path.abspath(__file__))) 

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
        edges_to=["test_node"],
        edges_from=[],
        embedding=[0.1, 0.2, 0.3], 
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
        edges_to=["test_node"],
        edges_from=[],
        embedding=[0.1, 0.2, 0.3], 
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

    print("Hello World!")