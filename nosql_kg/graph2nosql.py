from re import A
import networkx
from nosql_kg.data_model import NodeData, EdgeData, CommunityData

from abc import ABC, abstractmethod
from typing import Dict, List

import networkx as nx


class NoSQLKnowledgeGraph(ABC):
    """
    Base Class for storing and interacting with the KG and manages data model.  
    """
    networkx: nx.Graph | None = None  # networkx representation of graph in nosqldb

    @abstractmethod
    def add_node(self, node_uid: str, node_data: NodeData) -> None:
        """Adds an node to the knowledge graph."""
        pass

    @abstractmethod
    def get_node(self, node_uid: str) -> NodeData:
        """Retrieves an node from the knowledge graph."""
        pass

    @abstractmethod
    def update_node(self, node_uid: str, node_data: NodeData) -> None:
        """Updates an existing node in the knowledge graph."""
        pass

    @abstractmethod
    def remove_node(self, node_uid: str) -> None:
        """Removes an node from the knowledge graph."""
        pass

    @abstractmethod
    def add_edge(self, edge_data: EdgeData, directed: bool = True) -> None:
        """Adds an edge (relationship) between two entities in the knowledge graph."""
        pass

    @abstractmethod
    def get_edge(self, source_uid: str, target_uid: str) -> EdgeData:
        """Retrieves an edge between two entities."""
        pass

    @abstractmethod
    def update_edge(self, edge_data: EdgeData) -> None:
        """Updates an existing edge in the knowledge graph."""
        pass

    @abstractmethod
    def remove_edge(self, source_uid: str, target_uid: str) -> None:
        """Removes an edge between two entities."""
        pass

    @abstractmethod
    def build_networkx(self) -> None:
        """Builds the NetworkX representation of the full graph.
        https://networkx.org/documentation/stable/index.html
        """
        pass

    @abstractmethod
    def get_louvain_communities(self) -> List[set[str]]:
        """Computes and returns all Louvain communities for the given network.
        https://www.nature.com/articles/s41598-019-41695-z

        Sample Output:
        [{'"2023 NOBEL PEACE PRIZE"'}, {'"ANDREI SAKHAROV PRIZE"'},
        {'"ANDREI SAKHAROV"'}, {'"ENCIEH ERFANI"', '"INSTITUTE FOR ADVANCED STUDIES IN BASIC SCIENCES IN ZANJAN, IRAN"'}]
        """
        pass

    @abstractmethod
    def store_communities(self, communities: List[CommunityData]) -> None:
        """Takes valid graph community data and stores it in the database.
        https://www.nature.com/articles/s41598-019-41695-z
        """
        pass

    @abstractmethod
    def _generate_edge_uid(self, source_uid: str, target_uid: str) -> str:
        """Generates Edge uid for the network based on source and target nod uid"""
        return ""

    @abstractmethod
    def visualize_graph(self, filename: str) -> None:
        """Visualizes the provided networkx graph using matplotlib."""
        pass


if __name__ == "__main__":
    print("Hello World!")
