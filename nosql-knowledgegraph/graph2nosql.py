from data_model import NodeData, EdgeData, CommunityData

from abc import ABC, abstractmethod
from typing import Dict, List


class NoSQLKnowledgeGraph(ABC):
    """
    Base Class for storing and interacting with the KG and manages data model.  
    """

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
    def list_communities(self, community_uid: str) -> List[CommunityData]:
        """Lists all communities for the given network."""
        pass

    @abstractmethod
    def _update_communities(self) -> None:
        """Computes Network communities and updates datastore respectively."""
        pass

    def _generate_edge_uid(self, source_uid: str, target_uid: str) -> str:
        """Generates Edge uid for the network based on source and target nod uid"""
        return ""


if __name__ == "__main__":
    print("Hello World!")