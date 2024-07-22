from dataclasses import dataclass
from typing import List

from networkx import edges



@dataclass
class EdgeData:
    destination_uid: str
    target_uid: str 
    description: str


@dataclass
class NodeData:
    document_id: str # identifier for source knowlede base document for this entity
    node_uid: str
    node_title: str
    node_type: str
    node_description: str
    node_degree: int
    community_id: int | None = None # community id based on source document 
    edges_to: List[EdgeData] = []
    edges_from: List[str] = [] # in case of directed graph
    embedding: List[float] = [] # text embedding representing node e.g. combination of title & description


@dataclass
class CommunityData:
    document_id: str # identifier for source knowlede base document for this entity
    community_uid: str # community identifier
    community_nodes: List[str] # list of node_uid belonging to community
    community_title: str | None = None # title of comm, None if not yet computed
    community_description: str | None = None # description of comm, None if not yet computed
    community_embedding: List[float] = [] # text embedding representing community