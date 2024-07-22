from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EdgeData:
    destination_uid: str
    target_uid: str 
    description: str


@dataclass
class NodeData:
    node_uid: str
    node_title: str
    node_type: str
    node_description: str
    node_degree: int
    document_id: str # identifier for source knowlede base document for this entity
    community_id: int | None = None # community id based on source document 
    edges_to: Tuple[str, ...] = field(default_factory=tuple)
    edges_from: Tuple[str, ...] = field(default_factory=tuple)  # in case of directed graph
    embedding: Tuple[float, ...] = field(default_factory=tuple)  # text embedding representing node e.g. combination of title & description


@dataclass
class CommunityData:
    document_id: str # identifier for source knowlede base document for this entity
    community_uid: str # community identifier
    community_nodes: Tuple[str, ...] = field(default_factory=tuple) # list of node_uid belonging to community
    community_title: str | None = None # title of comm, None if not yet computed
    community_description: str | None = None # description of comm, None if not yet computed
    community_embedding: Tuple[float, ...] = field(default_factory=tuple) # text embedding representing community