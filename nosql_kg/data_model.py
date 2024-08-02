from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EdgeData:
    source_uid: str
    target_uid: str 
    description: str
    edge_uid: str | None = None
    document_id: str | None = None


@dataclass
class NodeData:
    node_uid: str
    node_title: str
    node_type: str
    node_description: str
    node_degree: int
    document_id: str # identifier for source knowlede base document for this entity
    community_id: int | None = None # community id based on source document 
    edges_to: list[str] = field(default_factory=list)
    edges_from: list[str] = field(default_factory=list)  # in case of directed graph
    embedding: list[float] = field(default_factory=list)  # text embedding representing node e.g. combination of title & description


@dataclass
class CommunityData:
    title: str # title of comm, None if not yet computed
    community_nodes: set[str] = field(default_factory=set) # list of node_uid belonging to community
    summary: str | None = None # description of comm, None if not yet computed
    document_id: str | None = None # identifier for source knowlede base document for this entity
    community_uid: str | None = None # community identifier
    community_embedding: Tuple[float, ...] = field(default_factory=tuple) # text embedding representing community
    rating: int | None = None
    rating_explanation: str | None = None
    findings: list[dict] | None = None