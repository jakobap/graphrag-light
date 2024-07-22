from dataclasses import dataclass


@dataclass
class NodeData:
    name: str
    description: str = None


@dataclass
class EdgeData:
    name: str
    description: str = None


@dataclass
class CommunityData:
    name: str
    description: str = None