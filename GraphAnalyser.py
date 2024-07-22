
from enum import nonmember
from matplotlib.pylab import noncentral_f
from IngestionSession import IngestionSession
from GraphExtractor import GraphExtractor

import html
from typing import Any, Dict, cast
from dataclasses import dataclass

import networkx as nx

from graspologic.utils import largest_connected_component
from graspologic.partition import hierarchical_leiden


@dataclass
class KnowledgeGraph:
    graph: nx.Graph
    communities: Dict


class GraphAnalyser:
    def __init__(self) -> None:
        self.communities = None

    def __call__(self, graph: nx.Graph) -> KnowledgeGraph:
        #compute communities
        comm = self._compute_leiden_communities(graph=graph,
                                                max_cluster_size=10,
                                                use_lcc=True)

        # summarize communities
        
        kg = KnowledgeGraph(graph=graph, communities=comm)
        return kg

    # Taken from graph_intelligence & adapted
    def _compute_leiden_communities(
        self,
        graph: nx.Graph | nx.DiGraph,
        max_cluster_size: int,
        use_lcc: bool,
        seed=0xDEADBEEF,
    ) -> dict[int, dict[str, int]]:
        print("leiden computing")
        """Return Leiden root communities."""
        if use_lcc:
            graph = self._stable_largest_connected_component(graph)

        community_mapping = hierarchical_leiden(
            graph, max_cluster_size=max_cluster_size, random_seed=seed
        )
        results: dict[int, dict[str, int]] = {}
        for partition in community_mapping:
            results[partition.level] = results.get(partition.level, {})
            results[partition.level][partition.node] = partition.cluster

        return results
    
    def _stable_largest_connected_component(self, graph: nx.Graph) -> nx.Graph:
        """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        graph = self._normalize_node_names(graph)
        return self._stabilize_graph(graph)

    def _stabilize_graph(self, graph: nx.Graph) -> nx.Graph:
        """Ensure an undirected graph with the same relationships will always be read the same way."""
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        # If the graph is undirected, we create the edges in a stable way, so we get the same results
        # for example:
        # A -> B
        # in graph theory is the same as
        # B -> A
        # in an undirected graph
        # however, this can lead to downstream issues because sometimes
        # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
        # but they base some of their logic on the order of the nodes, so the order ends up being important
        # so we sort the nodes in the edge in a stable way, so that we always get the same order
        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph
    
    def _normalize_node_names(self, graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
        """Normalize node names."""
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        return nx.relabel_nodes(graph, node_mapping)

if __name__ == "__main__":
    ingestion = IngestionSession()
    extractor = GraphExtractor()
    analyser = GraphAnalyser()

    document_string = ingestion(
        new_file_name="./pdf_articles/2024 United States presidential election - Wikipedia.pdf", ingest_local_file=True
    )

    extracted_graph = extractor(text_input=document_string)

    extractor.visualize_graph(extracted_graph)

    kg = analyser(graph=extracted_graph)
    print(kg)
    print("Hello World!")
