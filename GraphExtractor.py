from cgitb import text
from html import entities
from typing import Any

from httpx import get
from IngestionSession import IngestionSession
from LLMSession import LLMSession

from nosql_kg.graph2nosql import NoSQLKnowledgeGraph
import prompts
from nosql_kg.firestore_kg import FirestoreKG
from nosql_kg import data_model

import networkx as nx

import re
import numbers
import html
import datetime
from collections.abc import Mapping
import matplotlib.pyplot as plt
from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values


class GraphExtractor:
    def __init__(self, graph_db) -> None:
        self.tuple_delimiter = "<|>"
        self.record_delimiter = "##"
        self.completion_delimiter = "<|COMPLETE|>"
        self.entity_types = ["organization", "person", "geo", "event"]

        self.graph_extraction_system = prompts.GRAPH_EXTRACTION_SYSTEM.format(
            entity_types=", ".join(self.entity_types),
            record_delimiter=self.record_delimiter,
            tuple_delimiter=self.tuple_delimiter,
            completion_delimiter=self.completion_delimiter,
        )

        self.graph_db = graph_db

        self.llm = LLMSession(system_message=self.graph_extraction_system,
                              model_name="gemini-1.5-pro-001")

        self.ingestion = IngestionSession()

    @observe()
    def __call__(self, text_input: str, max_extr_rounds: int = 5) -> None:
        input_prompt = self._construct_extractor_input(input_text=text_input)

        # response = self.llm.generate(client_query_string=input_prompt)
        # print(response)

        print("+++++ Init Graph Extraction +++++")

        init_extr_result = self.llm.generate_chat(
            client_query_string=input_prompt, temperature=0, top_p=0)
        print(f"Init result: {init_extr_result}")

        for round_i in range(max_extr_rounds):

            print(f"+++++ Contd. Graph Extraction round {round_i} +++++")

            round_response = self.llm.generate_chat(
                client_query_string=prompts.CONTINUE_PROMPT, temperature=0, top_p=0)
            init_extr_result += round_response or ""

            print(f"Round response: {round_response}")

            if round_i >= max_extr_rounds - 1:
                break

            completion_check = self.llm.generate_chat(
                client_query_string=prompts.LOOP_PROMPT, temperature=0, top_p=0)

            if "YES" not in completion_check:
                print(
                    f"+++++ Complete with completion check after round {round_i} +++++")
                break

        langfuse_context.flush()

        self._process_fskg(results={0: init_extr_result})

    def _construct_extractor_input(self, input_text: str) -> str:
        formatted_extraction_input = prompts.GRAPH_EXTRACTION_INPUT.format(
            entity_types=", ".join(self.entity_types),
            input_text=input_text,
        )
        return formatted_extraction_input

    def _process_fskg(
        self,
        results: dict[int, str],
        join_descriptions: bool = True,
    ) -> None:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
        Returns:
            - output - unipartite graph in graphML format
        """

        fskg = self.graph_db

        for source_doc_id, extracted_data in results.items():
            records = [r.strip()
                       for r in extracted_data.split(self.record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(self.tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = self._clean_str(record_attributes[1].upper())
                    entity_type = self._clean_str(record_attributes[2].upper())
                    entity_description = self._clean_str(record_attributes[3])

                    if fskg.node_exist(entity_name):
                        node = fskg.get_node(entity_name)
                        if join_descriptions:
                            # Combine descriptions, avoiding duplicates with a set
                            combined_descriptions = set(
                                [node.node_description] + [entity_description])
                            node.node_description = "\n".join(
                                combined_descriptions)
                        else:
                            if len(entity_description) > len(node.node_description):
                                node.node_description = entity_description
                        # Combine source IDs, avoiding duplicates with a set
                        combined_source_ids = set(
                            [node.document_id] + [str(source_doc_id)])
                        node.document_id = ", ".join(combined_source_ids)
                        node.node_type = entity_type if entity_type != "" else node.node_type
                    else:
                        node_data = data_model.NodeData(
                            node_uid=entity_name,
                            node_title=entity_name,
                            node_type=entity_type,
                            node_description=entity_description,
                            document_id=str(source_doc_id),
                            node_degree=0,
                        )

                        fskg.add_node(node_uid=entity_name,
                                      node_data=node_data)

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = self._clean_str(record_attributes[1].upper())
                    target = self._clean_str(record_attributes[2].upper())
                    edge_description = self._clean_str(record_attributes[3])
                    edge_source_id = self._clean_str(str(source_doc_id))
                    weight = (
                        float(record_attributes[-1])
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )

                    if not fskg.node_exist(source):
                        node_data = data_model.NodeData(
                            node_uid=source,
                            node_title=source,
                            node_type="",
                            node_description="",
                            document_id="",
                            node_degree=0,
                        )

                        fskg.add_node(node_uid=source,
                                      node_data=node_data)

                    if not fskg.node_exist(target):
                        node_data = data_model.NodeData(
                            node_uid=target,
                            node_title=target,
                            node_type="",
                            node_description="",
                            document_id="",
                            node_degree=0,
                        )

                        fskg.add_node(node_uid=target,
                                      node_data=node_data)

                    if fskg.edge_exist(source, target):
                        edge_data = fskg.get_edge(source, target)
                        if edge_data is not None:
                            # weight += edge_data["weight"]
                            if join_descriptions:
                                # Combine descriptions, avoiding duplicates with a set
                                combined_descriptions = set(
                                    [edge_data.description] + [edge_description])
                                edge_description = "\n".join(
                                    combined_descriptions)
                            # # Combine source IDs, avoiding duplicates with a set
                            # combined_source_ids = f"{[edge_data.document_id]} + {[str(source_doc_id)]}"
                            # edge_source_id = ", ".join(combined_source_ids)

                    edge_data = data_model.EdgeData(
                        source_uid=source,
                        target_uid=target,
                        description=edge_description,
                        document_id=edge_source_id,
                    )

                    fskg.add_edge(edge_data=edge_data)

        return

    def _process_results(
        self,
        results: dict[int, str],
        join_descriptions: bool = True,
    ) -> nx.Graph:
        """Parse the result string to create an undirected unipartite graph.

        Args:
            - results - dict of results from the extraction chain
        Returns:
            - output - unipartite graph in graphML format
        """
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip()
                       for r in extracted_data.split(self.record_delimiter)]

            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(self.tuple_delimiter)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = self._clean_str(record_attributes[1].upper())
                    entity_type = self._clean_str(record_attributes[2].upper())
                    entity_description = self._clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if join_descriptions:
                            # Combine descriptions, avoiding duplicates with a set
                            combined_descriptions = set(
                                self._unpack_descriptions(node) + [entity_description])
                            node["description"] = "\n".join(
                                combined_descriptions)
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        # Combine source IDs, avoiding duplicates with a set
                        combined_source_ids = set(
                            self._unpack_source_ids(node) + [str(source_doc_id)])
                        node["source_id"] = ", ".join(combined_source_ids)
                        node["entity_type"] = entity_type if entity_type != "" else node["entity_type"]
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = self._clean_str(record_attributes[1].upper())
                    target = self._clean_str(record_attributes[2].upper())
                    edge_description = self._clean_str(record_attributes[3])
                    edge_source_id = self._clean_str(str(source_doc_id))
                    weight = (
                        float(record_attributes[-1])
                        if isinstance(record_attributes[-1], numbers.Number)
                        else 1.0
                    )
                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )

                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if join_descriptions:
                                # Combine descriptions, avoiding duplicates with a set
                                combined_descriptions = set(
                                    self._unpack_descriptions(edge_data) + [edge_description])
                                edge_description = "\n".join(
                                    combined_descriptions)
                            # Combine source IDs, avoiding duplicates with a set
                            combined_source_ids = set(self._unpack_source_ids(
                                edge_data) + [str(source_doc_id)])
                            edge_source_id = ", ".join(combined_source_ids)
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )

        return graph

    def _unpack_descriptions(self, data: Mapping) -> list[str]:
        value = data.get("description", None)
        return [] if value is None else value.split("\n")

    def _unpack_source_ids(self, data: Mapping) -> list[str]:
        value = data.get("source_id", None)
        return [] if value is None else value.split(", ")

    def _clean_str(self, input: Any) -> str:
        """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
        # If we get non-string input, just give it back
        if not isinstance(input, str):
            return input

        result = html.unescape(input.strip())
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
        result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
        # Remove double quotes if they exist
        result = result.replace('"', '')
        return result

    def generate_comm_reports(self, kg: NoSQLKnowledgeGraph) -> None:

        llm = LLMSession(system_message=prompts.COMMUNITY_REPORT_SYSTEM,
                         model_name="gemini-1.5-pro-001")

        comms = kg.get_louvain_communities()

        for c in comms:
            comm_nodes = []
            comm_edges = []
            for n in c:
                node = kg.get_node(n)
                node_edges_to = [{"edge_source_entity": kg.get_edge(source_uid=node.node_uid,
                                                                    target_uid=e),
                                  "edge_target_entity": kg.get_edge(source_uid=node.node_uid,
                                                                    target_uid=e),
                                  "edge_description": kg.get_edge(source_uid=node.node_uid,
                                                                  target_uid=e)} for e in node.edges_to]

                node_edges_from = [{"edge_source_entity": kg.get_edge(source_uid=e,
                                                                      target_uid=node.node_uid),
                                   "edge_target_entity": kg.get_edge(source_uid=e,
                                                                     target_uid=node.node_uid),
                                    "edge_description": kg.get_edge(source_uid=e,
                                                                    target_uid=node.node_uid)} for e in node.edges_from]

                node_data = {"entity_id": node.node_title,
                             "entity_type": node.node_type,
                             "entity_description": node.node_description}

                comm_nodes.append(node_data)
                comm_edges.extend(node_edges_to)
                comm_edges.extend(node_edges_from)

            response_schema = {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "string"
                    },
                    "rating": {
                        "type": "int"
                    },
                    "rating_explanation": {
                        "type": "string"
                    },
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string"
                                },
                                "explanation": {
                                    "type": "string"
                                }
                            },
                            # Ensure both fields are present in each finding
                            "required": ["summary", "explanation"]
                        }
                    }
                },
                # List required fields at the top level
                "required": ["title", "summary", "rating", "rating_explanation", "findings"]
            }

            comm_report = llm.generate(client_query_string=prompts.COMMUNITY_REPORT_QUERY.format(
                entities=comm_nodes,
                relationships=comm_edges,
                response_mime_type="application/json",
                response_schema=response_schema
            ))

            comm_report_dict = self.llm.parse_json_response(comm_report)

            if comm_report_dict == {}:
                comm_data = data_model.CommunityData(title=str(c),
                                    summary="",
                                    rating=0,
                                    rating_explanation="",
                                    findings=[{}],
                                    community_nodes=c)
            else:
                comm_data = data_model.CommunityData(title=comm_report_dict["title"],
                                                    summary=comm_report_dict["summary"],
                                                    rating=comm_report_dict["rating"],
                                                    rating_explanation=comm_report_dict["rating_explanation"],
                                                    findings=comm_report_dict["findings"],
                                                    community_nodes=c)
                
                fskg.store_community(community=comm_data)

        return None
 
    def update_node_embeddings(self) -> None: 

        node_embeddings = self.graph_db.get_node2vec_embeddings()

        for node_uid in node_embeddings.nodes:
            # Get the embedding for the current node
            index = node_embeddings.nodes.index(node_uid)
            embedding = node_embeddings.embeddings[index].tolist() 
            
            try:
                # Fetch the existing node data
                node_data = self.graph_db.get_node(node_uid)
                
                # Update the embedding
                node_data.embedding = embedding

                # Update the node in Firestore
                self.graph_db.update_node(node_uid, node_data)

            except KeyError:
                # Handle the case where the node doesn't exist 
                print(f"Warning: Node '{node_uid}' not found in Firestore. Skipping embedding update.")

        return None


if __name__ == "__main__":

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

    ingestion = IngestionSession()
    extractor = GraphExtractor(graph_db=fskg)

    # document_string = ingestion(
    #     new_file_name="./pdf_articles/Winners of Future Hamburg Award 2023 announced _ Hamburg News.pdf", ingest_local_file=True
    # )
    # extracted_graph = extractor(text_input=document_string, max_extr_rounds=1)
    
    # document_string = ingestion(
    #     new_file_name="./pdf_articles/Physicist Narges Mohammadi awarded Nobe... for human-rights work – Physics World.pdf", ingest_local_file=True
    # )
    # extracted_graph = extractor(text_input=document_string, max_extr_rounds=1)

    # fskg.visualize_graph("visualized.png")

    # extractor.generate_comm_reports(kg=fskg)

    extractor.update_node_embeddings()

    print("Hello World!")
