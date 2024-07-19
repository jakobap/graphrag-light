from cgitb import text
from typing import Any
from IngestionSession import IngestionSession
from LLMSession import LLMSession

import prompts

import networkx as nx

import re
import numbers
import html
from collections.abc import Mapping
import matplotlib.pyplot as plt


# SAMPLE_RESPONSE = """
#         ("entity"<|>"Colipi"<|>"organization"<|>"Colipi is a Hamburg-based biotech company that transforms renewable carbon from organic by-products and carbon dioxide into microbial Climate Oil, an alternative to palm oil usable in various fields.")##
#         ("entity"<|>"Make My Day"<|>"organization"<|>"Make My Day is an Israeli start-up that provides smart mobility solutions to help companies transition to all-green electric vehicles, including fleet management, route planning, and battery consumption prediction.")##
#         ("entity"<|>"AraBat"<|>"organization"<|>"AraBat is an Italian start-up that retrieves metals from batteries using agri-food waste, promoting resource efficiency and the circular economy.")##
#         ("entity"<|>"Hamburg Marketing"<|>"organization"<|>"Hamburg Marketing is an organization that presented the Future Hamburg Award during Plug and Play's Expo in the Factory Hammerbrooklyn.")##
#         ("entity"<|>"Plug and Play"<|>"organization"<|>"Plug and Play is an organization with a network of over 60,000 start-ups and 500 companies, supporting the Future Hamburg Award and shaping Hamburg's innovation ecosystem.")##
#         ("entity"<|>"Startup Unit Hamburg"<|>"organization"<|>"Startup Unit Hamburg is part of Hamburg Invest and supports start-ups by mediating contacts to investors and forging networks.")##
#         ("entity"<|>"Hamburg Invest"<|>"organization"<|>"Hamburg Invest is a business development agency that initiated the Future Hamburg Award in 2018.")##
#         ("entity"<|>"Future Hamburg Award"<|>"event"<|>"The Future Hamburg Award is a biannual event that honors forward-looking start-ups excelling in innovation, impact, and growth.")##
#         ("entity"<|>"Plug and Play's Expo"<|>"event"<|>"Plug and Play's Expo is an event held in the Factory Hammerbrooklyn where the Future Hamburg Award was presented.")##
#         ("entity"<|>"Sallar Faridi"<|>"person"<|>"Sallar Faridi is the Senior Director of Plug and Play, supporting the Future Hamburg Award and promoting Hamburg's innovation ecosystem.")##
#         ("entity"<|>"Hamburg"<|>"geo"<|>"Hamburg is a city in Germany, hosting the Future Hamburg Award and fostering a thriving start-up ecosystem.")##
#         ("entity"<|>"Silicon Valley, California"<|>"geo"<|>"Silicon Valley is a location in California, home to the Plug and Play Tech Center headquarters, where Colipi's prize includes a one-month accelerator program.")##
#         ("relationship"<|>"Colipi"<|>"Future Hamburg Award"<|>"Colipi won the top Future Hamburg Award in 2023."<|>9)##
#         ("relationship"<|>"Make My Day"<|>"Future Hamburg Award"<|>"Make My Day came second in the Future Hamburg Award in 2023."<|>8)##
#         ("relationship"<|>"AraBat"<|>"Future Hamburg Award"<|>"AraBat came third in the Future Hamburg Award in 2023."<|>7)##
#         ("relationship"<|>"Hamburg Marketing"<|>"Future Hamburg Award"<|>"Hamburg Marketing presented the Future Hamburg Award."<|>8)##
#         ("relationship"<|>"Plug and Play"<|>"Future Hamburg Award"<|>"Plug and Play is the main partner of the Future Hamburg Award."<|>9)##
#         ("relationship"<|>"Startup Unit Hamburg"<|>"Colipi"<|>"Startup Unit Hamburg invited Colipi to take part in its scheme."<|>7)##
#         ("relationship"<|>"Startup Unit Hamburg"<|>"Make My Day"<|>"Startup Unit Hamburg invited Make My Day to take part in its scheme."<|>7)##
#         ("relationship"<|>"Startup Unit Hamburg"<|>"AraBat"<|>"Startup Unit Hamburg invited AraBat to take part in its scheme."<|>7)##
#         ("relationship"<|>"Colipi"<|>"Silicon Valley, California"<|>"Colipi's prize includes a one-month accelerator at the Plug and Play Tech Center headquarters in Silicon Valley."<|>8)##
#         ("relationship"<|>"Hamburg Invest"<|>"Future Hamburg Award"<|>"Hamburg Invest initiated the Future Hamburg Award."<|>8)##
#         ("relationship"<|>"Future Hamburg Award"<|>"Hamburg"<|>"The Future Hamburg Award is presented in Hamburg."<|>9)##
#         ("relationship"<|>"Plug and Play's Expo"<|>"Hamburg"<|>"Plug and Play's Expo, where the Future Hamburg Award was presented, took place in Hamburg."<|>8)<|COMPLETE|>
#         """


class GraphExtractor:
    def __init__(self) -> None:
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

        self.llm = LLMSession(system_message=self.graph_extraction_system,
                              model_name="gemini-1.5-pro-001")

        self.ingestion = IngestionSession()

    def __call__(self, text_input: str, max_extr_rounds:int=5) -> nx.Graph:
        input_prompt = self._construct_extractor_input(input_text=text_input)

        # response = self.llm.generate(client_query_string=input_prompt)
        # print(response)

        print("+++++ Init Graph Extraction +++++")

        init_extr_result = self.llm.generate_chat(client_query_string=input_prompt)
        print(f"Init result: {init_extr_result}")

        for round_i in range(max_extr_rounds):

            print(f"+++++ Contd. Graph Extraction round {round_i} +++++")

            round_response = self.llm.generate_chat(client_query_string=prompts.CONTINUE_PROMPT)
            init_extr_result += round_response or ""

            print(f"Round response: {round_response}")

            if round_i >= max_extr_rounds - 1:
                break

            completion_check = self.llm.generate_chat(client_query_string=prompts.LOOP_PROMPT)

            if completion_check != "YES":
                print(f"+++++ Complete with completion check after round {round_i} +++++")
                break

        return self._process_results(results={0: init_extr_result})

    def _construct_extractor_input(self, input_text: str) -> str:
        formatted_extraction_input = prompts.GRAPH_EXTRACTION_INPUT.format(
            entity_types=", ".join(self.entity_types),
            input_text=input_text,
        )
        return formatted_extraction_input

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
        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    def visualize_graph(self, graph: nx.Graph) -> None:
        """Visualizes the provided networkx graph using matplotlib.

        Args:
            graph (nx.Graph): The graph to visualize.
        """

        # Create a larger figure for better visualization
        plt.figure(figsize=(12, 12))

        # Use a spring layout for a more visually appealing arrangement
        pos = nx.spring_layout(graph, k=0.3, iterations=50)

        # Draw nodes with different colors based on entity type
        entity_types = set(data["type"] for _, data in graph.nodes(data=True))
        color_map = plt.cm.get_cmap("tab10", len(entity_types))
        for i, entity_type in enumerate(entity_types):
            nodes = [n for n, d in graph.nodes(
                data=True) if d["type"] == entity_type]
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_color=[color_map(i)],  # type: ignore
                label=entity_type,
                node_size=[10 + 50 * graph.degree(n)
                           for n in nodes]  # type: ignore
            )

        # Draw edges with labels
        nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.5)
        # edge_labels = nx.get_edge_attributes(graph, "description")
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

        # Add node labels with descriptions
        node_labels = {
            node: node
            for node, data in graph.nodes(data=True)
        }
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)

        plt.title("Extracted Knowledge Graph")
        plt.axis("off")  # Turn off the axis

        # Add a legend for node colors
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=entity_type,
                   markersize=10, markerfacecolor=color_map(i)) for i, entity_type in enumerate(entity_types)])
        plt.show()
        return None


if __name__ == "__main__":
    ingestion = IngestionSession()
    extractor = GraphExtractor()

    document_string = ingestion(
        new_file_name="./pdf_articles/Winners of Future Hamburg Award 2023 announced _ Hamburg News.pdf", ingest_local_file=True
    )

    extracted_graph = extractor(text_input=document_string)
    extractor.visualize_graph(extracted_graph)

    print("Hello World!")
