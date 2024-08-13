
from graphrag.LLMSession import LLMSession
import graphrag.prompts as prompts
from graphrag.GraphExtractor import GraphExtractor

from nosql_kg.graph2nosql import NoSQLKnowledgeGraph
from nosql_kg.firestore_kg import FirestoreKG
from nosql_kg import data_model

from cgitb import text
from html import entities
from typing import Any

from httpx import get
import networkx as nx
from google.cloud.firestore_v1.vector import Vector

from collections.abc import Mapping
import matplotlib.pyplot as plt
from langfuse.decorators import observe, langfuse_context
from dotenv import dotenv_values

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

    # ingestion = IngestionSession()
    extractor = GraphExtractor(graph_db=fskg)

    extractor.generate_comm_reports(kg=fskg)

    extractor.update_node_embeddings()

    print("Hello World!")