# Copyright 2024 Google

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from rsc.EmbeddingSession import EmbeddingSession

import os
import json
from dotenv import dotenv_values
import io

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
from google.cloud import storage
from google.cloud import aiplatform_v1
import google.auth

# import firebase_admin
# from firebase_admin import firestore

# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from graphrag_lite.GraphExtractor import GraphExtractor, GCPGraphExtractor
from graph2nosql.graph2nosql.graph2nosql import NoSQLKnowledgeGraph
from graph2nosql.databases.firestore_kg import FirestoreKG


class IngestionSession:
    def __init__(self, graph_db: NoSQLKnowledgeGraph):
        self.secrets = dotenv_values(".env")

        self.credentials, _ = google.auth.load_credentials_from_file(
            self.secrets["GCP_CREDENTIAL_FILE"]
        )
        self.project_id = str(self.secrets["GCP_PROJECT_ID"])

        self.docai_processor_id = str(self.secrets['DOCUMENT_AI_PROCESSOR_ID'])
        self.docai_processor_version = str(
            self.secrets["DOCUMENT_AI_PROCESSOR_VERSION"])
        self.gcp_multiregion = str(self.secrets["GCP_MULTIREGION"])

        self.graph_db = graph_db

    def __call__(self, new_file_name: str,
                 file_to_ingest=None,
                 ingest_local_file: bool = False,
                 data_to_ingest=None) -> str:

        print("+++++ Upload raw PDF... +++++")
        self._store_raw_upload(
            new_file_name=new_file_name, file_to_ingest=file_to_ingest, ingest_local_file=ingest_local_file)

        print("+++++ Document OCR... +++++")
        document_string = self._ocr_pdf(
            processor_id=self.docai_processor_id,
            processor_version=self.docai_processor_version,
            location=self.gcp_multiregion,
            file_path=new_file_name,
            file_to_ingest=file_to_ingest,
            ingest_local_file=ingest_local_file)
        
        print("+++++ Extracting Graph Data +++++")
        extractor = GCPGraphExtractor(graph_db=self.graph_db)
        extracted_graph = extractor(text_input=document_string, max_extr_rounds=1)
        # extractor.generate_comm_reports(kg=self.graph_db)
        extractor.async_generate_comm_reports(kg=self.graph_db)
        extractor.update_node_embeddings()
        self.graph_db.visualize_graph(filename="./visualize_kg.png")

        print("+++++ Graph Ingestion Done. +++++")
        return document_string

    def _process_document(
        self,
        location: str,
        processor_id: str,
        processor_version: str,
        file_path: str,
        mime_type: str,
        process_options=None,
        file_to_ingest=None,
        ingest_local_file: bool = False,
    ) -> documentai.Document:

        client = documentai.DocumentProcessorServiceClient(
            credentials=self.credentials,
            client_options=ClientOptions(
                api_endpoint=f"{location}-documentai.googleapis.com"
            ),
        )

        # file_path = file_path.getvalue()

        name = client.processor_version_path(
            self.project_id, location, processor_id, processor_version
        )

        if ingest_local_file:
            # Read the file into memory
            with open(file_path, "rb") as image:
                image_content = image.read()
        else:
            image_content = file_to_ingest

        # Configure the process request
        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(
                content=image_content, mime_type=mime_type
            ),
            process_options=process_options,
        )

        result = client.process_document(request=request)

        return result.document

    def _ocr_pdf(self,
                 processor_id: str,
                 processor_version: str,
                 file_path: str,
                 location: str,
                 mime_type: str = "application/pdf",
                 file_to_ingest=None,
                 ingest_local_file: bool = False) -> str:

        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
                enable_symbol=True,
                premium_features=documentai.OcrConfig.PremiumFeatures(
                    compute_style_info=True,
                    enable_math_ocr=False,
                    enable_selection_mark_detection=True,
                ),
            )
        )

        # Online processing request to Document AI
        document = self._process_document(
            location=location,
            processor_id=processor_id,
            processor_version=processor_version,
            file_path=file_path,
            mime_type=mime_type,
            process_options=process_options,
            file_to_ingest=file_to_ingest,
            ingest_local_file=ingest_local_file,
        )

        return document.text

    def _store_raw_upload(
        self, new_file_name: str, file_to_ingest, ingest_local_file: bool = False
    ) -> None:
        # store raw uploaded pdf in gcs
        storage_client = storage.Client(credentials=self.credentials)
        bucket = storage_client.bucket(self.secrets["RAW_PDFS_BUCKET_NAME"])
        print(new_file_name)

        if ".pdf" in new_file_name:
            blob = bucket.blob("documents/raw_uploaded/" +
                               new_file_name.split("/")[-1])
        else:
            new_file_name = new_file_name + ".pdf"
            blob = bucket.blob("documents/raw_uploaded/" +
                               new_file_name.split("/")[-1])

        print(ingest_local_file)

        if ingest_local_file:
            blob.upload_from_filename(new_file_name)
        else:
            file_contents = io.BytesIO(file_to_ingest)
            blob.upload_from_file(file_contents)

        return None


if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)

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

    ingestion = IngestionSession(graph_db=fskg)

    ingestion(
        new_file_name="./pdf_articles/Winners of Future Hamburg Award 2023 announced _ Hamburg News.pdf", ingest_local_file=True
    )

    print("Hello World!")
