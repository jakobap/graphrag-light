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
# from google.cloud import storage
from google.cloud import aiplatform_v1
import google.auth

# import firebase_admin
# from firebase_admin import firestore

# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter


class IngestionSession:
    def __init__(self, chunk_size=1000, chunk_overlap=50):
        self.secrets = dotenv_values(".env")

        # if not firebase_admin._apps:
        #     credentials = firebase_admin.credentials.Certificate(
        #         self.secrets["GCP_CREDENTIAL_FILE"]
        #     )
        #     app = firebase_admin.initialize_app(credentials)

        self.credentials, _ = google.auth.load_credentials_from_file(
            self.secrets["GCP_CREDENTIAL_FILE"]
        )
        self.project_id = str(self.secrets["GCP_PROJECT_ID"])

        self.docai_processor_id = str(self.secrets['DOCUMENT_AI_PROCESSOR_ID'])
        self.docai_processor_version = str(
            self.secrets["DOCUMENT_AI_PROCESSOR_VERSION"])
        self.gcp_multiregion = str(self.secrets["GCP_MULTIREGION"])
        # self.embedding_session = EmbeddingSession()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, new_file_name: str,
                 file_to_ingest=None,
                 ingest_local_file: bool = False,
                 data_to_ingest=None) -> str:

        # print("+++++ Upload raw PDF... +++++")
        # self._store_raw_upload(
        #     new_file_name=new_file_name, file_to_ingest=file_to_ingest, ingest_local_file=ingest_local_file)

        print("+++++ Document OCR... +++++")
        document_string = self._ocr_pdf(
            processor_id=self.docai_processor_id,
            processor_version=self.docai_processor_version,
            location=self.gcp_multiregion,
            file_path=new_file_name,
            file_to_ingest=file_to_ingest,
            ingest_local_file=ingest_local_file)
        
        # print(document_string)

        # print("+++++ Chunking Document... +++++")
        # list_of_chunks = self._chunk_doc(stringified_doc=document_string,
        #                                 file_name=new_file_name,
        #                                 chunk_size=self.chunk_size,
        #                                 chunk_overlap=self.chunk_overlap)

        # print (list_of_chunks)
        # print("+++++ Store Embeddings & Document Identifiers in Firestore... +++++")
        # self._firestore_index_embeddings(list_of_chunks)

        # print("+++++ Generating Document Embeddings... +++++")
        # embeddings_to_ingest = self._chunk_to_index_input(list_of_chunks)

        # print("+++++ Updating Vector Search Index... +++++")
        # self._vector_index_streaming_upsert(embeddings_to_ingest)

        print("+++++ Ingestion Done. +++++")

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

    # def _chunk_doc(
    #     self, stringified_doc: str, file_name, chunk_size, chunk_overlap
    # ) -> list:
    #     # method to chunk a given doc
    #     doc = Document(page_content=stringified_doc)
    #     doc.metadata["document_name"] = file_name.split("/")[-1]

    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #         separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    #     )

    #     doc_splits = text_splitter.split_documents([doc])

    #     for idx, split in enumerate(doc_splits):
    #         split.metadata["chunk_identifier"] = (
    #             file_name.split("/")[-1].split(".pdf")[0] + "-chunk" + str(idx)
    #         )

    #     return doc_splits

    # def _generate_embedding(self, text_to_embed):
    #     # method to generate embedding for a given text
    #     embedding = self.embedding_session.get_vertex_embedding(
    #         text_to_embed=text_to_embed
    #     )
    #     return embedding

    # def _chunk_to_index_input(self, list_of_chunks: list) -> list:
    #     # turning chunk strings into jsons ready to be indexed by vector search

    #     # generate embeddings & merge with chunk embedding identifier
    #     embedded_docs = [
    #         json.dumps(
    #             {
    #                 "id": d.metadata["chunk_identifier"],
    #                 "embedding": self._generate_embedding(d.page_content),
    #             }
    #         )
    #         + "\n"
    #         for d in list_of_chunks
    #     ]

    #     return embedded_docs

    # def _store_raw_upload(
    #     self, new_file_name: str, file_to_ingest, ingest_local_file: bool = False
    # ) -> None:
    #     # store raw uploaded pdf in gcs
    #     storage_client = storage.Client(credentials=self.credentials)
    #     bucket = storage_client.bucket(self.secrets["RAW_PDFS_BUCKET_NAME"])
    #     print(new_file_name)

    #     # string = "This is a string containing a substring."
    #     # substring = "substring"

    #     if ".pdf" in new_file_name:
    #         blob = bucket.blob("documents/raw_uploaded/" +
    #                            new_file_name.split("/")[-1])
    #     else:
    #         new_file_name = new_file_name + ".pdf"
    #         blob = bucket.blob("documents/raw_uploaded/" +
    #                            new_file_name.split("/")[-1])

    #     print(ingest_local_file)

    #     if ingest_local_file:
    #         blob.upload_from_filename(new_file_name)
    #     else:
    #         file_contents = io.BytesIO(file_to_ingest)
    #         blob.upload_from_file(file_contents)

    #     return None

    # def _store_json_upload(
    #     self, new_file_name: str, file_to_ingest, ingest_local_file: bool = False
    # ) -> None:
    #     # store json file in gcs
    #     storage_client = storage.Client(credentials=self.credentials)
    #     bucket = storage_client.bucket(self.secrets["RAW_PDFS_BUCKET_NAME"])

    #     print(new_file_name)

    #     if ".json" in new_file_name:
    #         blob = bucket.blob("documents/raw_uploaded/" +
    #                            new_file_name.split("/")[-1])
    #     else:
    #         new_file_name = new_file_name + ".json"
    #         blob = bucket.blob("documents/raw_uploaded/" +
    #                            new_file_name.split("/")[-1])

    #     print(ingest_local_file)

    #     file_contents = io.BytesIO(file_to_ingest)
    #     blob.upload_from_file(file_contents)

    #     return None

    # def _firestore_index_embeddings(self, doc_splits: list) -> None:
    #     # upload embeddings to firestore

    #     if not firebase_admin._apps:
    #         credentials = firebase_admin.credentials.Certificate(
    #             self.secrets["GCP_CREDENTIAL_FILE"]
    #         )
    #         app = firebase_admin.initialize_app(credentials)

    #     db = firestore.Client(
    #         project=self.secrets["GCP_PROJECT_ID"], credentials=self.credentials, database=self.secrets["FIRESTORE_DATABASE_ID"])

    #     for split in doc_splits:
    #         data = {
    #             "id": split.metadata["chunk_identifier"],
    #             "document_name": split.metadata["document_name"],
    #             "page_content": split.page_content,
    #         }

    #         # Add a new doc in collection with embedding, doc name & chunk identifier
    #         db.collection(self.secrets["FIRESTORE_COLLECTION_NAME"]).document(
    #             str(split.metadata["chunk_identifier"])
    #         ).set(data)

    #     print(f"Added {[split.metadata['chunk_identifier']
    #           for split in doc_splits]}")

    #     return None

    # def _vector_index_streaming_upsert(self, upsert_datapoints: list) -> None:
    #     # method to upsert embeddings to vector search index

    #     index_client = aiplatform_v1.IndexServiceClient(credentials=self.credentials, client_options=dict(
    #         api_endpoint=f"{
    #             self.secrets['GCP_REGION']}-aiplatform.googleapis.com"
    #     ))

    #     index_name = f"projects/{self.secrets['GCP_PROJECT_NUMBER']}/locations/{
    #         self.secrets['GCP_REGION']}/indexes/{self.secrets['VECTOR_SEARCH_INDEX_ID']}"
    #     insert_datapoints_payload = []
    #     for dp in upsert_datapoints:
    #         dp_dict = json.loads(dp)
    #         insert_datapoints_payload.append(
    #             aiplatform_v1.IndexDatapoint(
    #                 datapoint_id=dp_dict["id"],
    #                 feature_vector=dp_dict["embedding"],
    #                 restricts=[],
    #             )
    #         )

    #     upsert_request = aiplatform_v1.UpsertDatapointsRequest(
    #         index=index_name, datapoints=insert_datapoints_payload)
    #     print('test3')
    #     index_client.upsert_datapoints(request=upsert_request)

    #     return None


if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)

    ingestion = IngestionSession()

    ingestion(
        new_file_name="./pdf_articles/Winners of Future Hamburg Award 2023 announced _ Hamburg News.pdf", ingest_local_file=True
    )

    print("Hello World!")
