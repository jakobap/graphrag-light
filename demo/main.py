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

import math
import datetime
from dotenv import dotenv_values

import streamlit as st
import pandas as pd

from google.cloud import storage
import google.auth

from graph2nosql.databases.firestore_kg import FirestoreKG

from graphrag_lite.KGraphQuery import GlobalQueryGCP
from graphrag_lite.IngestionSession import IngestionSession
from graphrag_lite.PreprocessingSession import PreprocessingSession

secrets = dotenv_values(".env")
credentials, _ = google.auth.load_credentials_from_file(secrets['GCP_CREDENTIAL_FILE'])

fskg = FirestoreKG(
        gcp_project_id=str(secrets["GCP_PROJECT_ID"]),
        gcp_credential_file=str(secrets["GCP_CREDENTIAL_FILE"]),
        firestore_db_id=str(secrets["FIRESTORE_DB_ID"]),
        node_collection_id=str(secrets["NODE_COLL_ID"]),
        edges_collection_id=str(secrets["EDGES_COLL_ID"]),
        community_collection_id=str(secrets["COMM_COLL_ID"]),
    )

class DocPreview:
    def __init__(self, list_of_docs: list):
        self.list_of_docs = list_of_docs
        self.row_count = math.ceil(len(self.list_of_docs) / 3) 
        self.doc_count = len(self.list_of_docs)
        self.doc_index = 0

    def render(self):
        with st.container(border=True):

            for _ in range(self.row_count):
                
                col1, col2, col3 = st.columns(3)

                with col1:
                    self._render_doc_col()

                with col2:
                    self._render_doc_col()

                with col3:
                    self._render_doc_col()

    def _render_doc_col(self):
        if self.doc_index < self.doc_count:
            self._render_doc_item(doc_name_and_url=self.list_of_docs[self.doc_index])
            self.doc_index += 1
        else:
            pass

    def _render_doc_item(self, doc_name_and_url):
        doc_name = doc_name_and_url[0]
        doc_link = doc_name_and_url[1]

        st.write(f"[{doc_name}](%s)" % doc_link)
        st.image("./img/PDF_file_icon.png", width=50)
        # st.button('delete', key=f'delete_{doc_name}', on_click=delete_file, args=[doc_name])

def main(client_query:str, model_name: str) -> None:  

    with st.spinner('Processing... This might take a minute or two.'):

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

        global_query = GlobalQueryGCP(secrets, fskg)
        final_response = global_query(user_query=client_query)

        # df = pd.DataFrame({"Sources": sources})
        # st.dataframe(df)

    st.write(final_response)

    return None


def extract_filename_from_url(text):
  """
  Extracts the string between the last `/` and `.pdf` from a text.

  Args:
    text: The text to extract from.

  Returns:
    The extracted string, or None if not found.
  """
  # Find the last occurrence of `&%`
  last_percent = text.rfind("/")

  # Find the next occurrence of `.pdf`
  next_pdf = text.find(".pdf", last_percent)

  # Extract the substring if both patterns were found
  if last_percent != -1 and next_pdf != -1:
    return text[last_percent + 1:next_pdf]
  else:
    return text 


def get_current_files(bucket_name, secrets=secrets, credentials=credentials) -> list:
    bucket = storage.Client(project=secrets['GCP_PROJECT_ID'], credentials=credentials).bucket(bucket_name)
    print(list(bucket.list_blobs()))
    files_with_links = [(extract_filename_from_url(blob.name), blob.generate_signed_url(expiration=datetime.timedelta(minutes=10))) for blob in bucket.list_blobs() if ".pdf" in str(blob.name)] 
       
    return files_with_links


def upload_new_file(new_file:bytes, new_file_name:str) -> None:
    preprocessing = PreprocessingSession(graph_db=fskg)
    preprocessing(new_file_name=new_file_name,
                  file_to_ingest=new_file, 
                  ingest_local_file=False,
                  max_pages_per_file=15,
                  ingest_pdf=True)
    return None


# def delete_file(document_name:str) -> None:
#     print(f'deletion {document_name}')
    
#     deletion = DeletionSession()

#     deletion(document_name=document_name)

#     return None


st.title('GenAI Demo')

st.header('Ingest data to your knowledge base.')

st.markdown('**These pdf files are currently in your knowledge base.**')

current_files = get_current_files(bucket_name=secrets["RAW_PDFS_BUCKET_NAME"])
df = pd.DataFrame(current_files)
st.dataframe(df)

DocPreview(list_of_docs=current_files).render()

st.header('Knowledge Graph Overview')
fskg.visualize_graph(filename="visualize_kg.png")
st.image("./visualize_kg.png")

st.markdown('**Upload a new PDF file for your knowledge base.**')

with st.form("file_upload_form"):
    uploaded_file = st.file_uploader("Choose a file")
    
    button = st.form_submit_button('Upload', help=None, on_click=None, args=None, kwargs=None, type="primary", disabled=False, use_container_width=False)

    if button and uploaded_file is not None:
        upladed_file_name = uploaded_file.name
        uploaded_file_bytes = uploaded_file.getvalue()
        upload_new_file(new_file=uploaded_file_bytes, new_file_name=upladed_file_name)

st.header('Ask a question.')
st.caption('This Q&A demo will answer to questions only related to your knowledge base documents.')

with st.form("transcript_submission_form"):

    client_query = st.text_input("Question:")

    # model_name = str(st.selectbox('Which Model would you like to ask?', ('gemini-1.5-flash','gemini-1.5-pro'),placeholder='gemini-1.5-pro'))

    button = st.form_submit_button('Ask', help=None, on_click=None, args=None, kwargs=None, type="primary", disabled=False, use_container_width=False)

    if button:
        main(client_query=client_query, model_name="model_name")