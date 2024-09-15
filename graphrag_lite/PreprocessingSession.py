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


from numpy import gradient
from graphrag_lite.IngestionSession import IngestionSession
from graph2nosql.graph2nosql.graph2nosql import NoSQLKnowledgeGraph
from dotenv import dotenv_values

from io import BytesIO
import PyPDF2


class PreprocessingSession:
    def __init__(self, graph_db: NoSQLKnowledgeGraph) -> None:
        self.secrets = dotenv_values(".env")
        self.graph_db = graph_db

    def __call__(self, new_file_name: str,
                 max_pages_per_file: int,
                 file_to_ingest=None,
                 ingest_local_file: bool = False,
                 ingest_pdf: bool = True) -> None:

        ingestion = IngestionSession(graph_db=self.graph_db)

        pdf_file = BytesIO(file_to_ingest)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        ingest_pdf = ingest_pdf

        # print number of pages
        print(f"Total Pages: {num_pages}")

        # check if PDF file exceeds the page limit
        if num_pages > max_pages_per_file:
            output_file_index = 1
            output_file_name = f"{
                new_file_name[:-4]}-part{output_file_index}.pdf"
            pdf_writer = PyPDF2.PdfWriter()
            tmp = BytesIO()

            for page_num in range(num_pages):
                pdf_writer.add_page(pdf_reader.pages[page_num])

                if page_num % max_pages_per_file == max_pages_per_file - 1:
                    pdf_writer.write(tmp)
                    output_file_bytes = tmp.getvalue()
                    ingestion(new_file_name=output_file_name, file_to_ingest=output_file_bytes,
                              ingest_local_file=False)

                    output_file_index += 1
                    output_file_name = f"{
                        new_file_name[:-4]}-part{output_file_index}.pdf"
                    pdf_writer = PyPDF2.PdfWriter()
                    tmp = BytesIO()

            # Write any remaining pages to the last output file
            if page_num % max_pages_per_file != max_pages_per_file - 1:
                pdf_writer.write(tmp)
                output_file_bytes = tmp.getvalue()
                ingestion(new_file_name=output_file_name, file_to_ingest=output_file_bytes,
                          ingest_local_file=False)

            print("Splitting & Ingestion completed.")

        else:
            new_file_name = f"{new_file_name[:-4]}-part0.pdf"
            ingestion(new_file_name=new_file_name, file_to_ingest=file_to_ingest,
                      ingest_local_file=False)
            print("PDF file has", num_pages,
                  "pages or less, no splitting was needed. Ingestion completed.")

        return None


if __name__ == "__main__":
    print("Hello world")
