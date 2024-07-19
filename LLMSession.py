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

from optparse import Option
from dotenv import dotenv_values
import base64
from google.cloud import aiplatform
from matplotlib.pyplot import hist
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

import prompts
import sample_txt

from typing import List, Optional


class LLMSession:
    def __init__(self, system_message: str, model_name: str):
        self.model_name = model_name
        self.system_message = system_message
        self.secrets = dotenv_values(".env")
        self.model = GenerativeModel(
            self.model_name, system_instruction=[system_message])
        self.model_chat = self.model.start_chat()

    def generate(
        self,
        client_query_string: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
        top_p: float = 0.5
    ) -> str:

        response = self.model.generate_content(
            [client_query_string],
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            stream=False
        )
        return response.text  # type: ignore

    def generate_chat(self,
                      client_query_string: str,
                      max_output_tokens: int = 8192,
                      temperature: float = 0.2,
                      top_p: float = 0.5) -> str:

        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p
        }

        response = self.model_chat.send_message(
            client_query_string,
            stream=False,
            generation_config=generation_config)
        return response.text  # type: ignore


if __name__ == "__main__":
    prompt = "Which is the city with the most bridges?"

    DEFAULT_TUPLE_DELIMITER = "<|>"
    DEFAULT_RECORD_DELIMITER = "##"
    DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
    DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

    graph_extraction_system = prompts.GRAPH_EXTRACTION_SYSTEM.format(
        entity_types=", ".join(DEFAULT_ENTITY_TYPES),
        record_delimiter=DEFAULT_RECORD_DELIMITER,
        tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
    )

    graph_extraction_input = prompts.GRAPH_EXTRACTION_INPUT.format(
        entity_types=", ".join(DEFAULT_ENTITY_TYPES),
        input_text=sample_txt.sample_text,
    )

    llm = LLMSession(
        system_message=graph_extraction_system,
        model_name="gemini-1.5-pro-001"
    )
    response = llm.generate(client_query_string=graph_extraction_input, )
    print(response)
