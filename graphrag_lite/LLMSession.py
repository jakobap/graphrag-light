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

import graphrag_lite.prompts as prompts

from optparse import Option
from dotenv import dotenv_values
import base64
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason, GenerationConfig, SafetySetting
import vertexai.preview.generative_models as generative_models
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import json


from typing import List, Optional, Dict, Any

from langfuse.decorators import observe, langfuse_context
from langfuse.model import ModelUsage


class LLMSession:
    def __init__(self, system_message: str, model_name: str):
        self.model_name = model_name
        self.system_message = system_message
        self.secrets = dotenv_values(".env")
        vertexai.init(project=self.secrets["GCP_PROJECT_ID"], location=self.secrets["GCP_REGION"])
        self.model = GenerativeModel(
            self.model_name, system_instruction=[system_message])
        self.model_chat = self.model.start_chat()

        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            )
        ]

    @observe(as_type="generation")
    def generate(
        self,
        client_query_string: str,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
        top_p: float = 0.5,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> str:

        response = self.model.generate_content(
            [client_query_string],
            generation_config=GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            ),
            safety_settings=self.safety_settings,
            stream=False
        )

        response_text = response.text  # type: ignore

        self._langfuse_observation_meta(observation_name="Text Generate",
                                        query_string=client_query_string,
                                        vertex_model_response=response)

        return response_text

    @observe(as_type="generation")
    def generate_chat(self,
                      client_query_string: str,
                      max_output_tokens: int = 8192,
                      temperature: float = 0.2,
                      top_p: float = 0.5,
                      response_mime_type: Optional[str] = None,
                      response_schema: Optional[Dict[str, Any]] = None) -> str:

        generation_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            response_mime_type=response_mime_type,
            response_schema=response_schema
        )

        response = self.model_chat.send_message(
            client_query_string,
            stream=False,
            safety_settings=self.safety_settings,
            generation_config=generation_config)

        text_response = response.text  # type: ignore

        self._langfuse_observation_meta(observation_name="Chat Generate",
                                        query_string=client_query_string,
                                        vertex_model_response=response)

        return text_response

    def parse_json_response(self, res: str) -> dict:
        # Remove the ```json\n and \n``` delimiters
        res = res.replace('```json\n', '').replace('\n```', '')

        # Parse the JSON response
        try:
            res_dict = json.loads(res)
            return res_dict

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}

    def embed_text(self, text: str,
                   task: str = "RETRIEVAL_DOCUMENT",
                   model_name: str = "text-embedding-004",
                   dimensionality: Optional[int] = 768) -> List[float]:
        """Embeds texts with a pre-trained, foundational model."""

        model = TextEmbeddingModel.from_pretrained(model_name)
        input = TextEmbeddingInput(text, task)
        # kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
        embedding = model.get_embeddings(
            texts=[input], output_dimensionality=dimensionality)
        return embedding

    def _vertex_price_estimation(self) -> tuple[float, float]:

        if "gemini-1.5-pro" in self.model_name:
            input_token_price = 0.00125 / 1000
            output_token_price = 0.00375 / 1000

        elif "gemini-1.5-flash" in self.model_name:
            input_token_price = 0.00001875 / 1000
            output_token_price = 0.000075 / 1000

        else:
            raise ValueError(f"Pricing for {self.model_name} not found.")

        return input_token_price, output_token_price

        pass

    def _langfuse_observation_meta(self, observation_name: str,
                                   query_string: str,
                                   vertex_model_response) -> None:
        """
        Update langfuse observation with usage metadata.
        """
        input_token_price, output_token_price = self._vertex_price_estimation()
        input_token_count = int(
            vertex_model_response.usage_metadata.prompt_token_count)
        output_token_count = int(
            vertex_model_response.usage_metadata.candidates_token_count)

        langfuse_context.update_current_observation(
            name=observation_name,
            input=query_string,
            output=vertex_model_response.text,
            usage=ModelUsage(
                unit="TOKENS",
                input=input_token_count,
                output=output_token_count,
                total=int(
                    vertex_model_response.usage_metadata.total_token_count),
                input_cost=float(input_token_price),
                output_cost=float(output_token_price),
                total_cost=float(input_token_price * input_token_count +
                                 output_token_price * output_token_count)
            )
        )
        return None


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
        input_text="",
    )

    llm = LLMSession(
        system_message=graph_extraction_system,
        model_name="gemini-1.5-pro-001"
    )

    # response_schema = {
    # "type": "array",
    # "items": {
    #     "type": "object",
    #     "properties": {
    #         "recipe_name": {
    #             "type": "string",
    #         },
    #     },
    #     "required": ["recipe_name"],
    # },
    # }

    emb = llm.embed_text(text="Hello World!")
    print(emb)
    print(len(emb[0].values))

    # response = llm.generate(client_query_string=graph_extraction_input, )
    # print(response)

    print("Hello World!")
