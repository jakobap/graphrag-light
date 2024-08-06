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
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason, GenerationConfig
import vertexai.preview.generative_models as generative_models
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import json

from typing import List, Optional, Dict, Any

from langfuse.decorators import observe


class LLMSession:
    def __init__(self, system_message: str, model_name: str):
        self.model_name = model_name
        self.system_message = system_message
        self.secrets = dotenv_values(".env")
        self.model = GenerativeModel(
            self.model_name, system_instruction=[system_message])
        self.model_chat = self.model.start_chat()

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
                response_schema=response_schema
            ),
            stream=False
        )
        return response.text  # type: ignore
    
    @observe(as_type="generation")
    def generate_chat(self,
                      client_query_string: str,
                      max_output_tokens: int = 8192,
                      temperature: float = 0.2,
                      top_p: float = 0.5,
                      response_mime_type: Optional[str] = None,
                      response_schema: Optional[Dict[str, Any]] = None) -> str:
        
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "response_mime_type": response_mime_type,
            "response_schema": response_schema
        }

        response = self.model_chat.send_message(
            client_query_string,
            stream=False,
            generation_config=generation_config)
        return response.text  # type: ignore

    def parse_json_response(self, res:str) ->  dict:
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
        embedding = model.get_embeddings(texts=[input], output_dimensionality=dimensionality)
        return embedding

if __name__ == "__main__":
    response_schema = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
            },
            "score": {
                "type": "integer",
            },
        },
        "required": ["response", "score"],
    }

    llm = LLMSession(
        system_message="Answer the following question truthfully and in json format. For every answer you generate you also need to generate a funnyness score to evaluate the fun factor of the question between 1 and 10.",
        model_name="gemini-1.5-pro-001"
    )

    response = llm.generate(client_query_string="Which city has the most bridges?",
                       response_schema=response_schema,
                       response_mime_type="application/json")
    
    print(response)

    print("Hello World!")

