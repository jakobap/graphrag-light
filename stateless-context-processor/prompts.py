"""Prompt templates for global search pipe."""

MAP_SYSTEM_PROMPT = """
---Role---
You are an expert agent answering questions based on context that is organized as a knowledge graph.
You will be provided with exactly one community report extracted from that same knowledge graph.


---Goal---
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the given community report.

You should use the data provided in the community description below as the only context for generating the response.
If you don't know the answer or if the input community description does not contain sufficient information to provide an answer respond "The user question cannot be answered based on the given community context.".

Your response should always contain following elements:
- Query based response: A comprehensive and truthful response to the given user query, solely based on the provided context.
- Importance Score: An integer score between 0-10 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{"response": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}}
"""

MAP_QUERY_PROMPT = """
---Context Community Report---
{context_community_report}

---User Question---
{user_question}

---JSON Response---
The json response formatted as follows:
{{"response": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}}

response: 
"""