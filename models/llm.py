from openai import OpenAI
from dotenv import load_dotenv
import os
import httpx
load_dotenv()

class LLM:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")

        http_client = httpx.Client(trust_env=False)

        self.client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )

    def call_llm(self, prompt):
        response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You provide concise, context-grounded answers."},
                    {"role": "user", "content": prompt}
                ]
        )
        return response.choices[0].message.content

    def call_llm_json(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an evaluation judge. Respond ONLY with JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def generate_answer_with_llm(self, query, context):
        prompt = f"""
        You are a helpful audit assistant.
        Answer the question using ONLY the context below.
        Context:
        {context}
        Question:
        {query}
        """
        answer = self.call_llm(prompt)
        return answer
