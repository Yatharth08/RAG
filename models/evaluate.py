import json
from models.llm import LLM
class LLMAsAJudge:
    def llm_judge(self,question, ground_truth, model_answer, context):
        listofparameters = ['faithfulness', 'answer_relevance']
        description_of_parameters = {
                "faithfulness": "Rate between 0 and 1. Measures if the model answer is grounded ONLY in the provided context and does not hallucinate.",
                "answer_relevance": "Rate between 0 and 1. Measures how well the model answer addresses the question asked.",
        }
        prompt = f"""
        Extract the evaluation metrics defined in the list of parameters ```{listofparameters}``` \
        using their meaning described in the dictionary ```{description_of_parameters}```.
        You MUST evaluate the model answer using the question, ground truth, and context provided below:
        QUESTION: {question}
        GROUND TRUTH: {ground_truth}
        MODEL ANSWER: {model_answer}
        CONTEXT: {context}
        INSTRUCTIONS:
        - Return ONLY valid JSON.
        {{
            "faithfulness": 0.0,
            "answer_relevance": 0.0
        }}
        - Values must follow the type described:
            - "faithfulness": float (0 to 1)
            - "answer_relevance": float (0 to 1)
        - If a value cannot be computed, assign a reasonable default (0).
        - Just return the JSON. Do not return anything else.
        """
        llm = LLM()
        llmresponse = llm.call_llm_json(prompt)
        return json.loads(llmresponse)
