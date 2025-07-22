import os
from mistralai import Mistral
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

class MistralLLM(LLM):
    _client: Mistral = PrivateAttr()
    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()

    def __init__(self, model="mistral-small", temperature=0.1, api_key=None, **kwargs):
        super().__init__(**kwargs)
        self._client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))
        print("Using API key:", os.getenv("MISTRAL_API_KEY"))
        self._model = model
        self._temperature = temperature

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.complete(
            model=self._model, messages=messages, temperature=self._temperature
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "mistral"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self._model, "temperature": self._temperature}
