# rag/llm.py
import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    num_ctx: int | None = None   # Ollama only

PROVIDERS = {
    "groq": LLMConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", ""),
        model="openai/gpt-oss-20b",
    ),
    "ollama": LLMConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",            # ignored, but the SDK requires a value
        model="qwen2.5:3b-instruct",
        num_ctx=2048,
    ),
}

class LLM:
    def __init__(self, provider: str = None):
        name = provider or os.environ.get("LLM_PROVIDER", "groq")
        self.cfg = PROVIDERS[name]
        self.name = name
        self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        extra = {"num_ctx": self.cfg.num_ctx} if self.cfg.num_ctx else {}
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"} if json_mode else None,
            extra_body=extra,
        )
        return resp.choices[0].message.content