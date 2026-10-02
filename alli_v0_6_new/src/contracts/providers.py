"""Phase 1 / 46 — Provider abstractions. No hardcoded Ollama. (rule 46)"""
from __future__ import annotations
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, structured_json: bool = False) -> str: ...

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[str]) -> list[str]: ...

class Storage(ABC):
    @abstractmethod
    def begin(self): ...
    @abstractmethod
    def commit(self): ...
