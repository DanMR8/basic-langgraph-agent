"""Fixtures compartidas.

El LLM real (Ollama) no está disponible en CI, así que la suite sustituye
`llm_with_tools` por un doble que devuelve respuestas preparadas. Eso permite
ejercitar el grafo completo --incluidos `interrupt()` y `recursion_limit`--
sin depender de ningún servicio externo.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import nodes
from agent.graph import build_agent


class FakeLLM:
    """Doble de `llm_with_tools`: devuelve respuestas en orden."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.prompts: list[list] = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.responses:
            raise AssertionError("El grafo llamó al LLM más veces de lo previsto")
        return self.responses.pop(0)


def pide_herramienta(nombre: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": call_id}])


def responde(texto: str) -> AIMessage:
    return AIMessage(content=texto)


@pytest.fixture
def config() -> dict:
    """Config de un hilo aislado, como la que usa la CLI."""
    return {"configurable": {"thread_id": "test"}, "recursion_limit": 10}


@pytest.fixture
def montar_grafo(monkeypatch):
    """Devuelve una fábrica que construye un grafo con LLM y thread controlados."""

    def _montar(responses: list[AIMessage], thread_id: str = "test"):
        fake = FakeLLM(responses)
        monkeypatch.setattr(nodes, "llm_with_tools", fake)
        grafo = build_agent(checkpointer=InMemorySaver())
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        return grafo, cfg, fake

    return _montar


__all__ = [
    "FakeLLM",
    "ToolMessage",
    "config",
    "montar_grafo",
    "pide_herramienta",
    "responde",
]
