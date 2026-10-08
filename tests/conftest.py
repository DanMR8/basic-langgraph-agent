"""Fixtures compartidas.

El LLM real (Ollama) no está disponible en CI, así que la suite sustituye las
tres vistas del modelo (`llm_router`, `llm_hablador` y `llm_trabajador`) por
dobles que devuelven respuestas preparadas. Eso permite ejercitar el grafo
completo --incluidos `interrupt()` y `recursion_limit`-- sin depender de
ningún servicio externo.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import nodes
from agent.graph import build_agent


class FakeLLM:
    """Doble del LLM: devuelve respuestas en orden.

    Cada nodo del grafo tiene el suyo, así que una cola vacía significa que el
    grafo llamó a un nodo que no debería haber llegado a ejecutarse.
    """

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.prompts: list[list] = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if not self.responses:
            raise AssertionError("El grafo llamó al LLM más veces de lo previsto")
        return self.responses.pop(0)

    def bind_tools(self, tools):
        """El trabajador del prototipo vincula según ruta; el doble lo ignora."""
        return self


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
    """Devuelve una fábrica que construye un grafo con LLM y thread controlados.

    `habla`/`trabaja` con cola vacía significan "este nodo no debe hablar":
    el doble falla si lo invocan, en vez de llamar a Ollama.
    """

    def _montar(router="TEXTO", habla=None, trabaja=None, thread_id="test"):
        monkeypatch.setattr(nodes, "llm_router", FakeLLM([responde(router)]))
        monkeypatch.setattr(nodes, "llm_hablador", FakeLLM(list(habla or [])))
        monkeypatch.setattr(nodes, "llm_trabajador", FakeLLM(list(trabaja or [])))

        grafo = build_agent(checkpointer=InMemorySaver())
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        return grafo, cfg

    return _montar


__all__ = [
    "FakeLLM",
    "ToolMessage",
    "config",
    "montar_grafo",
    "pide_herramienta",
    "responde",
]
