"""Estado compartido del grafo del agente."""

from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Cada nodo lee y escribe sobre este estado.

    add_messages acumula mensajes en lugar de sobrescribirlos,
    lo que da el efecto de "memoria" de la conversación.

    El límite de ciclos no vive aquí: lo impone LangGraph con
    `recursion_limit` y `GraphRecursionError`.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
