"""Construcción del grafo del agente con LangGraph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    approve_tools,
    call_model,
    route_after_approval,
    run_tools,
    should_continue,
)
from agent.state import AgentState


def build_agent(checkpointer=None):
    """Compila y devuelve el grafo ejecutable del agente.

    Flujo:
        START → agent ──(pidió herramientas)──→ approve
                   ▲                             │
                   │                    aprobado │ rechazado
                   │                             ▼
                   └──────────────────────  agent
                   ▲                             │
                   └────────────────────  tools ←┘ (aprobado)
                   │
                   └──(respuesta final)──→ END

    El `checkpointer` no es opcional en la práctica: `interrupt()` necesita un
    hilo persistente para poder suspender y reanudar el grafo.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("approve", approve_tools)
    workflow.add_node("tools", run_tools)

    workflow.add_edge(START, "agent")

    # El modelo pidió herramientas: exige confirmación antes de actuar.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"approve": "approve", "end": END},
    )

    # Aprobado -> se ejecuta. Rechazado -> el LLM se entera y sigue pensando.
    workflow.add_conditional_edges(
        "approve",
        route_after_approval,
        {"tools": "tools", "end": END},
    )

    # Tras ejecutar, el agente vuelve a razonar con el resultado en contexto.
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


# Instancia lista para usar, con memoria de sesión en proceso.
agent = build_agent(checkpointer=InMemorySaver())
