"""Construcción del grafo del agente con LangGraph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    approve,
    fallo,
    hablador,
    route_after_approval,
    route_router,
    route_tools,
    route_trabajador,
    router,
    run_tools,
    trabajador,
)
from agent.state import AgentState


def build_agent(checkpointer=None):
    """Compila y devuelve el grafo ejecutable del agente.

    Flujo por turno:
        router -> hablador -> END (texto)
        router -> trabajador -> approve -> tools -> trabajador (bucle ReAct)
        En el bucle: error con presupuesto -> reintenta; agotado -> fallo;
        ok -> el trabajador redacta. Rechazo o dato inventado -> fijo -> END.

    El router decide; el hablador no tiene herramientas vinculadas; el
    trabajador solo ve las de su ruta. Lo terminal es siempre un mensaje
    fijo, nunca una narración del LLM.

    El `checkpointer` no es opcional en la práctica: `interrupt()` necesita un
    hilo persistente para poder suspender y reanudar el grafo.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router)
    workflow.add_node("hablador", hablador)
    workflow.add_node("trabajador", trabajador)
    workflow.add_node("approve", approve)
    workflow.add_node("tools", run_tools)
    workflow.add_node("fallo", fallo)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_router,
        {"texto": "hablador", "herramienta": "trabajador"},
    )
    workflow.add_edge("hablador", END)

    workflow.add_conditional_edges(
        "trabajador",
        route_trabajador,
        {"approve": "approve", "end": END},
    )
    workflow.add_conditional_edges(
        "approve",
        route_after_approval,
        {"tools": "tools", "end": END},
    )
    workflow.add_conditional_edges(
        "tools",
        route_tools,
        {"trabajador": "trabajador", "fallo": "fallo"},
    )
    workflow.add_edge("fallo", END)

    return workflow.compile(checkpointer=checkpointer)


# Instancia lista para usar, con memoria de sesión en proceso.
agent = build_agent(checkpointer=InMemorySaver())
