"""Construcción del grafo del agente con LangGraph."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    approve_tools,
    conversar,
    responder,
    route_after_approval,
    run_tools,
    should_continue,
    validar,
)
from agent.state import AgentState


def build_agent(checkpointer=None):
    """Compila y devuelve el grafo ejecutable del agente.

    Flujo:
        START → conversar ──(respondió en texto)──────────────────────→ END
                    │
                    └─(pidió herramienta)→ validar
                                             ├─(veta)─────────────────→ responder → END
                                             └─(deja pasar)→ approve
                                                                        │
                                                            aprobado ▲ │ ▼ rechazado
                                                                     tools    END
                                                                        │
                                                                        ▼
                                                                  responder → END

    `conversar` abre el turno hablando con el usuario y `responder` lo cierra;
    entre los dos, `validar` decide si la herramienta que pide llega al humano
    o se descarta en el acto. El ciclo siempre termina en una respuesta: sea
    que se ejecutó algo, sea que se vetó.

    El `checkpointer` no es opcional en la práctica: `interrupt()` necesita un
    hilo persistente para poder suspender y reanudar el grafo.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("conversar", conversar)
    workflow.add_node("validar", validar)
    workflow.add_node("approve", approve_tools)
    workflow.add_node("tools", run_tools)
    workflow.add_node("responder", responder)

    workflow.add_edge(START, "conversar")

    # Pedir herramienta no es hablar con el usuario: se veta antes de molestar.
    workflow.add_conditional_edges(
        "conversar",
        should_continue,
        {"validar": "validar", "end": END},
    )

    # `validar` no usa aristas: devuelve Command(goto=...) hacia approve o
    # responder, según su veredicto.

    # Aprobado -> se ejecuta. Rechazado -> el turno termina aquí, sin narrarlo.
    workflow.add_conditional_edges(
        "approve",
        route_after_approval,
        {"tools": "tools", "end": END},
    )

    # Tras ejecutar, el resultado se evalúa y se responde al usuario.
    workflow.add_edge("tools", "responder")

    # El cierre del turno nunca pide herramientas, así que no puede reabrir el
    # ciclo: ni siquiera volviendo a `conversar`.
    workflow.add_edge("responder", END)

    return workflow.compile(checkpointer=checkpointer)


# Instancia lista para usar, con memoria de sesión en proceso.
agent = build_agent(checkpointer=InMemorySaver())
