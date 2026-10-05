"""Nodos del grafo: el modelo decide, el humano aprueba, la herramienta ejecuta."""

from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.state import AgentState
from agent.tools import TOOLS
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPT

# Modelo local vía Ollama
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0,
)

# Vinculamos las herramientas al LLM para que pueda "pedirlas"
llm_with_tools = llm.bind_tools(TOOLS)

# Nodo pre-construido de LangGraph que ejecuta las tool_calls
tool_executor = ToolNode(TOOLS)


def call_model(state: AgentState) -> dict:
    """Nodo 'agent': el LLM piensa y decide si usar herramientas."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    return {"messages": [llm_with_tools.invoke(messages)]}


def should_continue(state: AgentState) -> str:
    """Si el modelo pidió herramientas, hay que pasar por aprobación humana."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "approve"
    return "end"


def approve_tools(state: AgentState) -> dict:
    """Suspende el grafo y espera confirmación del usuario antes de ejecutar.

    `interrupt()` guarda el estado y detiene la ejecución aquí; el grafo se
    reanuda más adelante con `Command(resume=...)` desde este mismo punto.

    Nada de esto se ejecuta por cuenta propia: la decisión es siempre humana.
    """
    calls = getattr(state["messages"][-1], "tool_calls", []) or []
    requested = [{"name": c["name"], "args": c.get("args", {})} for c in calls]

    decision = interrupt({"requiere_aprobacion": True, "herramientas": requested})

    if decision == "aprobar":
        return {}

    # Rechazado: en lugar de devolver el control al LLM para que narre el
    # rechazo (lo cual no es determinista, especialmente en modelos pequeños),
    # registramos un ToolMessage informativo y dejamos que el turno termine.
    # Esta es una decisión de diseño para que el rechazo sea determinista y
    # seguro (no permite que el modelo invente un resultado tras rechazarlo).
    names = ", ".join(r["name"] for r in requested)
    return {
        "messages": [
            ToolMessage(
                content=(
                    f"Operación rechazada por el usuario: {names}. "
                    "La herramienta NO se ejecutó."
                ),
                tool_call_id=c["id"],
                name=c["name"],
            )
            for c in calls
        ]
    }


def route_after_approval(state: AgentState) -> str:
    """Si lo último fue un rechazo, termina el turno (rechazo terminal)."""
    if isinstance(state["messages"][-1], ToolMessage):
        return "end"
    return "tools"


def run_tools(state: AgentState) -> dict:
    """Nodo 'tools': ejecuta las herramientas ya aprobadas."""
    return tool_executor.invoke(state)
