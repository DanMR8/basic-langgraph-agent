"""Nodos del grafo: se conversa, se veta, se aprueba, se ejecuta, se responde.

La herramienta es un turno interno: el modelo la pide, un validador decide si
merece llegar al humano, el humano aprueba, se ejecuta y solo entonces se
responde al usuario. Si el validador la veta, se responde igualmente, pero sin
herramientas, para que el modelo no pueda seguir pidiendo.
"""

from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agent.state import AgentState
from agent.tools import TOOLS
from config.settings import (
    INSTRUCCION_VETO,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    TEMPERATURA_CONVERSAR,
    TEMPERATURA_RESPONDER,
    TEMPERATURA_VALIDADOR,
    VALIDADOR_PROMPT,
)


def _modelo(temperatura: float) -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperatura,
    )


# Tres capacidades del mismo modelo. Van por separado porque hacen trabajos
# distintos: cada una con su temperatura y con su propio doble en los tests.
llm_con_herramientas = _modelo(TEMPERATURA_CONVERSAR).bind_tools(TOOLS)  # conversar
llm_validador = _modelo(TEMPERATURA_VALIDADOR)  # validar: opina, no habla con el usuario
llm_sin_herramientas = _modelo(TEMPERATURA_RESPONDER)  # responder: solo texto, no pide

# Nodo pre-construido de LangGraph que ejecuta las tool_calls
tool_executor = ToolNode(TOOLS)

# Contenido fijo del ToolMessage de veto. Solo sirve como señal para
# `responder`: no es un registro que deba conservarse, porque lo que queda en
# el historial lo devuelve el modelo como si fuera suya.
VETO_CONTENIDO = "No se ejecutó: el pedido fue descartado antes de ejecutarse."


def _es_veto(mensaje) -> bool:
    return isinstance(mensaje, ToolMessage) and mensaje.content == VETO_CONTENIDO


def _mensajes_a_borrar(messages) -> list[RemoveMessage]:
    """El par tool_call + veto del turno que está cerrándose.

    Se borra de la vez porque el historial es lo único que el modelo mira: lo
    que queda ahí escrito, lo copia. Un registro suelto salió entero en la boca
    del asistente tres turnos después y, a partir de ahí, se repetía solo.

    Si el par no se puede identificar, no se borra nada y el turno se cierra
    igual: preferible conservar un registro de más a dejar una tool_call huérfana.
    """
    ultimo = messages[-1]
    if not _es_veto(ultimo) or not ultimo.id or not ultimo.tool_call_id:
        return []

    padre = next(
        (
            m
            for m in reversed(messages)
            if m.type == "ai"
            and any(
                tc.get("id") == ultimo.tool_call_id
                for tc in (getattr(m, "tool_calls", None) or [])
            )
        ),
        None,
    )
    if padre is None or not padre.id:
        return []

    return [RemoveMessage(id=padre.id), RemoveMessage(id=ultimo.id)]


def conversar(state: AgentState) -> dict:
    """Habla con el usuario; si necesita un dato, pide una herramienta.

    Con herramientas vinculadas el modelo tiende a pedirla casi siempre, así
    que su juicio sobre cuándo usarla no es fiable: a un saludo le responde
    `calculator(expression='hola')`. Por eso su petición no va directo al
    humano, sino que pasa por `validar`.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]]
    return {"messages": [llm_con_herramientas.invoke(messages)]}


def should_continue(state: AgentState) -> str:
    """Si pidió herramienta, hay que vetarla antes de preguntarle al humano."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "validar"
    return "end"


def validar(state: AgentState) -> Command:
    """Veta la petición de herramienta antes de que llegue a la aprobación.

    `Command(goto=...)` decide el destino sin tocar el estado: `AgentState`
    sigue siendo solo `messages`.

    Si la veta, devuelve un `ToolMessage` de texto fijo. No es solo para que
    el modelo se entere: también cierra la `tool_call` abierta, que la API de
    chat exige responder. Es una señal para `responder`, no un registro: él se
    encarga de borrar el par antes de terminar.
    """
    peticion = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    llamada = (getattr(state["messages"][-1], "tool_calls", None) or [{}])[0]
    propuesta = f"{llamada.get('name')}({llamada.get('args')})"

    veredicto = llm_validador.invoke(
        [
            {"role": "system", "content": VALIDADOR_PROMPT},
            {"role": "user", "content": f"Pregunta: {peticion}\nPropuesta: {propuesta}"},
        ]
    )
    texto = str(veredicto.content).strip().upper().replace("Í", "I")
    if (texto.split() or [""])[0] == "SI":
        return Command(goto="approve")

    return Command(
        goto="responder",
        update={
            "messages": [
                ToolMessage(
                    content=VETO_CONTENIDO,
                    tool_call_id=llamada.get("id", ""),
                    name=llamada.get("name", ""),
                )
            ]
        },
    )


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


def responder(state: AgentState) -> dict:
    """Cierra el turno dando la respuesta al usuario, sin herramientas.

    No puede pedir ninguna: es lo que garantiza que el ciclo interno de la
    herramienta termine siempre en una respuesta, tanto si se ejecutó como si
    el validador la vetó.

    Si el turno cierra con un veto, la orden correspondiente se añade aquí, al
    system prompt de esta llamada, y no a ningún mensaje. Y el par que la
    señaliza se borra del estado en el mismo update: lo único que persiste es
    lo que el usuario preguntó y lo que el asistente respondió.
    """
    sistema = SYSTEM_PROMPT
    borrar: list[RemoveMessage] = []
    if _es_veto(state["messages"][-1]):
        sistema = f"{sistema}\n{INSTRUCCION_VETO}"
        borrar = _mensajes_a_borrar(state["messages"])

    messages = [{"role": "system", "content": sistema}, *state["messages"]]
    return {"messages": [*borrar, llm_sin_herramientas.invoke(messages)]}
