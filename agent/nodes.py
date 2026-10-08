"""Nodos del grafo: se clasifica, se habla o se trabaja con gate humano.

El router decide la ruta; la rama texto habla sin herramientas vinculadas;
la rama herramienta es un mini-ReAct: el trabajador pide, el humano aprueba,
se ejecuta y el trabajador redacta. Los errores reintentables vuelven al
trabajador con presupuesto 2; lo terminal (rechazo, dato inventado, fallo
agotado) es un mensaje fijo, sin narración del LLM.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.state import AgentState
from agent.tools import TOOLS, calculator, get_weather, list_files, search_files
from config.settings import (
    HABLADOR_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ROUTER_PROMPT,
    TEMPERATURA_HABLADOR,
    TEMPERATURA_ROUTER,
    TEMPERATURA_TRABAJADOR,
    TRABAJADOR_PROMPT,
    mensaje_fallo,
    mensaje_falta_dato,
    mensaje_rechazo,
)

ETIQUETAS = {"TEXTO", "CALCULO", "CLIMA", "ARCHIVOS"}
REINTENTOS_MAXIMOS = 2


def _modelo(temperatura: float) -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperatura,
    )


llm_router = _modelo(TEMPERATURA_ROUTER)  # clasifica, no habla
llm_hablador = _modelo(TEMPERATURA_HABLADOR)  # texto, sin herramientas
llm_trabajador = _modelo(TEMPERATURA_TRABAJADOR)  # se vincula según ruta

# Qué puede pedir el trabajador según la ruta que decidió el router.
_HERRAMIENTAS_POR_RUTA = {
    "CALCULO": [calculator],
    "CLIMA": [get_weather],
    "ARCHIVOS": [search_files, list_files],
}


def herramientas_para(ruta: str) -> list:
    """Subconjunto vinculado al trabajador; fuera de ruta, todo (defensivo)."""
    return _HERRAMIENTAS_POR_RUTA.get(ruta, list(TOOLS))


def router(state: AgentState) -> dict:
    """Etiqueta el turno con UNA palabra; lo desconocido va a TEXTO."""
    peticion = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    veredicto = str(
        llm_router.invoke(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Pregunta: {peticion}"},
            ]
        ).content
        or ""
    ).strip().upper().split()
    etiqueta = veredicto[0] if veredicto else "TEXTO"
    return {"ruta": etiqueta if etiqueta in ETIQUETAS else "TEXTO", "reintentos": 0}


def route_router(state: AgentState) -> str:
    """Texto habla; herramienta trabaja."""
    return "texto" if state.get("ruta", "TEXTO") == "TEXTO" else "herramienta"


def hablador(state: AgentState) -> dict:
    """Responde en texto. No tiene herramientas vinculadas: no hay decisión
    que anunciar porque nunca se le ofreció ninguna."""
    messages = [{"role": "system", "content": HABLADOR_PROMPT}, *state["messages"]]
    return {"messages": [llm_hablador.invoke(messages)]}


def trabajador(state: AgentState) -> dict:
    """Pide la herramienta de su ruta o redacta el resultado.

    Solo ve las herramientas de la ruta que decidió el router. Si lo último
    es un error reintentable, cuenta un reintento: el mensaje de error ya
    está en el historial, que es lo que el modelo necesita para corregirse.
    """
    ruta = state.get("ruta", "TEXTO")
    reintentos = state.get("reintentos", 0)
    if _es_error(state["messages"][-1]):
        reintentos += 1
    messages = [{"role": "system", "content": TRABAJADOR_PROMPT}, *state["messages"]]
    respuesta = llm_trabajador.bind_tools(herramientas_para(ruta)).invoke(messages)
    return {"messages": [respuesta], "reintentos": reintentos}


def route_trabajador(state: AgentState) -> str:
    """Pidió herramienta -> aprobación; respondió -> fin del turno."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "approve"
    return "end"


def approve(state: AgentState) -> dict:
    """Gate humano con verificación de respaldo.

    El argumento tiene que venir del mensaje del usuario, no del modelo: si
    la ciudad o el término no aparecen ahí, ni siquiera se pregunta, se pide
    el dato con mensaje fijo. Lo rechazado termina aquí también con fijo: el
    LLM no lo narra, así que no puede inventar un resultado que nunca existió.
    """
    calls = getattr(state["messages"][-1], "tool_calls", []) or []
    requested = [{"name": c["name"], "args": c.get("args", {})} for c in calls]

    ok, faltante = _args_con_respaldo(calls, state)
    if not ok:
        return {"messages": [AIMessage(content=mensaje_falta_dato(faltante))]}

    decision = interrupt({"requiere_aprobacion": True, "herramientas": requested})
    if decision == "aprobar":
        return {}

    names = ", ".join(r["name"] for r in requested)
    return {"messages": [AIMessage(content=mensaje_rechazo(names))]}


def _args_con_respaldo(calls: list, state: AgentState) -> tuple[bool, str]:
    """El argumento tiene que venir del mensaje del usuario, no del modelo.

    Compara sin espacios ("langgraph" casa con "lang graph") porque el
    trabajador puede normalizar el término. La calculadora queda fuera: su
    expresión se deriva, no se cita. Limitación conocida: seguimientos como
    "¿y mañana?" sin ciudad no pasan; el trabajador debe pedirla en texto.
    """
    peticion = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    compacta = str(peticion or "").replace(" ", "").lower()
    for c in calls:
        args = c.get("args", {}) or {}
        if c.get("name") == "get_weather":
            ciudad = str(args.get("city", ""))
            if not ciudad or ciudad.replace(" ", "").lower() not in compacta:
                return False, "la ciudad"
        elif c.get("name") == "search_files":
            query = str(args.get("query", ""))
            if not query or query.replace(" ", "").lower() not in compacta:
                return False, "el término de búsqueda"
    return True, ""


def route_after_approval(state: AgentState) -> str:
    """Aprobado -> se ejecuta. Fijo (sin tool_calls) -> fin."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "end"


def run_tools(state: AgentState) -> dict:
    """Ejecuta solo lo ya aprobado."""
    return ToolNode(TOOLS).invoke(state)


def _es_error(mensaje) -> bool:
    """Error reintentable: la herramienta dice qué falló.

    "No se encontró" no lo es: la búsqueda funcionó, simplemente no hay
    coincidencias, y repetir la misma query no cambiaría nada.
    """
    if not isinstance(mensaje, ToolMessage):
        return False
    texto = str(mensaje.content or "")
    return texto.startswith("Error:") or texto.startswith("No pude obtener")


def route_tools(state: AgentState) -> str:
    """Error con presupuesto -> reintenta; sin presupuesto -> fijo; si no,
    el trabajador redacta el resultado."""
    if _es_error(state["messages"][-1]):
        if state.get("reintentos", 0) < REINTENTOS_MAXIMOS:
            return "trabajador"
        return "fallo"
    return "trabajador"


def fallo(state: AgentState) -> dict:
    """Presupuesto agotado: mensaje fijo con el motivo, fin del turno."""
    motivo = str(getattr(state["messages"][-1], "content", "") or "")[:120]
    return {"messages": [AIMessage(content=mensaje_fallo(motivo))]}
