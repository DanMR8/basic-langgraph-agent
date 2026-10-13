"""Nodos del grafo: se clasifica, se habla o se trabaja con gate humano.
El router decide la ruta; la rama texto habla sin herramientas vinculadas;
la rama herramienta es un mini-ReAct: el trabajador pide, el humano aprueba,
se ejecuta y el trabajador redacta. Los errores reintentables vuelven al
trabajador con presupuesto 2; lo terminal (rechazo, dato inventado, fallo
agotado) es un mensaje fijo, sin narración del LLM.
"""

import ast
import difflib
import json
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent.state import AgentState
from agent.tools import TOOLS, _normaliza, calculator, ecobici, get_weather
from config.settings import (
    FRASE_UBICACION_PROMPT,
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

ETIQUETAS = {"TEXTO", "CALCULO", "CLIMA", "MOVILIDAD"}
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
llm_frase = _modelo(TEMPERATURA_ROUTER)  # una frase acotada, con backstop

# Palabras que no respaldan nada: están en casi cualquier mensaje y el
# trabajador las usa como relleno ("cerca de", "mi"). Sin palabras útiles no
# hay respaldo, aunque la subcadena case.
_PARADA_RESPALDO = {
    "de", "la", "el", "en", "y", "a", "mi", "mis", "tu", "tus", "su", "sus",
    "cerca", "hay", "bici", "bicis", "estacion", "estaciones",
    "cicloestacion", "clima", "tiempo", "ciudad", "cuanto", "cuanta", "es",
    "son", "se", "del", "al", "los", "las", "una", "uno", "por", "para",
    "que", "como", "donde", "esta", "estan", "tiene", "tienen",
    "quiere", "saber", "dime", "favor", "porfavor",
}


def _utiles(valor: str) -> list[str]:
    """Palabras del argumento que sí identifican algo (sin paradas)."""
    return [
        w
        for w in _normaliza(valor).split()
        if w not in _PARADA_RESPALDO and (len(w) >= 3 or w.isdigit())
    ]


def _parecido(palabra: str, candidatas: list[str]) -> bool:
    """El trabajador puede corregir un typo ("noviembre"/"noviembvre")."""
    return bool(difflib.get_close_matches(palabra, candidatas, n=1, cutoff=0.85))


# Qué puede pedir el trabajador según la ruta que decidió el router.
_HERRAMIENTAS_POR_RUTA = {
    "CALCULO": [calculator],
    "CLIMA": [get_weather],
    # Cercanas NO está vinculada: solo llega como segundo eslabón vía `ofrecer`,
    # nunca por decisión del trabajador.
    "MOVILIDAD": [ecobici],
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
    if not getattr(respuesta, "tool_calls", None):
        # El 8B a veces escribe la llamada como texto en vez de tool_call
        # (confusión de rol en una sola generación, no memoria sucia: el
        # historial de ese turno está limpio). Si parsea y nombra una
        # herramienta real, se convierte y sigue el flujo normal con gate;
        # si no, queda como texto y el turno termina.
        rescate = _rescatar_llamada(getattr(respuesta, "content", ""))
        if rescate:
            nombre, params = rescate
            respuesta = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": nombre,
                        "args": params,
                        "id": f"recuperada-{uuid4().hex[:8]}",
                    }
                ],
            )
    return {"messages": [respuesta], "reintentos": reintentos}


_LLAMADA_JSON = re.compile(r'^\s*\{\s*"name"\s*:', re.DOTALL)


def _rescatar_llamada(texto) -> tuple | None:
    """Convierte '{"name": ..., "parameters": {...}}' en llamada real.

    El modelo escribe pseudo-JSON con booleanos Python (True/False) y a veces
    con cercas de markdown: se intenta JSON y luego literal_eval, ambos sin
    ejecutar nada. Si no parsea o nombra algo inexistente, None.
    """
    texto = str(texto or "").strip()
    texto = re.sub(r"^```\w*\s*|\s*```$", "", texto).strip()
    if not _LLAMADA_JSON.match(texto):
        return None
    datos = None
    for intento in (json.loads, ast.literal_eval):
        try:
            datos = intento(texto)
            break
        except Exception:
            continue
    if not isinstance(datos, dict):
        return None
    nombre = datos.get("name")
    params = datos.get("parameters", datos.get("args", {}))
    if nombre in {t.name for t in TOOLS} and isinstance(params, dict):
        return nombre, {k: v for k, v in params.items() if isinstance(k, str)}
    return None


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
        peticion = next(
            (m.content for m in reversed(state["messages"]) if m.type == "human"),
            "",
        )
        if _pide_ubicacion(peticion):
            frase = _frase_ubicacion(peticion)
            return {"messages": [AIMessage(content=f"{frase} {mensaje_falta_dato(faltante)}")]}
        return {"messages": [AIMessage(content=mensaje_falta_dato(faltante))]}

    decision = interrupt({"requiere_aprobacion": True, "herramientas": requested})
    if decision == "aprobar":
        return {}

    names = ", ".join(r["name"] for r in requested)
    return {"messages": [AIMessage(content=mensaje_rechazo(names))]}


def _args_con_respaldo(calls: list, state: AgentState) -> tuple[bool, str]:
    """El argumento tiene que venir del mensaje del usuario, no del modelo.

    Compara sin espacios y tolera typos del trabajador ("noviembre" casa con
    "noviembvre"). La calculadora queda fuera: su expresión se deriva, no se
    cita. Limitación conocida: seguimientos como "¿y mañana?" sin ciudad no
    pasan; el trabajador debe pedirla en texto.
    """
    peticion = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    compacta = str(peticion or "").replace(" ", "").lower()
    palabras = _normaliza(peticion).split()

    def respalda(valor: str) -> bool:
        utiles = _utiles(valor)
        if not utiles:
            return False
        return all(
            u in compacta or _parecido(u, palabras) for u in utiles
        )

    # Qué argumento respalda cada herramienta (la calculadora queda fuera: su
    # expresión se deriva, no se cita).
    _RESPALDO_POR_TOOL = {
        "get_weather": ("city", "la ciudad"),
        "ecobici": ("estacion", "la estación"),
        "ecobici_cercanas": ("estacion", "la estación"),
    }
    for c in calls:
        args = c.get("args", {}) or {}
        if c.get("name") in _RESPALDO_POR_TOOL and not respalda(
            args.get(_RESPALDO_POR_TOOL[c["name"]][0], "")
        ):
            return False, _RESPALDO_POR_TOOL[c["name"]][1]
    return True, ""


# Peticiones de ubicación sin referencia utilizable: el agente no sabe dónde
# está el usuario. "Cerca de X" con X concreta NO casa aquí (eso es una
# referencia válida); solo la ubicación del propio usuario.
_PATRON_UBICACION = re.compile(
    r"(cerca de m[ií]\b|donde estoy|mi ubicaci[óo]n|cerca de aqu[íi]\b"
    r"|por aqu[íi]\b|donde me encuentro)",
    re.IGNORECASE,
)

# Backstop de la frase: si el LLM añade bicis, estaciones, cifras o más de
# una frase, se descarta. El fallo medido era justo ese ("Sí, hay bicis
# cerca de ti" sin haber consultado nada).
_PATRON_FUGA = re.compile(
    r"(bici|disponib|estaci[óo]n|hay \d+|\d+ bicis|cercan[oa]s|cerca de [a-záéíóú])",
    re.IGNORECASE,
)
_FRASE_UBICACION_FIJA = "No tengo acceso a tu ubicación."


def _pide_ubicacion(peticion) -> bool:
    """True si pide ubicación del usuario sin dar referencia."""
    return bool(_PATRON_UBICACION.search(str(peticion or "")))


def _frase_ubicacion(peticion: str) -> str:
    """Una frase natural vía LLM, con fijo como red de seguridad."""
    texto = str(
        llm_frase.invoke(
            [
                {"role": "system", "content": FRASE_UBICACION_PROMPT},
                {"role": "user", "content": f"Petición: {peticion}"},
            ]
        ).content
        or ""
    ).strip()
    if not texto or len(texto) > 140 or _PATRON_FUGA.search(texto):
        return _FRASE_UBICACION_FIJA
    return _colapsa_frase(texto)


def _colapsa_frase(texto: str) -> str:
    """Deja una sola frase: colapsa repeticiones y recorta añadidos."""
    # Repetición unida por punto, coma, "y" o espacio ("X.X", "X, x", "X y X").
    doblada = re.fullmatch(
        r"(.+?)[\s.,;:!?…—–\-yY]{1,4}\1\.?", texto.strip(), re.IGNORECASE | re.DOTALL
    )
    if doblada and len(doblada.group(1).strip()) >= 10:
        return doblada.group(1).strip().rstrip(".,;:!?…—–-") + "."
    # Añadido distinto ("X. Con gusto te ayudo."): la primera con contenido
    # real (mínimo 10 para no cortar abreviaturas como "p. ej.").
    for parte in re.split(r"[.!?…;:,—–\-]+", texto):
        parte = parte.strip()
        if len(parte) >= 10:
            return parte + "."
    return _FRASE_UBICACION_FIJA


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
    """Error con presupuesto -> reintenta; sin presupuesto -> fijo.

    Si `ecobici` trajo alguna estación con 3 bicis o menos, se ofrece
    revisar cercanas antes de redactar. El mínimo se lee de la tabla que la
    propia herramienta genera; si no parsea, no se ofrece (fail-safe).

    Las cercanas no las redacta el trabajador (medido: contradice la tabla
    con "no encontré"): van a `resumir`, una línea determinista. Si no
    parsea, cae al trabajador como antes.
    """
    ultimo = state["messages"][-1]
    if _es_error(ultimo):
        if state.get("reintentos", 0) < REINTENTOS_MAXIMOS:
            return "trabajador"
        return "fallo"
    if getattr(ultimo, "name", "") == "ecobici_cercanas":
        contenido = str(getattr(ultimo, "content", "") or "")
        return "resumir" if _resumen_cercanas(contenido) else "trabajador"
    if getattr(ultimo, "name", "") == "ecobici":
        minimo = _min_bicis(str(getattr(ultimo, "content", "") or ""))
        if minimo is not None and minimo <= 3:
            return "ofrecer"
    return "trabajador"


def _min_bicis(texto: str) -> int | None:
    """Mínimo de la columna Bicis de la tabla; None si no hay tabla."""
    valores = re.findall(r"^\|[^|]*\|\s*(\d+)\s*\|", texto, re.MULTILINE)
    return min(map(int, valores)) if valores else None


def _resumen_cercanas(texto: str) -> str | None:
    """Una línea con lo mejor medido y desde dónde; None si no parsea."""
    ref = re.search(r"- Referencia: (.+?), con \d+ bici", texto)
    fila = re.search(
        r"^\| (.+?) \((\d+)\) \| (\d+) m \| (\d+) \|", texto, re.MULTILINE
    )
    if not ref or not fila:
        return None
    nombre, _short, dist, bicis = fila.groups()
    return (
        f"Cerca de {ref.group(1)}: lo mejor es {nombre}, "
        f"{bicis} bicis a {dist} m."
    )


def resumir(state: AgentState) -> dict:
    """Cierra el segundo eslabón sin pasar por el trabajador."""
    ultimo = state["messages"][-1]
    texto = _resumen_cercanas(str(getattr(ultimo, "content", "") or ""))
    return {"messages": [AIMessage(content=texto or "Revisa la tabla de arriba.")]}


def _ultima_llamada(nombre: str, messages) -> dict:
    """La tool_call más reciente a `nombre` (para reutilizar sus args)."""
    for m in reversed(messages):
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == nombre:
                return tc
    return {}


def ofrecer(state: AgentState) -> dict:
    """Ofrece el segundo eslabón con pregunta fija y espera sí/no.

    El "sí" autoriza esa consulta concreta: se construye la llamada a
    `ecobici_cercanas` con la referencia ya dicha y va directo a `tools`,
    sin segunda aprobación. El "no" deja el turno para que el trabajador
    redacte lo ya consultado.
    """
    llamada = _ultima_llamada("ecobici", state["messages"])
    x = str((llamada.get("args", {}) or {}).get("estacion", ""))
    decision = interrupt(
        {
            "pregunta": (
                "Hay estaciones con pocas bicis. ¿Reviso más estaciones "
                f"cerca de '{x}'?"
            ),
            "tipo": "oferta_cercanas",
        }
    )
    if str(decision or "").strip().lower() in {"si", "sí", "s", "yes", "y", "aprobar"}:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ecobici_cercanas",
                            "args": {"estacion": x, "radio_m": 1000},
                            "id": f"oferta-{uuid4().hex[:8]}",
                        }
                    ],
                )
            ]
        }
    return {}


def route_ofrecer(state: AgentState) -> str:
    """Aceptada -> se ejecuta (ya autorizada); si no -> se redacta."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "trabajador"


def fallo(state: AgentState) -> dict:
    """Presupuesto agotado: mensaje fijo con el motivo, fin del turno."""
    motivo = str(getattr(state["messages"][-1], "content", "") or "")[:120]
    return {"messages": [AIMessage(content=mensaje_fallo(motivo))]}
