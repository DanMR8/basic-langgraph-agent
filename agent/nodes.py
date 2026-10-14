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
from agent.tools import TOOLS, _normaliza, calculator, clima_aqui, ecobici, get_weather, ubicacion
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

ETIQUETAS = {"TEXTO", "CALCULO", "CLIMA", "MOVILIDAD", "UBICACION"}
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
    "CLIMA": [get_weather, clima_aqui],
    # Cercanas NO está vinculada: solo llega como segundo eslabón vía `ofrecer`
    # o con centro del flujo de ubicación, nunca por decisión del trabajador.
    "MOVILIDAD": [ecobici],
    "UBICACION": [ubicacion],
}


def herramientas_para(ruta: str) -> list:
    """Subconjunto vinculado al trabajador; fuera de ruta, todo (defensivo)."""
    return _HERRAMIENTAS_POR_RUTA.get(ruta, list(TOOLS))


def router(state: AgentState) -> dict:
    """Etiqueta el turno con UNA palabra; lo desconocido va a TEXTO.

    Ve historial reciente, no solo la pregunta: los seguimientos ("en
    Azcapotzalco" tras hablar del clima) solo se entienden con contexto.
    """
    ultimos = state["messages"][-6:]
    contexto = "\n".join(
        f"{'Usuario' if m.type == 'human' else 'Asistente'}: "
        f"{str(getattr(m, 'content', '') or '')[:200]}"
        for m in ultimos
    )
    peticion = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "",
    )
    veredicto = str(
        llm_router.invoke(
            [
                {"role": "system", "content": ROUTER_PROMPT},
                {
                    "role": "user",
                    "content": f"Historial:\n{contexto}\nPregunta a clasificar: {peticion}",
                },
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
    """Pidió herramienta -> aprobación; "dónde estoy" elige cómo; si no, fin."""
    llamadas = getattr(state["messages"][-1], "tool_calls", None)
    if llamadas:
        if any(c.get("name") == "ubicacion" for c in llamadas):
            return "elegir"
        return "approve"
    return "end"


def approve(state: AgentState) -> dict:
    """Gate humano con verificación de respaldo.

    El argumento tiene que venir del mensaje del usuario, no del modelo: si
    la ciudad o el término no aparecen ahí, ni siquiera se pregunta, se pide
    el dato con mensaje fijo. Lo rechazado termina aquí también con fijo: el
    LLM no lo narra, así que no puede inventar un resultado que nunca existió.

    Si pedía ubicación propia sin referencia, se interrumpe pidiéndola: el
    front la resuelve (diálogo o toggle) y la CLI declara que no puede. El
    resume trae {"estado": ...} y de ahí se sigue sin cerrar el turno.
    """
    calls = getattr(state["messages"][-1], "tool_calls", []) or []
    requested = [{"name": c["name"], "args": c.get("args", {})} for c in calls]

    ok, faltante = _args_con_respaldo(calls, state)
    if not ok:
        peticion = next(
            (m.content for m in reversed(state["messages"]) if m.type == "human"),
            "",
        )
        if _pide_ubicacion(peticion) and any(
            c.get("name") in ("ecobici", "ecobici_cercanas") for c in calls
        ):
            return _flujo_ubicacion(state, peticion)
        if _pide_ubicacion(peticion):
            frase = _frase_ubicacion(peticion)
            return {"messages": [AIMessage(content=f"{frase} {mensaje_falta_dato(faltante)}")]}
        return {"messages": [AIMessage(content=mensaje_falta_dato(faltante))]}

    decision = interrupt({"requiere_aprobacion": True, "herramientas": requested})
    if decision == "aprobar":
        return {}

    names = ", ".join(r["name"] for r in requested)
    return {"messages": [AIMessage(content=mensaje_rechazo(names))]}


def elegir_ubicacion(state: AgentState) -> dict:
    """Un interrupt por nodo: este solo pregunta precisa/IP/rechazo.

    Precisa con coords válidas ejecuta directo (elegirla ya autorizó);
    aproximada vuelve al gate normal (el grafo la manda a `approve`);
    rechazo con fijo.
    """
    respuesta = interrupt({"tipo": "ubicacion_respuesta"})
    estado = (
        (respuesta or {}).get("estado", "rechazado")
        if isinstance(respuesta, dict)
        else "rechazado"
    )
    if estado == "precisa":
        lat, lon = (respuesta or {}).get("lat"), (respuesta or {}).get("lon")
        try:
            punto = (float(lat), float(lon))
            if -90 <= punto[0] <= 90 and -180 <= punto[1] <= 180:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "geocodificar_inverso",
                                    "args": {"lat": punto[0], "lon": punto[1]},
                                    "id": f"geo-{uuid4().hex[:8]}",
                                }
                            ],
                        )
                    ]
                }
        except (TypeError, ValueError):
            pass
    if estado == "ip":
        return {}
    return {"messages": [AIMessage(content=mensaje_rechazo("ubicacion"))]}


def route_elegir(state: AgentState) -> str:
    """Geocodificar ya autorizado -> tools; IP -> gate normal; fijo -> fin."""
    ultimo = state["messages"][-1]
    if getattr(ultimo, "tool_calls", None):
        llamada = (ultimo.tool_calls or [{}])[0]
        if llamada.get("name") == "geocodificar_inverso":
            return "tools"
        return "approve"
    return "end"



def _flujo_ubicacion(state: AgentState, peticion: str) -> dict:
    """Pide la ubicación precisa y sigue según lo que responda el front.

    El resume trae {"estado": ...}: "ok" trae además lat/lon y se ejecuta
    directo (el "sí" ya autorizó esa consulta); cualquier otro estado deja
    un fijo con el motivo y la guía, y el turno sigue vivo.
    """
    respuesta = interrupt({"tipo": "ubicacion"})
    estado = (
        (respuesta or {}).get("estado", "denegado")
        if isinstance(respuesta, dict)
        else "denegado"
    )
    if estado == "ok":
        lat, lon = (respuesta or {}).get("lat"), (respuesta or {}).get("lon")
        try:
            punto = (float(lat), float(lon))
            if -90 <= punto[0] <= 90 and -180 <= punto[1] <= 180:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "ecobici_cercanas",
                                    "args": {
                                        "lat": punto[0],
                                        "lon": punto[1],
                                        "radio_m": 1000,
                                    },
                                    "id": f"ubi-{uuid4().hex[:8]}",
                                }
                            ],
                        )
                    ]
                }
        except (TypeError, ValueError):
            pass
    guia = mensaje_falta_dato("la estación")
    if estado == "timeout":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No pude obtener tu ubicación (tiempo agotado o sin "
                        f"señal). Seguimos sin ella: {guia}"
                    )
                )
            ]
        }
    if estado == "apagado":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Necesitas tener activada la ubicación en tu "
                        f"dispositivo y darle permiso a la app para usarla. {guia}"
                    )
                )
            ]
        }
    frase = _frase_ubicacion(peticion)
    return {"messages": [AIMessage(content=f"{frase} {guia}")]}


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
    if getattr(ultimo, "name", "") == "geocodificar_inverso":
        return "resumir"
    if getattr(ultimo, "name", "") == "ubicacion":
        return "resumir"
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
    fila = re.search(
        r"^\| (.+?) \((\d+)\) \| (\d+) m \| (\d+) \|", texto, re.MULTILINE
    )
    if not fila:
        return None
    ref = re.search(r"- Referencia: tu ubicación\.", texto)
    nombre_ref = "tu ubicación" if ref else None
    if nombre_ref is None:
        ref2 = re.search(r"- Referencia: (.+?), con \d+ bici", texto)
        if not ref2:
            return None
        nombre_ref = ref2.group(1)
    nombre, _short, dist, bicis = fila.groups()
    return (
        f"Cerca de {nombre_ref}: lo mejor es {nombre}, "
        f"{bicis} bicis a {dist} m."
    )


def resumir(state: AgentState) -> dict:
    """Cierra sin pasar por el trabajador cuando hay resumen determinista."""
    contenido = str(getattr(state["messages"][-1], "content", "") or "")
    texto = _resumen_cercanas(contenido) or _eco_ubicacion(contenido)
    return {"messages": [AIMessage(content=texto or "Revisa el resultado de arriba.")]}


def _eco_ubicacion(texto: str) -> str | None:
    """La tool ya responde en frase completa: se repite tal cual."""
    texto = texto.strip()
    return texto if texto.startswith("Estás en") else None


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
