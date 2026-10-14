"""Herramientas del agente: calculadora, clima y búsqueda en archivos."""

import ast
import json
import math
import operator
import unicodedata
import urllib.parse
import urllib.request

from langchain_core.tools import tool

from config.settings import DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# 1. Calculadora
# ─────────────────────────────────────────────────────────────────────────────

_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Sin topes, `9**9**9` genera un entero de ~387 millones de dígitos y cuelga
# el proceso. Se acota el exponente y el tamaño del resultado.
_EXPONENTE_MAXIMO = 100
_RESULTADO_MAXIMO = 10**1000


def _evaluar(nodo: ast.AST) -> int | float:
    """Interpreta un AST aritmético usando solo operaciones de la tabla.

    Rechaza cualquier construcción que no sea un número, un operador o un
    paréntesis: no hay forma de llegar a un nombre, atributo o llamada.
    """
    if isinstance(nodo, ast.Expression):
        return _evaluar(nodo.body)

    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, bool) or not isinstance(nodo.value, (int, float)):
            raise ValueError("solo se admiten números.")
        return nodo.value

    if isinstance(nodo, ast.BinOp):
        operacion = _OPERADORES.get(type(nodo.op))
        if operacion is None:
            raise ValueError("operador no permitido.")
        izquierda = _evaluar(nodo.left)
        derecha = _evaluar(nodo.right)
        if operacion is operator.pow and abs(derecha) > _EXPONENTE_MAXIMO:
            raise ValueError(f"el exponente no puede superar {_EXPONENTE_MAXIMO}.")
        resultado = operacion(izquierda, derecha)
        if abs(resultado) > _RESULTADO_MAXIMO:
            raise ValueError("el resultado es demasiado grande.")
        return resultado

    if isinstance(nodo, ast.UnaryOp):
        if isinstance(nodo.op, ast.UAdd):
            return +_evaluar(nodo.operand)
        if isinstance(nodo.op, ast.USub):
            return -_evaluar(nodo.operand)
        raise ValueError("operador unario no permitido.")

    raise ValueError("expresión no permitida.")


@tool
def calculator(expression: str) -> str:
    """Úsala SOLO cuando la respuesta dependa de una cuenta aritmética concreta.

    Soporta: + - * / ** % // paréntesis y números (enteros y decimales).
    Ejemplo: "(4 + 5) * 3 / 2"
    Si la pregunta no es una cuenta, no la llames: responde tú en texto.
    """
    try:
        return f"Resultado: {_evaluar(ast.parse(expression.strip(), mode='eval'))}"
    except ZeroDivisionError:
        return "Error: división entre cero."
    except SyntaxError:
        return "Error: expresión no válida."
    except ValueError as e:
        return f"Error: {e}"


# wttr.in devuelve la condición en inglés cuando no trae `lang_es`. Tabla
# cerrada con lo común: el trabajador no traduce términos climáticos con
# fiabilidad (medido: "Cloudy" -> "despejado").
_CLIMA_ES = {
    "Clear": "despejado",
    "Sunny": "soleado",
    "Partly cloudy": "parcialmente nublado",
    "Cloudy": "nublado",
    "Overcast": "cubierto",
    "Mist": "neblina",
    "Fog": "niebla",
    "Light rain": "lluvia ligera",
    "Moderate rain": "lluvia moderada",
    "Heavy rain": "lluvia fuerte",
    "Light drizzle": "llovizna ligera",
    "Thunderstorm": "tormenta",
    "Light snow": "nieve ligera",
    "Heavy snow": "nieve fuerte",
}


def _condicion_en_espanol(current: dict) -> str:
    if "lang_es" in current:
        return current["lang_es"][0]["value"]
    original = str(current["weatherDesc"][0]["value"]).strip()
    return _CLIMA_ES.get(original, original)


def _formatea_clima(data: dict, titulo: str) -> str:
    current = data["current_condition"][0]
    temp_c = current["temp_C"]
    feels_c = current["FeelsLikeC"]
    desc = _condicion_en_espanol(current)
    humidity = current["humidity"]
    wind = current["windspeedKmph"]
    return (
        f"🌤️ {titulo}:\n"
        f"  • Temperatura: {temp_c}°C (sensación {feels_c}°C)\n"
        f"  • Condición: {desc}\n"
        f"  • Humedad: {humidity}%\n"
        f"  • Viento: {wind} km/h"
    )

@tool
def ubicacion() -> str:
    """Ciudad aproximada donde estás, por IP (sin GPS).

    Úsala SOLO si preguntan dónde están, en qué ciudad están o cuál es su
    ubicación. Es aproximada a nivel ciudad/zona, nunca una dirección.
    """
    try:
        url = (
            "http://ip-api.com/json/"
            "?fields=status,message,country,regionName,city,lat,lon"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "basic-langgraph-agent"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") != "success":
            return f"No pude obtener tu ubicación: {data.get('message', 'sin motivo')}."
        ciudad = data.get("city") or "?"
        region = data.get("regionName") or "?"
        pais = data.get("country") or "?"
        return (
            f"Estás en {ciudad}, {region} ({pais}), aproximado por IP. "
            "No es tu dirección exacta."
        )
    except Exception as e:
        return f"No pude obtener tu ubicación: {e}"


@tool
def geocodificar_inverso(lat: float, lon: float) -> str:
    """Colonia y alcaldía de unas coordenadas (Nominatim, sin key).

    Nunca la pide el trabajador: solo la construye el flujo de ubicación
    con coords ya autorizadas. Responde en frase completa para eco directo.
    """
    try:
        if (
            isinstance(lat, bool)
            or isinstance(lon, bool)
            or not isinstance(lat, (int, float))
            or not isinstance(lon, (int, float))
            or not (-90 <= lat <= 90)
            or not (-180 <= lon <= 180)
        ):
            return "Error: coordenadas inválidas."
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&accept-language=es"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "basic-langgraph-agent/0.1"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        a = data.get("address", {}) or {}
        zona = a.get("neighbourhood") or a.get("road")
        ciudad = (
            a.get("borough") or a.get("city") or a.get("state")
        )
        if zona and ciudad:
            return f"Estás en {zona}, {ciudad} (ubicación precisa)."
        if ciudad:
            return f"Estás en {ciudad} (ubicación precisa)."
        if zona:
            return f"Estás en {zona} (ubicación precisa)."
        return "Estás por aquí (ubicación precisa)."
    except Exception as e:
        return f"Error: no pude geocodificar: {e}"


@tool
def clima_aqui() -> str:
    """Clima ACTUAL de tu ubicación aproximada (por IP, sin ciudad).

    Úsala SOLO si piden el clima aquí, donde están o en su ubicación, o se
    entiende que lo quieren donde se encuentran. Si nombran una ciudad,
    usa `get_weather` con esa ciudad.
    """
    try:
        url = "https://wttr.in/?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        lugar = _nombre_ubicacion()
        return _formatea_clima(data, f"Clima en {lugar} (ubicación aproximada)")
    except Exception as e:
        return f"No pude obtener el clima de tu ubicación: {e}"


def _nombre_ubicacion() -> str:
    """Ciudad que wttr.in asigna a tu IP; "por aquí" si no responde."""
    try:
        req = urllib.request.Request(
            "https://wttr.in/?format=%l", headers={"User-Agent": "curl"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            nombre = resp.read().decode().strip()
        return nombre or "por aquí"
    except Exception:
        return "por aquí"


@tool
def get_weather(city: str) -> str:
    """Devuelve el clima ACTUAL de un lugar concreto (obligatorio).

    Úsala cuando nombren el lugar: ciudad, alcaldía o colonia, tal cual.
    Si piden el clima aquí, donde están o en su ubicación sin nombrar
    lugar, usa `clima_aqui` en su lugar.

    Args:
        city: Nombre del lugar en inglés o español.
    """
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city.strip())}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return _formatea_clima(data, f"Clima en {city.strip().title()}")
    except Exception as e:
        return f"No pude obtener el clima de '{city}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Búsqueda en archivos locales
# ─────────────────────────────────────────────────────────────────────────────

def _load_documents() -> dict[str, str]:
    """Lee todos los archivos de texto de DATA_DIR y los devuelve como dict."""
    docs: dict[str, str] = {}
    if not DATA_DIR.exists():
        return docs

    for ext in ("*.txt", "*.md"):
        for file_path in DATA_DIR.rglob(ext):
            try:
                docs[str(file_path.relative_to(DATA_DIR))] = file_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception:
                continue
    return docs


@tool
def search_files(query: str) -> str:
    """Busca un término DENTRO de los archivos .txt/.md de la carpeta data/.

    Args:
        query: Una sola palabra o término corto que aparezca en esos archivos
            (case-insensitive). No le pases una frase entera ni preguntas.
    """
    docs = _load_documents()
    if not docs:
        return f"No hay archivos en '{DATA_DIR}'. Agrega .txt o .md para buscar."

    query_lower = query.lower()
    matches: list[str] = []

    for filename, content in docs.items():
        lines = content.splitlines()
        for i, line in enumerate(lines, start=1):
            if query_lower in line.lower():
                matches.append(f"📄 {filename}:{i} → {line.strip()[:120]}")

    if not matches:
        return f"No se encontró '{query}' en ningún archivo."

    header = f"Se encontraron {len(matches)} coincidencia(s) para '{query}':\n"
    return header + "\n".join(matches[:20])  # máx 20 resultados


# ─────────────────────────────────────────────────────────────────────────────
# 4. Listado de archivos locales
# ─────────────────────────────────────────────────────────────────────────────


@tool
def list_files() -> str:
    """Lista los archivos .txt/.md disponibles en la carpeta data/.

    No lleva argumentos. Úsala cuando pregunten qué archivos hay o qué se
    puede consultar, antes de buscar nada dentro.
    """
    if not DATA_DIR.exists():
        return f"No existe la carpeta '{DATA_DIR}'."
    archivos = sorted(
        [p for ext in ("*.txt", "*.md") for p in DATA_DIR.rglob(ext)]
    )
    if not archivos:
        return f"No hay archivos en '{DATA_DIR}'. Agrega .txt o .md."
    lineas = []
    for p in archivos:
        try:
            primera = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            descripcion = primera[0].strip()[:100] if primera else "(vacío)"
        except Exception:
            descripcion = "(no se pudo leer)"
        lineas.append(f"📁 {p.relative_to(DATA_DIR)}: {descripcion}")
    return f"Archivos en '{DATA_DIR}':\n" + "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Disponibilidad de Ecobici (GBFS público, sin key)
# ─────────────────────────────────────────────────────────────────────────────

_ECOBICI_BASE = "https://gbfs.mex.lyftbikes.com/gbfs/es"
_ECOBICI_TIMEOUT = 10
_ECOBICI_MAXIMAS = 5


def _normaliza(texto: str) -> str:
    """Minúsculas sin acentos ni espacios de más, para casar búsquedas."""
    sin_acentos = "".join(
        c
        for c in unicodedata.normalize("NFD", str(texto or ""))
        if not unicodedata.combining(c)
    )
    return " ".join(sin_acentos.lower().split())


def _gbfs(nombre: str) -> dict:
    req = urllib.request.Request(
        f"{_ECOBICI_BASE}/{nombre}.json", headers={"User-Agent": "basic-langgraph-agent"}
    )
    with urllib.request.urlopen(req, timeout=_ECOBICI_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _casa(estacion: dict, clave: str) -> bool:
    texto = f"{estacion.get('name', '')} {estacion.get('short_name', '')}"
    return clave in _normaliza(texto)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos (fórmula haversiana)."""
    radio = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radio * math.asin(math.sqrt(a))


def _coords(estacion: dict) -> tuple | None:
    try:
        return float(estacion["lat"]), float(estacion["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def _valida_centro(lat, lon) -> tuple | None:
    """Coords válidas o None (también si solo viene una)."""
    if lat is None and lon is None:
        return None
    if (
        isinstance(lat, bool)
        or isinstance(lon, bool)
        or not isinstance(lat, (int, float))
        or not isinstance(lon, (int, float))
        or not (-90 <= lat <= 90)
        or not (-180 <= lon <= 180)
    ):
        return "invalido"
    return float(lat), float(lon)


def _id_num(estacion: dict) -> int:
    """Primer número del id ("192-193" -> 192); lo no numérico va al final."""
    primero = str(estacion.get("station_id", "")).split("-")[0]
    return int(primero) if primero.isdigit() else 10**9


@tool
def ecobici_cercanas(
    estacion: str = "", lat: float | None = None, lon: float | None = None,
    radio_m: int = 1000,
) -> str:
    """Bicis disponibles cerca de un punto, en tiempo real.

    Dos modos excluyentes: por palabra (`estacion`, contra nombres) o por
    centro directo (`lat`/`lon`, para la ubicación del usuario). Con ambos,
    manda el centro directo.

    Args:
        estacion: palabra o número tal como lo escribió el usuario.
        lat: latitud del centro (-90..90). Solo la pone el flujo de
            ubicación, nunca el trabajador.
        lon: longitud del centro (-180..180). Igual que `lat`.
        radio_m: radio en metros. Siempre 1000; no lo cambies.
    """
    import time

    if isinstance(radio_m, bool) or not isinstance(radio_m, int) or radio_m <= 0:
        return "Error: radio en metros, entero mayor que 0."
    centro = _valida_centro(lat, lon)
    if centro == "invalido":
        return "Error: coordenadas inválidas."
    if centro is None and not _normaliza(estacion):
        return "Error: dime la calle o número de referencia."
    try:
        info = _gbfs("station_information")["data"]["stations"]
        estado = _gbfs("station_status")
    except Exception as e:
        return f"Error: no pude consultar Ecobici: {e}"

    por_id = {e["station_id"]: e for e in estado["data"]["stations"]}
    if centro is not None:
        titulo = "tu ubicación"
        nombre_ref = titulo
        ref_id = None
        bicis_ref = 0
        linea_ref = "- Referencia: tu ubicación."
        punto = centro
    else:
        clave = _normaliza(estacion)
        candidatas = [s for s in info if _casa(s, clave)]
        if not candidatas:
            return (
                f"- No encontré estación con '{estacion}'.\n"
                "- Prueba con la calle donde está (p. ej. 'Reforma', "
                "'20 de Noviembre' o 'Insurgentes') o el número de la cicloestación."
            )
        # Referencia: la de más bicis entre las que casan (desempate: id
        # menor). Se declara con el criterio a la vista y no se lista como
        # hallazgo: presentarla a 0 m como "la más cercana" era redundante
        # y elegir por orden del feed parecía arbitrario.
        ref = max(
            candidatas,
            key=lambda s: (
                por_id.get(s["station_id"], {}).get("num_bikes_available", 0),
                -_id_num(s),
            ),
        )
        punto = _coords(ref)
        if punto is None:
            return "Error: la referencia no trae coordenadas."
        titulo = f"'{estacion}'"
        ref_id = ref["station_id"]
        bicis_ref = por_id.get(ref["station_id"], {}).get("num_bikes_available", 0)
        nombre_ref = f"{ref.get('name')} ({ref.get('short_name')})"
        bicis_txt = "1 bici" if bicis_ref == 1 else f"{bicis_ref} bicis"
        linea_ref = (
            f"- Referencia: {nombre_ref}, con {bicis_txt}: "
            f"la de más bicis entre {len(candidatas)} coincidencia(s) con '{estacion}'."
        )
    vecinas = []
    for s in info:
        if ref_id is not None and s["station_id"] == ref_id:
            continue
        lugar = _coords(s)
        if lugar is None:
            continue
        e = por_id.get(s["station_id"], {})
        d = _haversine_m(punto[0], punto[1], lugar[0], lugar[1])
        if e.get("is_renting", 1) and e.get("num_bikes_available", 0) > 0 and d <= radio_m:
            vecinas.append((d, s, e))
    vecinas.sort(key=lambda t: t[0])

    ts = estado.get("last_updated", 0)
    edad = f" (dato de hace {int(time.time() - ts)}s)" if ts else ""
    if not vecinas:
        if bicis_ref > 0:
            return (
                f"Ecobici cerca de {titulo}{edad}:\n"
                f"{linea_ref}\n"
                f"- Sin otras estaciones con bicis en {radio_m} m."
            )
        detalle = f" (referencia: {nombre_ref})." if ref_id is not None else "."
        return (
            f"- No hay bicis en {radio_m} m alrededor de {titulo}{detalle}\n"
            "- Prueba con otra calle o número."
        )
    lineas = [
        f"Ecobici cerca de {titulo}{edad}:",
        linea_ref,
        "| Estación | Distancia | Bicis | Espacios |",
        "|---|---|---|---|",
    ]
    for d, s, e in vecinas[:_ECOBICI_MAXIMAS]:
        lineas.append(
            f"| {s.get('name')} ({s.get('short_name')}) | {int(round(d))} m "
            f"| {e.get('num_bikes_available', '?')} "
            f"| {e.get('num_docks_available', '?')} |"
        )
    return "\n".join(lineas)


@tool
def ecobici(estacion: str, conclusion: bool = False) -> str:
    """Disponibilidad de bicis Ecobici CDMX por estación, en tiempo real.

    Args:
        estacion: palabra o número tal como lo escribió el usuario
            ("reforma", "710", "lealtad"). No inventes nombres: si no hay
            coincidencia, dilo.
        conclusion: veredicto determinista ("hay bicis en N de M"). Solo
            para pruebas sin LLM; con LLM redacta el trabajador.
    """
    import time

    clave = _normaliza(estacion)
    if not clave:
        return "Error: dime el nombre o número de la estación."
    try:
        info = _gbfs("station_information")["data"]["stations"]
        estado = _gbfs("station_status")
    except Exception as e:
        return f"Error: no pude consultar Ecobici: {e}"

    por_id = {e["station_id"]: e for e in estado["data"]["stations"]}
    halladas = [s for s in info if _casa(s, clave)]
    if not halladas:
        return (
            f"- No encontré estación con '{estacion}'.\n"
            "- Prueba con la calle donde está (p. ej. 'Reforma', "
            "'20 de Noviembre' o 'Insurgentes') o el número de la cicloestación."
        )

    halladas.sort(
        key=lambda s: por_id.get(s["station_id"], {}).get("num_bikes_available", 0),
        reverse=True,
    )
    ts = estado.get("last_updated", 0)
    edad = f" (dato de hace {int(time.time() - ts)}s)" if ts else ""
    n = len(halladas)
    conteo = (
        "- 1 coincidencia, top 5 por bicis:"
        if n == 1
        else f"- {n} coincidencias, top {_ECOBICI_MAXIMAS} por bicis:"
    )
    lineas = [
        f"Ecobici — '{estacion}'{edad}:",
        conteo,
        "| Estación | Bicis | Espacios | Estado |",
        "|---|---|---|---|",
    ]
    for s in halladas[:_ECOBICI_MAXIMAS]:
        e = por_id.get(s["station_id"], {})
        estado_txt = "abierta" if e.get("is_renting", 1) else "sin servicio"
        lineas.append(
            f"| {s.get('name')} ({s.get('short_name')}) "
            f"| {e.get('num_bikes_available', '?')} "
            f"| {e.get('num_docks_available', '?')} | {estado_txt} |"
        )
    if len(halladas) > _ECOBICI_MAXIMAS:
        lineas.append(f"- …y {len(halladas) - _ECOBICI_MAXIMAS} más con '{estacion}'.")
    if conclusion:
        con_bicis = sum(
            1
            for s in halladas
            if por_id.get(s["station_id"], {}).get("num_bikes_available", 0) > 0
        )
        # Veredicto en el texto: el trabajador copia lo escrito, así no tiene
        # que interpretar los números al redactar (medido: con 0 bicis decía
        # "hay"). Solo para pruebas sin LLM; con LLM redacta el trabajador.
        lineas.append(
            "- Conclusión: "
            + (
                f"hay bicis en {con_bicis} de {len(halladas)}."
                if con_bicis
                else "sin bicis en las coincidencias."
            )
        )
    return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# Lista de herramientas disponibles para el grafo
# ─────────────────────────────────────────────────────────────────────────────

# search_files y list_files existen pero están inhabilitadas: la búsqueda en
# archivos locales no es parte del producto. Se conservan el código y sus
# tests por si se reactivan.
TOOLS = [
    calculator, get_weather, clima_aqui, ecobici, ecobici_cercanas,
    ubicacion, geocodificar_inverso,
]
