"""Herramientas del agente: calculadora, clima y búsqueda en archivos."""

import ast
import json
import operator
import unicodedata
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clima (API pública wttr.in — sin API key)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Devuelve el clima ACTUAL de una ciudad.

    Úsala solo cuando pregunten por el tiempo de una ciudad concreta. Para
    cualquier otra pregunta, no la llames.

    Args:
        city: Nombre de la ciudad en inglés o español.
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        current = data["current_condition"][0]
        temp_c = current["temp_C"]
        feels_c = current["FeelsLikeC"]
        if "lang_es" in current:
            desc = current["lang_es"][0]["value"]
        else:
            desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]

        return (
            f"🌤️ Clima en {city.title()}:\n"
            f"  • Temperatura: {temp_c}°C (sensación {feels_c}°C)\n"
            f"  • Condición: {desc}\n"
            f"  • Humedad: {humidity}%\n"
            f"  • Viento: {wind} km/h"
        )
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
            "- Prueba con una colonia, avenida o lugar (p. ej. 'Reforma', "
            "'Azcapotzalco') o el número de la cicloestación."
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
TOOLS = [calculator, get_weather, ecobici]
