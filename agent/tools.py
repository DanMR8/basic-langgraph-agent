"""Herramientas del agente: calculadora, clima y búsqueda en archivos."""

import ast
import json
import operator
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
    """Evalúa una expresión matemática de forma segura.

    Soporta: + - * / ** % // paréntesis y números (enteros y decimales).
    Ejemplo: "(4 + 5) * 3 / 2"
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
    """Obtiene el clima actual de una ciudad usando wttr.in.

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
    """Busca una palabra o frase en los archivos de texto de la carpeta data/.

    Args:
        query: Término de búsqueda (case-insensitive).
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
# Lista de herramientas disponibles para el grafo
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [calculator, get_weather, search_files]
