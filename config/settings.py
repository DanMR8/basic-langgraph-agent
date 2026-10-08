"""Configuración centralizada del agente."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Modelo ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Temperatura por nodo:
#   - `router` clasifica en una palabra: reproducibilidad total, 0.0.
#   - `hablador` y `trabajador` generan texto y llamadas: 0.5, el punto
#     medido donde el 8B ni se agarra a plantillas ni suelta JSON crudo.
TEMPERATURA_ROUTER: float = 0.0
TEMPERATURA_HABLADOR: float = 0.5
TEMPERATURA_TRABAJADOR: float = 0.5

# ── Datos ───────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Agente ──────────────────────────────────────────────────────────────────
# Límite de pasos del grafo por turno. LangGraph lanza GraphRecursionError al
# superarlo, así que no hace falta un contador propio.
RECURSION_LIMIT: int = 25

# El router decide la ruta con ejemplos salidos de sesiones reales. Lo
# desconocido va a TEXTO: ante la duda se habla, no se actúa.
ROUTER_PROMPT: str = (
    "Clasificas la pregunta del usuario en UNA sola palabra:\n"
    "TEXTO: saludos, preguntas de conocimiento, opiniones, gracias.\n"
    "CALCULO: pide una cuenta aritmética concreta.\n"
    "CLIMA: pregunta por el tiempo de una ciudad.\n"
    "ARCHIVOS: pregunta qué archivos hay en data/ o qué dice un archivo.\n"
    "Responde SOLO con la palabra.\n"
    "Ejemplos:\n"
    "Pregunta: hola -> TEXTO\n"
    "Pregunta: ¿quién eres? -> TEXTO\n"
    "Pregunta: ¿qué es un grafo? -> TEXTO\n"
    "Pregunta: gracias -> TEXTO\n"
    "Pregunta: ¿acabas de consultarlo? -> TEXTO\n"
    "Pregunta: cuánto es (4+5)*3/2 -> CALCULO\n"
    "Pregunta: clima en Madrid -> CLIMA\n"
    "Pregunta: ¿lloverá hoy en Tecamac? -> CLIMA\n"
    "Pregunta: ¿qué archivos hay en data/? -> ARCHIVOS\n"
    "Pregunta: dame un resumen de langgraph -> ARCHIVOS\n"
    "Pregunta:"
)

# El hablador conoce las herramientas en texto pero no las tiene vinculadas:
# puede hablar de ellas y nunca anunciar una decisión, porque no decide nada.
HABLADOR_PROMPT: str = (
    "Eres un asistente conversacional. Respondes en español, de forma concisa "
    "y directa: tu respuesta es la conversación.\n"
    "Para datos que no puedes conocer tienes una calculadora, el clima actual "
    "mediante un servicio en línea y una búsqueda y listado de archivos en "
    "data/. Fuera del clima, no tienes acceso a internet ni a noticias en "
    "tiempo real. Si te preguntan por ellas, descríbelas; si te preguntan si "
    "consultaste algo, di la verdad según el historial.\n"
    "Ejemplo:\n"
    "Usuario: hola\n"
    "Asistente: Hola, ¿en qué puedo ayudarte?\n"
)

# El trabajador no decide si hace falta una herramienta (eso ya lo decidió el
# router): solo rellena la llamada con la chuleta de sintaxis. Si falta el
# dato no lo inventa, responde en texto pidiéndolo; la puerta de `approve`
# lo verifica de todos modos antes de molestar al humano.
TRABAJADOR_PROMPT: str = (
    "Pides la herramienta que resuelve la petición del usuario. Devuelve SOLO "
    "la llamada, sin texto.\n"
    "Calculadora: una expresión con + - * / ** % // y paréntesis. NO acepta "
    "nombres ni funciones: 'sqrt(16)' falla; escribe '16 ** 0.5'.\n"
    "Clima: el nombre de la ciudad.\n"
    "Archivos: una sola palabra que aparezca en el texto para buscar, o pide "
    "el listado si preguntan qué hay.\n"
    "Si falta el dato (por ejemplo la ciudad), no lo inventes: responde en "
    "texto pidiéndolo.\n"
    "Si la petición no necesita ningún dato externo, responde en texto.\n"
)


def mensaje_rechazo(nombres: str) -> str:
    """Fijo y terminal: lo rechazó el humano, el LLM no lo narra."""
    return (
        f"Rechazaste el uso de la herramienta necesaria: {nombres}. "
        "No se ejecutó y no tengo su resultado."
    )


def mensaje_falta_dato(faltante: str) -> str:
    """Fijo y terminal: el trabajador inventó un argumento."""
    return f"Me falta {faltante} en tu mensaje: dímelo y lo consulto."


def mensaje_fallo(motivo: str) -> str:
    """Fijo y terminal: se agotó el presupuesto de reintentos."""
    return f"La herramienta falló dos veces ({motivo}). No puedo completar tu petición."
