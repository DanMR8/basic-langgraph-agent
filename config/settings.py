"""Configuración centralizada del agente."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Modelo ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Temperatura por nodo. Cada uno hace un trabajo distinto, así que no comparte
# un único valor:
#   - `conversar` y `responder` hablan con el usuario. A 0 el de 8B devolvía
#     literalmente la misma frase en cada turno.
#   - `validar` clasifica SI/NO a partir de ejemplos. Ahí la reproducibilidad
#     importa más que la gracia, así que se queda en 0.
# Medido con llama3.1:8b sobre la misma sesión de 6 turnos, contando cuántas
# sesiones se comieron JSON de herramienta crudo por texto
# (`{"name": "Hola", "parameters": {}}` en vez de llamarla):
#     conversar=0.7 -> 2 de 8    conversar=0.5 -> 0 de 10    conversar=0.3 -> 0 de 8
# El riesgo es solo de `conversar`, el único con herramientas vinculadas;
# `responder` no puede pedir ninguna, así que se queda en 0.7 sin exponerse.
TEMPERATURA_CONVERSAR: float = 0.5
TEMPERATURA_RESPONDER: float = 0.7
TEMPERATURA_VALIDADOR: float = 0.0

# ── Datos ───────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Agente ──────────────────────────────────────────────────────────────────
# Límite de pasos del grafo por turno. LangGraph lanza GraphRecursionError al
# superarlo, así que no hace falta un contador propio.
RECURSION_LIMIT: int = 25

# El marco importa: el prompt anterior decía tres veces "Usa la herramienta de
# X" y nunca decía que se pudiera contestar en texto, así que el modelo pedía
# herramientas por defecto. Los ejemplos son los que hacen que el modelo de 8B
# acierte: la sola instrucción, en negativo o en positivo, no bastaba.
#
# Tampoco sirve que los ejemplos sean instrucciones de enrutado
# ("hola -> responde en texto"): el modelo se las tomó como plantilla de
# respuesta y empezó a anunciar su decisión cada vez que declinaba usar una
# herramienta: "No hay necesidad de llamar a una función para responder a esa
# pregunta. La respuesta es: ...". Prohibirle la frase con la frase escrita
# no tuvo efecto.
#
# Medido con llama3.1:8b, 8 rondas sobre cada una de dos sesiones y el mismo
# detector en ambos casos:
#     ejemplos de enrutado (original)  -> 26/80 respuestas con la frase
#     diálogo de ejemplo, sin marco binario -> 17/80
# Baja de una tercera parte a una quinta, pero no desaparece. Lo que no probó:
# quitarle la frase al texto ya escrito, que es lo único determinista.
SYSTEM_PROMPT: str = (
    "Eres un asistente conversacional. Respondes en español, de forma concisa "
    "y sin preámbulos: tu respuesta es la conversación.\n"
    "Tienes una calculadora, un servicio de clima y una búsqueda en archivos "
    "en data/ para cuando el usuario necesite consultar uno de esos datos; si "
    "te pide que la uses, hazlo.\n"
    "Ejemplo de conversación:\n"
    "Usuario: hola\n"
    "Asistente: Hola, ¿en qué puedo ayudarte?\n"
    "Usuario: ¿quién eres?\n"
    "Asistente: Soy un asistente conversacional.\n"
    "Usuario: ¿qué es un grafo de estado?\n"
    "Asistente: Un modelo que describe un sistema mediante estados y las "
    "transiciones entre ellos.\n"
    "Usuario: gracias por la ayuda\n"
    "Asistente: De nada, a ti.\n"
    "Usuario: ¿acabas de consultarlo verdad?\n"
    "Asistente: No, no lo he consultado.\n"
)

# Se inyecta en el system prompt de `responder` solo cuando el turno cierra con
# un veto, y no se guarda en ningún sitio: lo que se escribe en el historial
# persiste y el modelo lo copia en su respuesta. Medido con llama3.1:8b, una
# orden metida en el ToolMessage de veto salía en la boca del asistente tres
# turnos después, y a partir de ahí se repetía sola.
INSTRUCCION_VETO: str = (
    "Una herramienta que querías usar fue descartada: no se ejecutó y no tienes "
    "ningún dato de ella. No digas que la consultaste ni te inventes su "
    "resultado. Contesta en texto con lo que sí sabes, y si no puedes "
    "responder, dilo."
)

# El modelo con herramientas NO decide bien cuándo usarlas: medido con
# llama3.1:8b, pide `calculator(expression='hola')` ante un saludo. La decisión
# no se le puede delegar, así que se veta cada petición antes de enseñársela al
# humano. Ve la propuesta concreta, no solo la pregunta, y eso le basta para
# descartar un argumento que no encaja. Los ejemplos salen de llamadas reales.
VALIDADOR_PROMPT: str = (
    "Vetas peticiones de herramientas de un asistente. Te llega una pregunta "
    "del usuario y la herramienta que el asistente quiere usar con sus "
    "argumentos. Responde con UNA sola palabra: SI o NO.\n"
    "SI: la herramienta aporta un dato que el asistente no puede conocer: una "
    "cuenta exacta, el tiempo actual de una ciudad o el contenido de un archivo "
    "de data/.\n"
    "NO: la pregunta se responde en texto, o el argumento no encaja con la "
    "herramienta.\n"
    "Ejemplos:\n"
    "Pregunta: hola | Propuesta: calculator({'expression': 'hola'}) -> NO\n"
    "Pregunta: gracias | Propuesta: search_files({'query': 'gracias'}) -> NO\n"
    "Pregunta: ¿qué es un grafo? | Propuesta: search_files({'query': 'grafo'}) -> NO\n"
    "Pregunta: acabas de consultarlo | Propuesta: search_files({'query': 'consultarlo'}) -> NO\n"
    "Pregunta: ¿y eso? | Propuesta: calculator({'expression': '¿y eso?'}) -> NO\n"
    "Pregunta: cuánto es (4+5)*3/2 | Propuesta: calculator({'expression': '(4+5)*3/2'}) -> SI\n"
    "Pregunta: clima en Madrid | Propuesta: get_weather({'city': 'Madrid'}) -> SI\n"
    "Pregunta: archivo sobre grafos | Propuesta: search_files({'query': 'grafos'}) -> SI\n"
    "Respuesta:"
)
