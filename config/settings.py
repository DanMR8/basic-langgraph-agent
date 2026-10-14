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
    "CLIMA: pregunta por el tiempo (de una ciudad o de aquí).\n"
    "MOVILIDAD: pregunta por bicis o espacios de una cicloestación Ecobici.\n"
    "UBICACION: pregunta dónde está, en qué ciudad está o cuál es su ubicación.\n"
    "Responde SOLO con la palabra.\n"
    "Ejemplos:\n"
    "Pregunta: hola -> TEXTO\n"
    "Pregunta: ¿quién eres? -> TEXTO\n"
    "Pregunta: ¿qué es un grafo? -> TEXTO\n"
    "Pregunta: gracias -> TEXTO\n"
    "Pregunta: ¿acabas de consultarlo? -> TEXTO\n"
    "Pregunta: cuánto es (4+5)*3/2 -> CALCULO\n"
    "Pregunta: dame un script de algoritmo burbuja -> TEXTO\n"
    "Pregunta: ahora de quicksort -> TEXTO\n"
    "Pregunta: ordena esta lista [3,1,2] -> TEXTO\n"
    "Pregunta: clima en Madrid -> CLIMA\n"
    "Pregunta: ¿lloverá hoy en Tecamac? -> CLIMA\n"
    "Pregunta: ¿qué clima hace aquí? -> CLIMA\n"
    "Pregunta: ¿lloverá donde estoy? -> CLIMA\n"
    "Pregunta: ¿hay bicis en reforma? -> MOVILIDAD\n"
    "Pregunta: ¿tiene espacios la 261? -> MOVILIDAD\n"
    "Pregunta: y que tal cerca del monumento a la madre? -> MOVILIDAD\n"
    "Pregunta: ¿y en Insurgentes? -> MOVILIDAD\n"
    "Pregunta: en 5 de mayo -> MOVILIDAD\n"
    "Pregunta: que hay de los olivos -> MOVILIDAD\n"
    "Pregunta: 20 de noviembre -> MOVILIDAD\n"
    "Pregunta: ¿hay bicis cerca de la Lealtad? -> MOVILIDAD\n"
    "Pregunta: bicis cerca de mí -> MOVILIDAD\n"
    "Pregunta: ¿dónde estoy? -> UBICACION\n"
    "Pregunta: ¿en qué ciudad estoy? -> UBICACION\n"
    "Pregunta:"
)

# El hablador conoce las herramientas en texto pero no las tiene vinculadas:
# puede hablar de ellas y nunca anunciar una decisión, porque no decide nada.
HABLADOR_PROMPT: str = (
    "Eres un asistente conversacional. Respondes en español, de forma concisa "
    "y directa: tu respuesta es la conversación.\n"
    "Para datos que no puedes conocer tienes una calculadora, el clima actual "
    "mediante un servicio en línea, la disponibilidad de bicis Ecobici por "
    "estación y tu ciudad aproximada si preguntas dónde estás. Fuera de eso, "
    "no tienes acceso a internet ni a noticias en tiempo real, y no consultas "
    "archivos. Si te preguntan por ellas, descríbelas; si te preguntan si "
    "consultaste algo, di la verdad según el historial.\n"
    "Ejemplo:\n"
    "Usuario: hola\n"
    "Asistente: Hola, ¿en qué puedo ayudarte?\n"
    "Usuario: ¿qué sabes sobre ecobici?\n"
    "Asistente: Es el sistema de bicis compartidas de CDMX, y puedo consultar "
    "la disponibilidad en tiempo real si me das una estación.\n"
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
    "Código, scripts o explicaciones van en texto aunque mencionen números "
    "o listas: la calculadora solo evalúa cuentas aritméticas explícitas, "
    "nunca programa ni ordena nada.\n"
    "Clima: el nombre del lugar con `get_weather` (ciudad, alcaldía, colonia: "
    "tal cual lo escriban); si piden aquí, donde están o en su ubicación sin "
    "nombrar lugar, `clima_aqui` (sin argumentos).\n"
    "Ecobici: la palabra o número de estación tal como lo escribió el "
    "usuario, sin inventar nombres, y conclusion en False: tú redactas el "
    "resultado y la conclusión determinista solo estorba.\n"
    "Si piden 'cerca de', usa ecobici con la referencia tal cual: las "
    "cercanas solo llegan como segundo eslabón ofrecido, nunca las pidas "
    "tú directamente.\n"
    "Si falta el dato (por ejemplo la ciudad), no lo inventes: responde en "
    "texto pidiéndolo.\n"
    "Si la petición no necesita ningún dato externo, responde en texto.\n"
    "Al redactar un resultado responde en una frase completa y natural con "
    "los datos de la tabla, sin copiarla entera. Redacta EL resultado "
    "obtenido, no el formato del ejemplo: cada herramienta tiene el suyo. "
    "Ejemplos:\n"
    "Tabla: | CE-507 Mar Mediterráneo - Azcapotzalco (507) | 11 | 20 | abierta |\n"
    "Redacción: Hay 11 bicis en CE-507 Mar Mediterráneo (507), con 20 "
    "espacios libres.\n"
    "Resultado: Clima en Madrid: Temperatura: 17°C (sensación 12°C), "
    "Condición: Clear.\n"
    "Redacción: En Madrid hay 17°C despejado, con sensación de 12°C.\n"
    "Resultado: Estás en Naucalpan, México (Mexico), aproximado por IP.\n"
    "Redacción: Estás en Naucalpan, México (aproximado por IP).\n"
    "Usa los números tal como los devuelve la herramienta: si es 0, dilo "
    "sin rodeos.\n"
    "Si la herramienta no encontró coincidencias, di que no encontraste con "
    "ese nombre; nunca digas que no hay en la zona.\n"
    "Redacta SOLO el último resultado; ignora los temas anteriores de la "
    "conversación.\n"
    "Si se rechazó revisar cercanas, redacta solo la tabla original: jamás "
    "conviertas bicis en metros ni afirmes cercanía.\n"
    "Si pediste cercanía, la tabla trae distancias medidas: reporta la "
    "referencia declarada y las vecinas en metros tal cual. Solo afirma "
    "cercanía con metros medidos; nunca preguntes por ciudad (Ecobici no "
    "usa ese dato).\n"
)


def mensaje_rechazo(nombres: str) -> str:
    """Fijo y terminal: lo rechazó el humano, el LLM no lo narra."""
    return (
        f"Rechazaste el uso de la herramienta necesaria: {nombres}. "
        "No se ejecutó y no tengo su resultado."
    )


# Frase de ubicación: el detector (regex, sin LLM) encuentra peticiones como
# "cerca de mí", y esta instrucción redacta la frase con tono natural. Lleva
# backstop en el nodo: si el modelo añade de más (bicis, estaciones, cifras),
# se descarta y va el fijo. El LLM adjetiva, lo determinista decide.
FRASE_UBICACION_PROMPT: str = (
    "Escribes UNA sola frase para un asistente: que no tiene acceso a la "
    "ubicación del usuario. Devuelve solo esa frase, sin saludar, sin ofrecer "
    "nada y sin mencionar bicis, estaciones ni disponibilidad.\n"
    "Ejemplo:\n"
    "Petición: hay bicis cerca de mí\n"
    "Frase: No tengo acceso a tu ubicación.\n"
    "Petición:"
)


def mensaje_falta_dato(faltante: str) -> str:
    """Fijo y terminal: el trabajador inventó un argumento.

    No basta con decir que falta: dice qué dar y con qué formato, para que
    el reintento salga bien a la primera.
    """
    guia = {
        "la ciudad": (
            "Me falta la ciudad en tu mensaje: dime de qué ciudad quieres "
            "el clima (p. ej. 'Coyoacán')."
        ),
        "la estación": (
            "Me falta una referencia en tu mensaje: dime la calle donde está "
            "(p. ej. 'Reforma', '20 de Noviembre' o 'Insurgentes') o el número "
            "de la cicloestación (p. ej. '261')."
        ),
    }
    return guia.get(faltante, f"Me falta {faltante} en tu mensaje: dímelo y lo consulto.")


def mensaje_fallo(motivo: str) -> str:
    """Fijo y terminal: se agotó el presupuesto de reintentos."""
    return (
        f"La herramienta falló dos veces ({motivo}). "
        "Intenta de nuevo en un momento o reformula la petición."
    )
