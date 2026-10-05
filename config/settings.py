"""Configuración centralizada del agente."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Modelo ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Datos ───────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Agente ──────────────────────────────────────────────────────────────────
# Límite de pasos del grafo por turno. LangGraph lanza GraphRecursionError al
# superarlo, así que no hace falta un contador propio.
RECURSION_LIMIT: int = 25
SYSTEM_PROMPT: str = (
    "Eres un asistente útil con acceso a herramientas. "
    # ── EXPERIMENTO ────────────────────────────────────────────────────────
    # Descomenta estas 3 líneas y reinicia. Pretende evitar que el modelo pida
    # herramientas para preguntas conversacionales (hoy dispara search_files).
    # Probado con llama3.1:8b sin efecto, aquí y al final del prompt: puede
    # que sí funcione con un modelo más grande.
    # "Si la pregunta no requiere datos externos ni una herramienta, "
    # "contesta en texto directamente y no llames a ninguna. "
    # ───────────────────────────────────────────────────────────────────────
    "Usa la herramienta de cálculo para operaciones matemáticas. "
    "Usa la herramienta de clima para preguntas sobre el clima. "
    "Usa la herramienta de búsqueda en archivos para encontrar información "
    "en documentos locales. "
    "Responde siempre en español de forma concisa."
)
