"""Tests de las herramientas del agente.

Ninguno requiere un LLM ni conexión a la red, salvo donde se mockea
explícitamente.
"""

import json
import urllib.error
import urllib.request

import pytest

from agent.tools import calculator, ecobici, search_files

# ─────────────────────────────────────────────────────────────────────────────
# Calculadora
# ─────────────────────────────────────────────────────────────────────────────


class TestCalculator:
    def test_operacion_basica(self):
        assert calculator.invoke({"expression": "2 + 2"}) == "Resultado: 4"

    def test_precedencia_de_operadores(self):
        assert calculator.invoke({"expression": "(4 + 5) * 3 / 2"}) == "Resultado: 13.5"

    def test_potencia(self):
        assert calculator.invoke({"expression": "2 ** 8"}) == "Resultado: 256"

    def test_division_por_cero(self):
        assert "división entre cero" in calculator.invoke({"expression": "1 / 0"})

    def test_rechaza_potencia_encadenada(self):
        """`9**9**9` colgaba el proceso antes; ahora se rechaza."""
        resultado = calculator.invoke({"expression": "9 ** 9 ** 9"})
        assert "exponente" in resultado

    def test_rechaza_resultado_enorme(self):
        assert "demasiado grande" in calculator.invoke({"expression": "(10**100) ** 100"})

    def test_rechaza_desbordamiento_a_infinito(self):
        assert "demasiado grande" in calculator.invoke({"expression": "1e308 * 10"})

    def test_acepta_operadores_completos(self):
        assert calculator.invoke({"expression": "10 % 3"}) == "Resultado: 1"
        assert calculator.invoke({"expression": "7 // 2"}) == "Resultado: 3"
        assert calculator.invoke({"expression": "-5 + 3"}) == "Resultado: -2"
        assert calculator.invoke({"expression": "2 ** 0.5"}).startswith("Resultado: 1.41")

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",
            "open('secreto.txt')",
            "(1).__class__",
            "[x for x in range(3)]",
            "lambda: 1",
            "a + 1",
            "1; 2",
            "",
        ],
    )
    def test_rechaza_entrada_no_segura(self, expression):
        resultado = calculator.invoke({"expression": expression})
        assert resultado.startswith("Error"), resultado

    def test_no_puede_leer_el_disco(self):
        """El sandbox no debe permitir I/O aunque la expresión lo pida."""
        resultado = calculator.invoke({"expression": "open('C:/Windows/win.ini').read()"})
        assert resultado.startswith("Error")


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda en archivos
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchFiles:
    @pytest.fixture
    def data_dir(self, tmp_path, monkeypatch):
        """DATA_DIR temporal con .txt, .md en subcarpeta y un binario."""
        (tmp_path / "notas.txt").write_text(
            "LangGraph es un orquestador de grafos\nOllama corre modelos locales",
            encoding="utf-8",
        )
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "guia.md").write_text(
            "# Guia\nUn conditional edge decide la ruta", encoding="utf-8"
        )
        (tmp_path / "ignorado.bin").write_text("LangGraph", encoding="utf-8")
        monkeypatch.setattr("agent.tools.DATA_DIR", tmp_path)
        return tmp_path

    def test_encuentra_en_txt(self, data_dir):
        assert "notas.txt:1" in search_files.invoke({"query": "LangGraph"})

    def test_recorre_subdirectorios(self, data_dir):
        assert "guia.md" in search_files.invoke({"query": "conditional edge"})

    def test_ignora_extensiones_no_texto(self, data_dir):
        assert "ignorado.bin" not in search_files.invoke({"query": "LangGraph"})

    def test_case_insensitive(self, data_dir):
        assert "coincidencia" in search_files.invoke({"query": "langgraph"})

    def test_sin_coincidencias(self, data_dir):
        assert "No se encontró" in search_files.invoke({"query": "inexistente"})

    def test_directorio_inexistente(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent.tools.DATA_DIR", tmp_path / "no-existe")
        assert "No hay archivos" in search_files.invoke({"query": "x"})


# ─────────────────────────────────────────────────────────────────────────────
# Ecobici (GBFS público, red mockeada)
# ─────────────────────────────────────────────────────────────────────────────


INFO_MOCK = {
    "data": {
        "stations": [
            {"station_id": "1", "name": "CE-710 Molino del Rey", "short_name": "710"},
            {"station_id": "2", "name": "Rubén Darío - Arquímedes", "short_name": "102"},
            {"station_id": "3", "name": "Reforma - Río Guadalquivir", "short_name": "13"},
            {"station_id": "4", "name": "Reforma - Insurgentes", "short_name": "27"},
            {"station_id": "5", "name": "Reforma - Sevilla", "short_name": "31"},
            {"station_id": "6", "name": "Reforma - Eje 2", "short_name": "45"},
            {"station_id": "7", "name": "Reforma - Juárez", "short_name": "52"},
            {"station_id": "8", "name": "Reforma - Bucareli", "short_name": "63"},
        ]
    }
}

ESTADO_MOCK = {
    "last_updated": 1700000000,
    "data": {
        "stations": [
            {
                "station_id": "1",
                "num_bikes_available": 11,
                "num_docks_available": 27,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "2",
                "num_bikes_available": 0,
                "num_docks_available": 20,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "3",
                "num_bikes_available": 8,
                "num_docks_available": 12,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "4",
                "num_bikes_available": 0,
                "num_docks_available": 20,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "5",
                "num_bikes_available": 0,
                "num_docks_available": 0,
                "is_renting": 0,
                "is_returning": 0,
            },
            {
                "station_id": "6",
                "num_bikes_available": 3,
                "num_docks_available": 5,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "7",
                "num_bikes_available": 1,
                "num_docks_available": 9,
                "is_renting": 1,
                "is_returning": 1,
            },
            {
                "station_id": "8",
                "num_bikes_available": 5,
                "num_docks_available": 2,
                "is_renting": 1,
                "is_returning": 1,
            },
        ]
    },
}


class _Respuesta:
    def __init__(self, datos):
        self._crudo = json.dumps(datos).encode()

    def read(self):
        return self._crudo

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def gbfs_mock(monkeypatch):
    def falsa(peticion, timeout=None):
        url = peticion.full_url if hasattr(peticion, "full_url") else peticion
        nombre = url.rsplit("/", 1)[-1].replace(".json", "")
        return _Respuesta({"station_information": INFO_MOCK, "station_status": ESTADO_MOCK}[nombre])

    monkeypatch.setattr(urllib.request, "urlopen", falsa)


class TestEcobici:
    def test_encuentra_por_nombre(self, gbfs_mock):
        salida = ecobici.invoke({"estacion": "molino", "conclusion": True})
        assert "Molino del Rey" in salida and "| 11 | 27 |" in salida
        assert "- 1 coincidencia, top 5 por bicis:" in salida
        assert "- Conclusión: hay bicis en 1 de 1." in salida

    def test_sin_conclusion_a_pedido(self, gbfs_mock):
        salida = ecobici.invoke({"estacion": "molino", "conclusion": False})
        assert "| 11 | 27 |" in salida and "Conclusión" not in salida

    def test_encuentra_por_numero(self, gbfs_mock):
        assert "Molino del Rey" in ecobici.invoke({"estacion": "710"})

    def test_insensible_a_acentos_y_mayusculas(self, gbfs_mock):
        assert "Rubén Darío" in ecobici.invoke({"estacion": "RUBEN"})

    def test_sin_bicis_pero_abierta(self, gbfs_mock):
        salida = ecobici.invoke({"estacion": "102", "conclusion": True})
        assert "| 0 | 20 |" in salida
        assert "- Conclusión: sin bicis en las coincidencias." in salida

    def test_sin_servicio_lo_dice(self, gbfs_mock):
        assert "sin servicio" in ecobici.invoke({"estacion": "sevilla"})

    def test_varias_ordenadas_por_bicis_y_topadas(self, gbfs_mock):
        salida = ecobici.invoke({"estacion": "reforma"})
        assert "…y 1 más" in salida
        assert salida.index("| 8 |") < salida.index("| 5 |") < salida.index("| 3 |")

    def test_sin_coincidencias_guia_reintento(self, gbfs_mock):
        salida = ecobici.invoke({"estacion": "zzz"})
        assert "No encontré estación" in salida
        assert "calle donde está" in salida

    def test_vacia_es_error(self, gbfs_mock):
        assert ecobici.invoke({"estacion": "  "}).startswith("Error:")

    def test_error_de_red_es_error(self, monkeypatch):
        def rota(peticion, timeout=None):
            raise urllib.error.URLError("sin red")

        monkeypatch.setattr(urllib.request, "urlopen", rota)
        assert ecobici.invoke({"estacion": "reforma"}).startswith("Error:")
