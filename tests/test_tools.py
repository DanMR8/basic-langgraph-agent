"""Tests de las herramientas del agente.

Ninguno requiere un LLM ni conexión a la red, salvo donde se mockea
explícitamente.
"""

import json
import urllib.error
import urllib.request

import pytest

from agent.tools import calculator, clima_aqui, ecobici, ecobici_cercanas, get_weather, search_files

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


# ─────────────────────────────────────────────────────────────────────────────
# Clima (red mockeada)
# ─────────────────────────────────────────────────────────────────────────────


CLIMA_MOCK = {
    "current_condition": [
        {
            "temp_C": "22",
            "FeelsLikeC": "20",
            "lang_es": [{"value": "Soleado"}],
            "weatherDesc": [{"value": "Sunny"}],
            "humidity": "48",
            "windspeedKmph": "10",
        }
    ]
}


class _TextoPlano:
    def __init__(self, texto: str):
        self._crudo = texto.encode()

    def read(self):
        return self._crudo

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestClima:
    def test_ciudad_con_espacios_se_codifica(self, monkeypatch):
        urls = []

        class R(_Respuesta):
            pass

        def falsa(peticion, timeout=None):
            url = peticion.full_url if hasattr(peticion, "full_url") else peticion
            urls.append(url)
            return _Respuesta(CLIMA_MOCK)

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = get_weather.invoke({"city": "centro de la cdmx"})
        assert " " not in urls[0].split("?", 1)[0]
        assert "22" in salida and "Soleado" in salida

    def test_fallo_dice_donde(self, monkeypatch):
        def rota(peticion, timeout=None):
            raise urllib.error.URLError("sin red")

        monkeypatch.setattr(urllib.request, "urlopen", rota)
        assert get_weather.invoke({"city": "Madrid"}).startswith("No pude obtener")

    def test_ubicacion_ciudad_aproximada(self, monkeypatch):
        from agent.tools import ubicacion

        def falsa(peticion, timeout=None):
            return _Respuesta(
                {"status": "success", "country": "Mexico",
                 "regionName": "México", "city": "Naucalpan",
                 "lat": 19.5, "lon": -99.2}
            )

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = ubicacion.invoke({})
        assert "Naucalpan" in salida and "aproximado por IP" in salida

    def test_ubicacion_fallo_api(self, monkeypatch):
        from agent.tools import ubicacion

        def falsa(peticion, timeout=None):
            return _Respuesta({"status": "fail", "message": "reserved range"})

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        assert ubicacion.invoke({}).startswith("No pude obtener tu ubicación")


class TestGeocodificarInverso:
    def test_barrio_y_alcaldia(self, monkeypatch):
        from agent.tools import geocodificar_inverso

        def falsa(peticion, timeout=None):
            return _Respuesta(
                {"address": {"neighbourhood": "Roma Norte", "borough": "Cuauhtémoc"}}
            )

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = geocodificar_inverso.invoke({"lat": 19.43, "lon": -99.16})
        assert salida == "Estás en Roma Norte, Cuauhtémoc (ubicación precisa)."

    def test_sin_barrio_usa_calle(self, monkeypatch):
        from agent.tools import geocodificar_inverso

        def falsa(peticion, timeout=None):
            return _Respuesta({"address": {"road": "Eje Central", "city": "CDMX"}})

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        assert "Eje Central" in geocodificar_inverso.invoke({"lat": 19.4, "lon": -99.1})

    def test_sin_datos_no_inventa_ciudad(self, monkeypatch):
        from agent.tools import geocodificar_inverso

        def falsa(peticion, timeout=None):
            return _Respuesta({"address": {}})

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = geocodificar_inverso.invoke({"lat": 19.4, "lon": -99.1})
        assert salida == "Estás por aquí (ubicación precisa)."

    def test_coords_invalidas_es_error(self):
        from agent.tools import geocodificar_inverso

        # LangChain valida tipos antes de ejecutar: el string ni llega.
        assert geocodificar_inverso.invoke({"lat": 999, "lon": 0}).startswith("Error:")

    def test_aqui_sin_ciudad_usa_ip(self, monkeypatch):
        urls = []

        def falsa(peticion, timeout=None):
            url = peticion.full_url if hasattr(peticion, "full_url") else peticion
            urls.append(url)
            return _Respuesta(CLIMA_MOCK)

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = clima_aqui.invoke({})
        assert urls[0].split("?", 1)[0].endswith("wttr.in/")
        assert "ubicación aproximada" in salida and "22" in salida

    def test_aqui_muestra_la_ciudad(self, monkeypatch):
        from agent.tools import clima_aqui

        def falsa(peticion, timeout=None):
            url = peticion.full_url if hasattr(peticion, "full_url") else peticion
            if "format=%l" in url:
                return _TextoPlano("Tlalnepantla, México")
            return _Respuesta(CLIMA_MOCK)

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = clima_aqui.invoke({})
        assert "Clima en Tlalnepantla, México (ubicación aproximada)" in salida

    def test_condicion_inglesa_se_traduce(self, monkeypatch):
        import copy

        sin_es = copy.deepcopy(CLIMA_MOCK)
        del sin_es["current_condition"][0]["lang_es"]
        sin_es["current_condition"][0]["weatherDesc"] = [{"value": "Cloudy "}]

        def falsa(peticion, timeout=None):
            url = peticion.full_url if hasattr(peticion, "full_url") else peticion
            if "format=%l" in url:
                return _TextoPlano("X")
            return _Respuesta(sin_es)

        monkeypatch.setattr(urllib.request, "urlopen", falsa)
        salida = clima_aqui.invoke({})
        assert "nublado" in salida and "Cloudy" not in salida


# ─────────────────────────────────────────────────────────────────────────────
# Ecobici cercanas (mismo mock de red, con coordenadas)
# ─────────────────────────────────────────────────────────────────────────────


INFO_CERCA = {
    "data": {
        "stations": [
            {"station_id": "0", "name": "Lealtad - Toluca", "short_name": "0",
             "lat": 19.4340, "lon": -99.1900},
            {"station_id": "1", "name": "Lealtad - Referencia", "short_name": "1",
             "lat": 19.4300, "lon": -99.1900},
            {"station_id": "2", "name": "Vecina Norte", "short_name": "2",
             "lat": 19.4330, "lon": -99.1900},
            {"station_id": "3", "name": "Cerrada Cerca", "short_name": "3",
             "lat": 19.4310, "lon": -99.1900},
            {"station_id": "4", "name": "Vacía Cerca", "short_name": "4",
             "lat": 19.4320, "lon": -99.1900},
            {"station_id": "5", "name": "Lejos Con Bicis", "short_name": "5",
             "lat": 19.4500, "lon": -99.1900},
            {"station_id": "6", "name": "Otra Lealtad Sur", "short_name": "6",
             "lat": 19.4290, "lon": -99.1900},
        ]
    }
}

ESTADO_CERCA = {
    "last_updated": 1700000000,
    "data": {
        "stations": [
            {"station_id": "0", "num_bikes_available": 0, "num_docks_available": 9,
             "is_renting": 1, "is_returning": 1},
            {"station_id": "1", "num_bikes_available": 5, "num_docks_available": 5,
             "is_renting": 1, "is_returning": 1},
            {"station_id": "2", "num_bikes_available": 7, "num_docks_available": 3,
             "is_renting": 1, "is_returning": 1},
            {"station_id": "3", "num_bikes_available": 9, "num_docks_available": 9,
             "is_renting": 0, "is_returning": 0},
            {"station_id": "4", "num_bikes_available": 0, "num_docks_available": 9,
             "is_renting": 1, "is_returning": 1},
            {"station_id": "5", "num_bikes_available": 9, "num_docks_available": 9,
             "is_renting": 1, "is_returning": 1},
            {"station_id": "6", "num_bikes_available": 2, "num_docks_available": 8,
             "is_renting": 1, "is_returning": 1},
        ]
    },
}


@pytest.fixture
def gbfs_cerca(monkeypatch):
    def falsa(peticion, timeout=None):
        url = peticion.full_url if hasattr(peticion, "full_url") else peticion
        nombre = url.rsplit("/", 1)[-1].replace(".json", "")
        return _Respuesta(
            {"station_information": INFO_CERCA, "station_status": ESTADO_CERCA}[nombre]
        )

    monkeypatch.setattr(urllib.request, "urlopen", falsa)


class TestEcobiciCercanas:
    def test_vecinas_ordenadas_y_referencia_a_cero(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "lealtad"})
        assert "- Referencia: Lealtad - Referencia (1), con 5 bicis" in salida
        assert "la de más bicis entre 3 coincidencia(s)" in salida
        assert salida.index("| 111 m |") < salida.index("| 334 m |")
        assert "Vecina Norte" in salida and "Otra Lealtad Sur" in salida

    def test_referencia_no_se_lista_como_hallazgo(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "lealtad"})
        assert "| Lealtad - Referencia (1) | 0 m |" not in salida

    def test_referencia_con_bicis_sin_vecinas(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "norte", "radio_m": 50})
        assert "Sin otras estaciones con bicis en 50 m." in salida

    def test_excluye_cerrada_vacia_y_lejana(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "lealtad"})
        assert "Cerrada Cerca" not in salida
        assert "Vacía Cerca" not in salida
        assert "Lejos Con Bicis" not in salida

    def test_sin_bicis_en_radio(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "vacia", "radio_m": 100})
        assert "No hay bicis en 100 m alrededor" in salida

    def test_radio_invalido_es_error(self, gbfs_cerca):
        assert ecobici_cercanas.invoke({"estacion": "lealtad", "radio_m": 0}).startswith("Error:")
        assert ecobici_cercanas.invoke({"estacion": "lealtad", "radio_m": -5}).startswith("Error:")

    def test_sin_referencia_guia(self, gbfs_cerca):
        salida = ecobici_cercanas.invoke({"estacion": "zzz"})
        assert "No encontré estación" in salida and "calle donde está" in salida
