# Makefile — Pipeline OSM → Sionna RT
# Uso: make [all | install | setup | scene | test | clean]
#
# Antes de rodar pela primeira vez:
#   1. Edite BLOSM_ZIP abaixo (ou exporte a variável de ambiente)
#   2. make all

# ─── Carrega variáveis do .env se existir ─────────────────────────────────────
-include .env

# ─── Variáveis configuráveis ──────────────────────────────────────────────────
BLENDER_BIN   ?= $(HOME)/tools/blender-3.6/blender
BLOSM_ZIP     ?= $(CURDIR)/blosm.zip
MITSUBA_ZIP   ?= $(CURDIR)/addons/mitsuba-blender.zip
VENV_DIR      ?= $(CURDIR)/venv
VENV_PYTHON    = $(VENV_DIR)/bin/python

# Bounding box padrão — Interlagos, SP
MIN_LAT ?= -23.7064
MAX_LAT ?= -23.6974
MIN_LON ?= -46.7064
MAX_LON ?= -46.6954

# Arquivo de saída
OUTPUT_XML ?= $(CURDIR)/output/interlagos.xml

# ─── Targets ──────────────────────────────────────────────────────────────────
.PHONY: all install setup scene test clean check-blender check-venv \
        inspect gnb smoke-gnb all-gnb

## all: Executa install → setup → scene → test em sequência
all: install setup scene test

## inspect: Inspeciona geometria e materiais da cena exportada
inspect: check-venv
	@echo "=== Inspeção da cena ==="
	"$(VENV_PYTHON)" scripts/inspect_scene.py

## gnb: Converte GPS da gNB para ENU e salva output/gnb_config.pkl
gnb: check-venv
	@echo "=== Posicionamento da gNB ==="
	"$(VENV_PYTHON)" scripts/place_gnb.py

## smoke-gnb: PathSolver + RadioMap com a gNB; salva PNGs em output/
smoke-gnb: check-venv
	@echo "=== Smoke test gNB ==="
	"$(VENV_PYTHON)" scripts/smoke_test_gnb.py

## all-gnb: inspect → gnb → smoke-gnb em sequência
all-gnb: inspect gnb smoke-gnb

## install: Baixa Blender 3.6 LTS, mitsuba-blender e cria o venv Python
install:
	@echo "=== [1/4] Instalação ==="
	BLOSM_ZIP="$(BLOSM_ZIP)" bash install.sh

## setup: Habilita os add-ons Blosm e Mitsuba no Blender headless
setup: check-blender
	@echo "=== [2/4] Setup dos add-ons no Blender ==="
	@if [ ! -f "$(BLOSM_ZIP)" ]; then \
		echo "[ERRO] Blosm ZIP não encontrado: $(BLOSM_ZIP)"; \
		echo "       Defina BLOSM_ZIP=/caminho/correto e tente novamente."; \
		exit 1; \
	fi
	BLOSM_ZIP="$(BLOSM_ZIP)" MITSUBA_ZIP="$(MITSUBA_ZIP)" \
		"$(BLENDER_BIN)" -b -P setup_blender.py

## scene: Importa OSM, aplica materiais ITU e exporta output/interlagos.xml
scene: check-blender
	@echo "=== [3/4] Construção da cena ==="
	"$(BLENDER_BIN)" -b -P build_scene.py -- $(MIN_LAT) $(MAX_LAT) $(MIN_LON) $(MAX_LON)
	@echo "Arquivo gerado: $(OUTPUT_XML)"

## test: Roda smoke_test.py no venv (carrega XML, PathSolver, salva preview.png)
test: check-venv
	@echo "=== [4/4] Smoke test Sionna RT ==="
	"$(VENV_PYTHON)" smoke_test.py \
		--xml "$(OUTPUT_XML)" \
		--preview "$(CURDIR)/output/preview.png"

## clean: Remove artefatos gerados (output/, osm_data/); preserva venv e ~/tools
clean:
	@echo "Removendo output/ e osm_data/ ..."
	rm -rf output/ osm_data/
	@echo "Pronto. Para reinstalar tudo: make all"

## clean-all: Remove também venv e addons baixados (não toca em ~/tools/blender-3.6)
clean-all: clean
	rm -rf venv/ addons/ .env
	@echo "Removido venv/, addons/ e .env."

# ─── Guards ───────────────────────────────────────────────────────────────────
check-blender:
	@if [ ! -x "$(BLENDER_BIN)" ]; then \
		echo "[ERRO] Blender não encontrado em: $(BLENDER_BIN)"; \
		echo "       Execute 'make install' primeiro."; \
		exit 1; \
	fi

check-venv:
	@if [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "[ERRO] venv Python não encontrado em: $(VENV_DIR)"; \
		echo "       Execute 'make install' primeiro."; \
		exit 1; \
	fi

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Pipeline OSM → Sionna RT"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  make /'
	@echo ""
	@echo "Variáveis principais:"
	@echo "  BLENDER_BIN = $(BLENDER_BIN)"
	@echo "  BLOSM_ZIP   = $(BLOSM_ZIP)"
	@echo "  VENV_DIR    = $(VENV_DIR)"
	@echo "  OUTPUT_XML  = $(OUTPUT_XML)"
	@echo ""
