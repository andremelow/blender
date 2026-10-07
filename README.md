# Interlagos RT — Pipeline OSM → Blender → Sionna RT para localização de UEs

Simulação de propagação por ray tracing (Sionna RT 2.0.1) no entorno do
Autódromo José Carlos Pace (Interlagos, São Paulo), com uma gNB a 3.5 GHz
e milhares de UEs. O projeto gera medidas de RSRP, AoA e delay por UE,
avalia estimadores de posição (RSRP, ToA) e compara mapas de densidade de
usuários obtidos por ray tracing contra modelos analíticos.

## Visão geral

```
OpenStreetMap ──Blosm──▶ Blender 3.6 ──mitsuba-blender──▶ interlagos.xml (Mitsuba)
                                                                │
                                                                ▼
                     gNB (GPS → ENU, 62 m AGL, UPA 8×8 TR38901) + UEs (grid / espectadores)
                                                                │
                                                     Sionna RT PathSolver
                                                                │
                              ┌─────────────────────────────────┼───────────────────────┐
                              ▼                                 ▼                       ▼
                   measurements_*.npz / .mat           delays_v2.npz            figuras em output/
                   (RSRP, AoA, path_type, n_paths)     (tau_dom, tau_min)
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
     Estimadores de     Calibração de       MATLAB: densidade de UEs
     posição (RSRP/ToA) path loss           RT vs analítico
```

### Parâmetros do cenário de referência

| Parâmetro | Valor |
|---|---|
| Origem da cena (ENU) | lat −23.7019, lon −46.7009 |
| Bounding box OSM | raio 1200 m em torno da origem |
| gNB | lat −23.701926, lon −46.700974, 62 m AGL (torre treliçada sobre prédio de 2 pav.) |
| Antena gNB | UPA 8×8, λ/2, padrão TR 38.901, polarização VH |
| Potência / frequência | 23 dBm (200 mW), 3.5 GHz (NR n78), 100 MHz |
| UEs (baseline) | grade 41×41 = 1681 pontos, ±400 m, passo 20 m, 1.5 m de altura |
| UEs (F1) | 10 000 espectadores distribuídos pelos setores do autódromo |
| PathSolver | max_depth 3, LoS + reflexão especular + difração, sem espalhamento difuso |
| Materiais | ITU concreto (ε_r 5.24, σ 0.123 S/m) |

## Requisitos

- Linux com `bash`, `make`, `curl`/`wget` e Python 3.11+ (o venv atual usa 3.13)
- GPU opcional (Sionna RT roda em CPU via DrJit, mais lento)
- Blender 3.6 LTS (baixado automaticamente para `~/tools/blender-3.6`)
- Add-on **Blosm** (importador OSM). Não é redistribuído aqui: baixe o zip e
  coloque-o em `blosm.zip` na raiz ou aponte `BLOSM_ZIP` para ele
- Add-on **mitsuba-blender** (baixado por `install.sh` para `addons/`)
- MATLAB (opcional) para os scripts de densidade em `MATLAB/`

## Instalação e primeira execução

```bash
# 1. Instala Blender, mitsuba-blender e cria ./venv com Sionna RT
BLOSM_ZIP=/caminho/para/blosm.zip make install

# 2. Habilita os add-ons, importa OSM, exporta o XML e roda o smoke test
make setup scene test
```

`make all` executa as quatro etapas em sequência e é idempotente.
`install.sh` grava um `.env` com `BLENDER_BIN`, `BLOSM_ZIP`, `MITSUBA_ZIP`
e `VENV_DIR`, que o Makefile carrega automaticamente.

O bounding box pode ser alterado via variáveis `MIN_LAT`, `MAX_LAT`,
`MIN_LON`, `MAX_LON` do Makefile ou passando as coordenadas diretamente a
`build_scene.py`.

## Fluxo de trabalho

### 1. gNB e grade de UEs (targets do Makefile)

```bash
make inspect        # geometria e materiais da cena exportada
make gnb            # GPS → ENU, salva output/gnb_config.pkl
make smoke-gnb      # PathSolver + RadioMap com a gNB, gera PNGs
make ue-grid        # grade 41×41 filtrada por cobertura → output/ue_grid.npz
make simulate       # 100 primeiros UEs (validação rápida)
make simulate-full  # todos os 1681 UEs (~40 s em GPU) → output/measurements.npz
make plot-meas      # figura de diagnóstico
```

### 2. Gerador de cenários por YAML (`generators/`)

Substitui os scripts `generate_ue_grid.py` + `simulate_ues.py` por uma
configuração declarativa validada. Os cenários ficam em `scenarios/`:

| Arquivo | Descrição |
|---|---|
| `schema.yaml` | Documentação dos campos aceitos |
| `interlagos_baseline.yaml` | Reproduz a grade 41×41 do pipeline original |
| `interlagos_f1.yaml` | 10 000 espectadores do GP de F1 (posições pré-computadas) |

```bash
source venv/bin/activate

# Baseline: grade de UEs
python -m generators.generate scenarios/interlagos_baseline.yaml
python scripts/validate_r4.py            # confere o gerador contra measurements_v2.npz

# Cenário F1: espectadores nos setores do autódromo
python scripts/generate_spectator_ues.py --n-ues 10000 --seed 42
python -m generators.generate scenarios/interlagos_f1.yaml
python scripts/plot_spectators.py        # output/spectators_map.png
```

Opções úteis de `generators.generate`: `--dry-run`, `--limit N`,
`--resume`, `--checkpoint ARQ`, `--save-gnb-pkl`.

O arquivo `tools/sector_editor.html` é um editor visual (abra no navegador)
para desenhar e ajustar os polígonos dos setores de espectadores.

### 3. Análise das medidas e estimadores de posição (`scripts/`)

Os scripts abaixo leem `output/measurements_v2.npz` e gravam figuras e
tabelas em `output/`. Execute na ordem indicada a partir da raiz do projeto.

| Script | O que faz | Saída |
|---|---|---|
| `relabel_paths.py` | Reclassifica `path_type` do caminho dominante (LoS / diffracted / reflected / transmitted) | `measurements_v2.npz/.mat` |
| `plot_measurements_v2.py` | Diagnóstico visual: RSRP, tipo de caminho, erro AoA | `measurements_diag_v2.png` |
| `diagnose_refracted.py` | Auditoria de materiais e dos caminhos refratados | `material_audit.txt` |
| `summary_stats.py` | Tabela resumo por categoria | `summary_stats.md` |
| `analyze_bad_regime.py` | Regime de AoA ruim (reflected + diffracted) | `bad_regime_analysis.png` |
| `shadow_geometry.py` | Obstrução gNB→UE pelos footprints dos prédios | `shadow_geometry.png` |
| `calibrate_pathloss.py` | Regressão `RSRP = A − 10·n·log10(d)` por categoria | `pathloss_calibrated.json/.png` |
| `reestimate_positions.py` | Re-estimativa com path loss calibrado (oracle vs global) | `cdf_pos_error_calibrated.png` |
| `decompose_error.py` | Erro de posição em componentes radial e tangencial | `error_decomposition.md/.png` |
| `rsrp_compensated.py` | Compensa o ganho do elemento TR 38.901 antes da calibração | `pathloss_compensated.json`, `rsrp_compensation.png` |
| `extract_delays.py` | Re-executa o PathSolver para extrair `tau_dom` e `tau_min` | `delays_v2.npz` |
| `toa_estimator.py` | Estimador ToA (caminho dominante e primeira chegada) | `toa_estimator.png`, `toa_summary.md` |
| `final_position_estimation.py` | Figura-matriz comparando todos os estimadores | `final_position_estimation.png` |
| `export_to_matlab.py` | Exporta NPZ para `.mat` (`--source grid` ou `--source spectators`) | `measurements_rt.mat`, `measurements_spectators_v2.mat` |

Visualização da cena: `preview_3d.py`, `plot_3d_scene.py`,
`blender_render_scene.py` (headless) e `blender_open_scene.py` (GUI).

### 4. Densidade de UEs no MATLAB (`MATLAB/`)

Executar a partir da raiz do projeto, após `export_to_matlab.py`:

```bash
matlab -batch "addpath('MATLAB'); run('MATLAB/gnb_density_batch.m')"        # mapa de densidade RT
matlab -batch "addpath('MATLAB'); run('MATLAB/compare_rt_vs_analytic.m')"   # RT vs analítico
matlab -batch "addpath('MATLAB'); run('MATLAB/analyze_errors.m')"           # erro de posição, espectadores
```

`load_measurements.m` é um substituto direto de `simulate_measurements` que
devolve as medidas do Sionna RT. `osm_basemap.m` baixa tiles do OSM para o
fundo dos mapas. `gnb_density_demo_rt.m` é a versão interativa.

## Resultados principais (grade 41×41, 3.5 GHz)

Distribuição do caminho dominante: LoS 52.8 %, transmitted (O2I) 40.0 %,
reflected 3.8 %, diffracted 3.5 %.

| Estimador de posição | Erro mediano (m) | p95 (m) |
|---|---:|---:|
| RSRP, n = 2.7 fixo | 208.5 | 490.6 |
| RSRP, path loss calibrado global | 376.5 | 7570.2 |
| ToA do caminho dominante | 0.0 | 199.9 |
| ToA da primeira chegada | 0.0 | 168.4 |

Achados:

- A calibração direta do path loss tem R² ≈ 0.05 porque o RSRP embute o
  ganho direcional do UPA. Compensando o padrão do elemento, o R² do LoS
  sobe para 0.98.
- A primeira chegada é o caminho LoS em quase todos os UEs, mesmo nos
  classificados como reflected ou diffracted. O erro radial do ToA é
  praticamente nulo e o erro residual é tangencial (erro de AoA).
- No cenário F1, o mapa de densidade por RT recupera os três hotspots com
  erro zero e viés de massa de +7 %, contra +71 % do modelo analítico
  (`output/rho_comparison_metrics.txt`).

## Sistema de coordenadas

O XML é exportado com `axis_up="Y"` (Mitsuba). Sionna trabalha em ENU Z-up:

```
ENU.X (East)  =  Mitsuba.X
ENU.Y (North) = −Mitsuba.Z
ENU.Z (Up)    =  Mitsuba.Y
```

Os scripts MATLAB usam frame local NED (North → +X, East → +Y). A
conversão é feita em `export_to_matlab.py`; `gnb_pos` lá é
`[0, 0, −altura]`.

## Estrutura do repositório

```
.
├── Makefile                 # pipeline OSM → XML → smoke test + targets gNB/UE
├── install.sh               # Blender 3.6, mitsuba-blender, venv Sionna
├── setup_blender.py         # habilita add-ons no Blender headless
├── build_scene.py           # importa OSM, aplica materiais ITU, exporta XML
├── smoke_test.py            # valida o XML com Sionna RT
├── generators/              # gerador de cenários por YAML
│   ├── config.py            # carga e validação do YAML (dataclasses)
│   ├── gnb_placer.py        # GPS → ENU e criação da gNB
│   ├── ue_sampler.py        # grade / posições pré-computadas
│   ├── channel_sim.py       # loop do PathSolver e extração de RSRP/AoA
│   ├── exporter.py          # NPZ e MAT compatíveis com o pipeline original
│   └── generate.py          # CLI
├── scenarios/               # YAMLs de cenário
├── scripts/                 # gNB, grade, simulação, análise e figuras
├── MATLAB/                  # densidade de UEs e análise de erro
├── tools/sector_editor.html # editor de setores de espectadores
├── output/                  # artefatos gerados (dados pesados ignorados pelo git)
├── addons/                  # mitsuba-blender.zip
└── venv/                    # ambiente Python (ignorado pelo git)
```

## Limpeza

```bash
make clean       # remove output/ e osm_data/
make clean-all   # remove também venv/, addons/ e .env (não toca em ~/tools)
```

## Próximos passos possíveis

- Adicionar targets no Makefile para o fluxo por YAML, ToA e exportação MATLAB.
- Estudo de ISAC: atribuir `velocity` a objetos (veículos na pista), ligar
  `diffuse_reflection`, usar `paths.doppler` e `paths.cir` para mapas
  range-Doppler, e adicionar um receptor co-localizado na gNB para o modo
  monostático.
