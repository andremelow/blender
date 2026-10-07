"""
config.py — Carrega e valida arquivo YAML de cenário.

Usa dataclasses + validação manual (pydantic não disponível no venv).
Falha com mensagem clara se qualquer campo obrigatório estiver ausente
ou fora dos limites aceitos.

Uso:
    from generators.config import load_config
    cfg = load_config("scenarios/interlagos_baseline.yaml")
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ─── Sub-configurações ────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    name: str
    scene_xml: str


@dataclass
class ArrayConfig:
    rows: int
    cols: int
    spacing_factor: float
    pattern: str
    polarization: str


@dataclass
class GnbConfig:
    lat: float
    lon: float
    scene_origin_lat: float
    scene_origin_lon: float
    height_m: float
    orientation_deg: List[float]
    array: ArrayConfig
    power_dbm: float
    frequency_hz: float
    bandwidth_hz: float


@dataclass
class GridConfig:
    east_min: float
    east_max: float
    east_step: float
    north_min: float
    north_max: float
    north_step: float


@dataclass
class UeConfig:
    mode: str
    height_m: float
    grid: Optional[GridConfig] = None
    coverage_threshold_db: float = -140.0
    precomputed_npz: Optional[str] = None   # caminho do NPZ para mode='precomputed'


@dataclass
class SimulationConfig:
    max_depth: int
    los: bool
    specular_reflection: bool
    diffraction: bool
    diffuse_reflection: bool
    batch_size: int
    random_seed: int
    rsrp_threshold_dbm: float


@dataclass
class OutputConfig:
    measurements_npz: str
    mat_file: Optional[str] = None
    figures_dir: Optional[str] = None


@dataclass
class Config:
    scenario: ScenarioConfig
    gnb: GnbConfig
    ues: UeConfig
    simulation: SimulationConfig
    output: OutputConfig


# ─── Helpers de validação ─────────────────────────────────────────────────────

def _req(mapping: dict, key: str, context: str):
    """Falha cedo se campo obrigatório ausente."""
    if key not in mapping:
        raise ValueError(f"[config] Campo obrigatório '{key}' ausente em '{context}'.")
    return mapping[key]


def _in_range(val, lo, hi, name: str):
    if not (lo <= val <= hi):
        raise ValueError(f"[config] '{name}' = {val} fora de [{lo}, {hi}].")


def _positive(val, name: str):
    if val <= 0:
        raise ValueError(f"[config] '{name}' deve ser > 0, mas é {val}.")


def _nonneg_int(val, name: str):
    if not isinstance(val, int) or val < 0:
        raise ValueError(f"[config] '{name}' deve ser inteiro ≥ 0, mas é {val!r}.")


# ─── Parsers por seção ────────────────────────────────────────────────────────

def _parse_scenario(d: dict) -> ScenarioConfig:
    name     = _req(d, "name",     "scenario")
    scene_xml = _req(d, "scene_xml", "scenario")
    return ScenarioConfig(name=str(name), scene_xml=str(scene_xml))


def _parse_array(d: dict) -> ArrayConfig:
    rows    = int(_req(d, "rows",           "gnb.array"))
    cols    = int(_req(d, "cols",           "gnb.array"))
    spacing = float(_req(d, "spacing_factor", "gnb.array"))
    pattern = str(_req(d, "pattern",        "gnb.array"))
    polariz = str(_req(d, "polarization",   "gnb.array"))

    if rows < 1:
        raise ValueError(f"[config] 'gnb.array.rows' deve ser ≥ 1, mas é {rows}.")
    if cols < 1:
        raise ValueError(f"[config] 'gnb.array.cols' deve ser ≥ 1, mas é {cols}.")
    if spacing <= 0:
        raise ValueError(f"[config] 'gnb.array.spacing_factor' deve ser > 0.")

    return ArrayConfig(rows=rows, cols=cols, spacing_factor=spacing,
                       pattern=pattern, polarization=polariz)


def _parse_gnb(d: dict) -> GnbConfig:
    lat     = float(_req(d, "lat",              "gnb"))
    lon     = float(_req(d, "lon",              "gnb"))
    ori_lat = float(_req(d, "scene_origin_lat", "gnb"))
    ori_lon = float(_req(d, "scene_origin_lon", "gnb"))
    h       = float(_req(d, "height_m",         "gnb"))
    orient  = list(_req(d, "orientation_deg",   "gnb"))
    pwr     = float(_req(d, "power_dbm",        "gnb"))
    freq    = float(_req(d, "frequency_hz",     "gnb"))
    bw      = float(_req(d, "bandwidth_hz",     "gnb"))

    _in_range(lat, -90., 90.,   "gnb.lat")
    _in_range(lon, -180., 180., "gnb.lon")
    _in_range(ori_lat, -90., 90.,   "gnb.scene_origin_lat")
    _in_range(ori_lon, -180., 180., "gnb.scene_origin_lon")
    _positive(h, "gnb.height_m")
    _positive(freq, "gnb.frequency_hz")
    _positive(bw,   "gnb.bandwidth_hz")

    if len(orient) != 3:
        raise ValueError("[config] 'gnb.orientation_deg' deve ter exatamente 3 elementos.")

    if not (0. <= pwr <= 50.):
        warnings.warn(
            f"[config] 'gnb.power_dbm' = {pwr} dBm fora do intervalo típico [0, 50] dBm.",
            stacklevel=4,
        )

    arr = _parse_array(_req(d, "array", "gnb"))

    return GnbConfig(
        lat=lat, lon=lon,
        scene_origin_lat=ori_lat, scene_origin_lon=ori_lon,
        height_m=h,
        orientation_deg=[float(v) for v in orient],
        array=arr,
        power_dbm=pwr,
        frequency_hz=freq,
        bandwidth_hz=bw,
    )


def _parse_grid(d: dict) -> GridConfig:
    e_min  = float(_req(d, "east_min",  "ues.grid"))
    e_max  = float(_req(d, "east_max",  "ues.grid"))
    e_step = float(_req(d, "east_step", "ues.grid"))
    n_min  = float(_req(d, "north_min", "ues.grid"))
    n_max  = float(_req(d, "north_max", "ues.grid"))
    n_step = float(_req(d, "north_step","ues.grid"))

    if e_min >= e_max:
        raise ValueError("[config] 'ues.grid.east_min' deve ser < 'east_max'.")
    if n_min >= n_max:
        raise ValueError("[config] 'ues.grid.north_min' deve ser < 'north_max'.")
    if e_step <= 0:
        raise ValueError("[config] 'ues.grid.east_step' deve ser > 0.")
    if n_step <= 0:
        raise ValueError("[config] 'ues.grid.north_step' deve ser > 0.")

    return GridConfig(east_min=e_min, east_max=e_max, east_step=e_step,
                      north_min=n_min, north_max=n_max, north_step=n_step)


_VALID_MODES = {"grid", "list", "clusters", "precomputed"}


def _parse_ues(d: dict) -> UeConfig:
    mode = str(_req(d, "mode", "ues"))
    h    = float(_req(d, "height_m", "ues"))
    thresh = float(d.get("coverage_threshold_db", -140.0))

    if mode not in _VALID_MODES:
        raise ValueError(
            f"[config] 'ues.mode' = '{mode}' inválido. "
            f"Valores aceitos: {sorted(_VALID_MODES)}."
        )
    _positive(h, "ues.height_m")

    grid = None
    if mode == "grid":
        grid = _parse_grid(_req(d, "grid", "ues"))

    precomputed_npz = None
    if mode == "precomputed":
        precomputed_npz = str(_req(d, "precomputed_npz", "ues"))

    return UeConfig(mode=mode, height_m=h, grid=grid,
                    coverage_threshold_db=thresh,
                    precomputed_npz=precomputed_npz)


def _parse_simulation(d: dict) -> SimulationConfig:
    depth    = int(_req(d, "max_depth",           "simulation"))
    los      = bool(_req(d, "los",                "simulation"))
    spec_ref = bool(_req(d, "specular_reflection","simulation"))
    diffr    = bool(_req(d, "diffraction",        "simulation"))
    diff_ref = bool(_req(d, "diffuse_reflection", "simulation"))
    batch    = int(d.get("batch_size", 0))
    seed     = int(d.get("random_seed", 42))
    thresh   = float(d.get("rsrp_threshold_dbm", -120.0))

    _in_range(depth, 1, 10, "simulation.max_depth")
    _nonneg_int(batch, "simulation.batch_size")

    return SimulationConfig(
        max_depth=depth,
        los=los,
        specular_reflection=spec_ref,
        diffraction=diffr,
        diffuse_reflection=diff_ref,
        batch_size=batch,
        random_seed=seed,
        rsrp_threshold_dbm=thresh,
    )


def _parse_output(d: dict) -> OutputConfig:
    npz     = str(_req(d, "measurements_npz", "output"))
    mat     = d.get("mat_file")
    fig_dir = d.get("figures_dir")
    return OutputConfig(
        measurements_npz=npz,
        mat_file=str(mat) if mat else None,
        figures_dir=str(fig_dir) if fig_dir else None,
    )


# ─── Entrada pública ──────────────────────────────────────────────────────────

def load_config(yaml_path: str) -> Config:
    """Carrega e valida YAML de cenário; levanta ValueError em caso de erro."""
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"[config] Arquivo de cenário não encontrado: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("[config] YAML deve ser um mapeamento de chave-valor no nível raiz.")

    for section in ("scenario", "gnb", "ues", "simulation", "output"):
        if section not in raw:
            raise ValueError(f"[config] Seção obrigatória '{section}' ausente no YAML.")

    return Config(
        scenario   = _parse_scenario(_req(raw, "scenario",   "root")),
        gnb        = _parse_gnb     (_req(raw, "gnb",        "root")),
        ues        = _parse_ues     (_req(raw, "ues",        "root")),
        simulation = _parse_simulation(_req(raw, "simulation","root")),
        output     = _parse_output  (_req(raw, "output",     "root")),
    )
