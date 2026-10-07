"""
gnb_placer.py — Converte lat/lon da gNB para ENU e adiciona Transmitter à cena.

Usa a mesma fórmula flat-Earth de place_gnb.py para garantir consistência.
"""

from __future__ import annotations

import math
import pickle
import numpy as np
import sionna.rt as rt

from .config import Config, GnbConfig

_R_EARTH = 6_378_137.0   # raio da Terra em metros (WGS-84 equatorial)
_DEG2RAD = math.pi / 180.


def _lat_lon_to_enu(lat: float, lon: float,
                    lat0: float, lon0: float) -> tuple[float, float]:
    """Aproximação flat-Earth: retorna (east_m, north_m) relativo à origem."""
    east  = (lon - lon0) * _R_EARTH * math.cos(lat0 * _DEG2RAD) * _DEG2RAD
    north = (lat - lat0) * _R_EARTH * _DEG2RAD
    return east, north


def place_gnb(scene: rt.Scene, cfg: Config) -> np.ndarray:
    """Adiciona Transmitter à cena; retorna posição ENU (3,) em metros."""
    g = cfg.gnb
    east, north = _lat_lon_to_enu(
        g.lat, g.lon, g.scene_origin_lat, g.scene_origin_lon
    )
    orient_rad = [math.radians(v) for v in g.orientation_deg]
    pos_enu = np.array([east, north, g.height_m], dtype=float)

    scene.add(rt.Transmitter(
        name="gnb",
        position=pos_enu,
        orientation=orient_rad,
    ))
    return pos_enu


def gnb_config_dict(cfg: Config) -> dict:
    """Retorna dict compatível com output/gnb_config.pkl para scripts legados."""
    g = cfg.gnb
    a = g.array
    east, north = _lat_lon_to_enu(
        g.lat, g.lon, g.scene_origin_lat, g.scene_origin_lon
    )
    p_tx_w = 10 ** ((g.power_dbm - 30) / 10)   # dBm → Watts
    return dict(
        pos_enu    = [east, north, g.height_m],
        freq_hz    = g.frequency_hz,
        bw_hz      = g.bandwidth_hz,
        num_rows   = a.rows,
        num_cols   = a.cols,
        spacing    = a.spacing_factor,
        pattern    = a.pattern,
        polariz    = a.polarization,
        orientacao = [math.radians(v) for v in g.orientation_deg],
        p_tx_w     = p_tx_w,
    )


def save_gnb_config(cfg: Config, path: str = "output/gnb_config.pkl") -> None:
    """Persiste config da gNB em pkl (compatibilidade com simulate_ues.py)."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(gnb_config_dict(cfg), fh)
