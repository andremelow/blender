"""
ue_sampler.py — Gera grid/lista de posições de UE a partir da config.

Modo "grid": usa os parâmetros east/north_min/max/step do YAML, exatamente
como generate_ue_grid.py (posições em ENU absoluto relativo à origem da cena).

Outros modos ("list", "clusters") são reservados para extensões futuras.
"""

from __future__ import annotations

import numpy as np

from .config import Config, GridConfig


def _make_grid(gc: GridConfig, height_m: float) -> np.ndarray:
    """Retorna array (N, 3) com posições ENU do grid."""
    east_vals  = np.arange(gc.east_min,  gc.east_max  + 1e-9, gc.east_step)
    north_vals = np.arange(gc.north_min, gc.north_max + 1e-9, gc.north_step)
    grid_e, grid_n = np.meshgrid(east_vals, north_vals)
    n_total = grid_e.size
    positions = np.column_stack([
        grid_e.flatten(),
        grid_n.flatten(),
        np.full(n_total, height_m),
    ])
    return positions


def sample_ue_positions(cfg: Config) -> np.ndarray:
    """Retorna posições (N, 3) ENU em metros para os UEs do cenário."""
    u = cfg.ues
    if u.mode == "grid":
        if u.grid is None:
            raise ValueError("[ue_sampler] mode='grid' requer seção 'ues.grid' no YAML.")
        return _make_grid(u.grid, u.height_m)

    if u.mode == "precomputed":
        if u.precomputed_npz is None:
            raise ValueError("[ue_sampler] mode='precomputed' requer 'precomputed_npz' no YAML.")
        return _load_precomputed(u.precomputed_npz)

    raise NotImplementedError(
        f"[ue_sampler] mode='{u.mode}' ainda não implementado. "
        "Modos disponíveis: 'grid', 'precomputed'."
    )


def _load_precomputed(npz_path: str) -> np.ndarray:
    """Carrega posições de um NPZ pré-gerado (ex: ue_spectators.npz)."""
    import os
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"[ue_sampler] NPZ não encontrado: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    if "positions" not in d:
        raise ValueError(f"[ue_sampler] NPZ '{npz_path}' não tem campo 'positions'.")
    positions = d["positions"]          # (N, 3) ENU
    if "valid_mask" in d:
        mask = d["valid_mask"].astype(bool)
        positions = positions[mask]
    print(f"[ue_sampler] {len(positions)} posições carregadas de {npz_path}")
    return positions
