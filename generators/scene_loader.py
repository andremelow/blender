"""
scene_loader.py — Carrega cena Sionna RT e configura arrays de antenas.

Chama sionna.rt.load_scene com o XML especificado na config e aplica
frequência, banda e arrays TX/RX exatamente como em simulate_ues.py.
"""

from __future__ import annotations

import sionna.rt as rt

from .config import Config


def load_scene(cfg: Config) -> rt.Scene:
    """Carrega cena XML e configura frequência, banda e arrays de antenas."""
    g = cfg.gnb
    a = g.array

    scene = rt.load_scene(cfg.scenario.scene_xml)
    scene.frequency = g.frequency_hz
    scene.bandwidth = g.bandwidth_hz

    scene.tx_array = rt.PlanarArray(
        num_rows=a.rows,
        num_cols=a.cols,
        vertical_spacing=a.spacing_factor,
        horizontal_spacing=a.spacing_factor,
        pattern=a.pattern,
        polarization=a.polarization,
    )
    # RX isotrópico de 1 elemento — igual ao pipeline original
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    return scene
