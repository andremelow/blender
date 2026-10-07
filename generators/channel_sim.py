"""
channel_sim.py — Orquestra o PathSolver para um array de posições de UE.

Mantém a mesma lógica de simulate_ues.py (checkpoint, extração de
RSRP/AoA/path_type) atrás de uma função chamável pela config.

Diferenças em relação ao script original:
  • Recebe posições como ndarray (sem carregar ue_grid.npz).
  • Recebe Config (sem ler PKL separado).
  • Sem argparse — parâmetros vêm exclusivamente da Config.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import sionna.rt as rt
from sionna.rt.constants import InteractionType

from .config import Config

_ITYPE_STR = {
    InteractionType.NONE:        "LoS",
    InteractionType.SPECULAR:    "reflected",
    InteractionType.DIFFUSE:     "diffuse",
    InteractionType.REFRACTION:  "refracted",
    InteractionType.DIFFRACTION: "diffracted",
}

_CHECKPOINT_EVERY = 50


def run_simulation(
    scene: rt.Scene,
    gnb_pos: np.ndarray,
    ue_positions: np.ndarray,
    cfg: Config,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """
    Simula canal RT para cada UE e retorna dict com arrays de resultado.

    Parâmetros
    ----------
    scene         : cena Sionna já configurada (load_scene + add Transmitter)
    gnb_pos       : posição ENU (3,) da gNB em metros
    ue_positions  : (N, 3) posições ENU dos UEs
    cfg           : Config carregado do YAML
    checkpoint_path : caminho .npz para checkpoint intermediário (opcional)
    resume        : retoma checkpoint existente se True

    Retorna dict com chaves:
        rsrp_dbm, aoa_az_rad, aoa_el_rad, path_type, n_paths,
        pos_east, pos_north, pos_up
    """
    sim = cfg.simulation
    n_ues = len(ue_positions)
    p_tx_w = 10 ** ((cfg.gnb.power_dbm - 30) / 10)

    rsrp_dbm    = np.full(n_ues, np.nan)
    aoa_az_rad  = np.full(n_ues, np.nan)
    aoa_el_rad  = np.full(n_ues, np.nan)
    path_type   = np.empty(n_ues, dtype=object)
    n_paths_arr = np.zeros(n_ues, dtype=int)
    pos_east    = ue_positions[:, 0].copy()
    pos_north   = ue_positions[:, 1].copy()
    pos_up      = ue_positions[:, 2].copy()

    resume_from = 0
    if resume and checkpoint_path and __import__("os").path.exists(checkpoint_path):
        chk = np.load(checkpoint_path, allow_pickle=True)
        n_done = len(chk["rsrp_dbm"])
        rsrp_dbm[:n_done]    = chk["rsrp_dbm"]
        aoa_az_rad[:n_done]  = chk["aoa_az_rad"]
        aoa_el_rad[:n_done]  = chk["aoa_el_rad"]
        path_type[:n_done]   = chk["path_type"]
        n_paths_arr[:n_done] = chk["n_paths"]
        resume_from = n_done
        print(f"[channel_sim] Retomando do checkpoint: {n_done}/{n_ues} prontos.")

    solver = rt.PathSolver()
    t0 = time.time()

    for i in range(resume_from, n_ues):
        pos_ue = ue_positions[i]
        rx_name = f"ue_{i}"
        scene.add(rt.Receiver(name=rx_name, position=pos_ue))
        try:
            paths = solver(
                scene=scene,
                max_depth=sim.max_depth,
                los=sim.los,
                specular_reflection=sim.specular_reflection,
                diffuse_reflection=sim.diffuse_reflection,
                diffraction=sim.diffraction,
                synthetic_array=True,
            )
            # paths.a[0]/[1]: real/imag, shape (1, n_rx_ant, 1, n_tx_ant, n_paths)
            a_r = np.array(paths.a[0])
            a_i = np.array(paths.a[1])
            a_sq = a_r ** 2 + a_i ** 2
            # Soma sobre todos os eixos de antena (todos menos o último = paths)
            pow_per_path = a_sq.sum(axis=tuple(range(a_sq.ndim - 1)))  # (n_paths,)

            # Filtra caminhos válidos
            valid_flat = np.array(paths.valid)[0, 0, :]   # (n_paths,) bool
            pow_valid  = pow_per_path * valid_flat.astype(float)

            n_valid = int(valid_flat.sum())
            n_paths_arr[i] = n_valid
            if n_valid == 0:
                path_type[i] = "none"
                scene.remove(rx_name)
                continue

            dom_idx = int(np.argmax(pow_valid))
            rsrp_dbm[i] = 10 * math.log10(p_tx_w * float(pow_valid.sum()) + 1e-30) + 30.

            # AoA na gNB = ângulos de partida TX por reciprocidade
            # phi_t, theta_t shape: (1, 1, n_paths)
            phi_t_all   = np.array(paths.phi_t  )[0, 0, :]
            theta_t_all = np.array(paths.theta_t)[0, 0, :]
            aoa_az_rad[i] = float(phi_t_all[dom_idx])
            aoa_el_rad[i] = math.pi / 2 - float(theta_t_all[dom_idx])

            # Tipo do caminho: interactions shape (max_depth, 1, 1, n_paths)
            inter     = np.array(paths.interactions)       # (max_depth, 1, 1, n_paths)
            inter_dom = inter[:, 0, 0, dom_idx]            # (max_depth,)
            inter_set = set(inter_dom.tolist())
            inter_set.discard(int(InteractionType.NONE))
            if not inter_set:
                path_type[i] = "LoS"
            elif int(InteractionType.DIFFRACTION) in inter_set:
                path_type[i] = "diffracted"
            elif int(InteractionType.SPECULAR) in inter_set or int(InteractionType.DIFFUSE) in inter_set:
                path_type[i] = "reflected"
            elif int(InteractionType.REFRACTION) in inter_set:
                path_type[i] = "transmitted"
            else:
                path_type[i] = "none"

        except Exception as exc:
            path_type[i] = "none"
            print(f"[channel_sim] UE {i}: erro — {exc}")
        finally:
            try:
                scene.remove(rx_name)
            except Exception:
                pass

        if checkpoint_path and (i + 1) % _CHECKPOINT_EVERY == 0:
            np.savez(
                checkpoint_path,
                rsrp_dbm=rsrp_dbm[: i + 1],
                aoa_az_rad=aoa_az_rad[: i + 1],
                aoa_el_rad=aoa_el_rad[: i + 1],
                path_type=path_type[: i + 1],
                n_paths=n_paths_arr[: i + 1],
            )

        if (i + 1) % 100 == 0 or (i + 1) == n_ues:
            elapsed = time.time() - t0
            rate = (i + 1 - resume_from) / max(elapsed, 1e-6)
            eta  = (n_ues - i - 1) / max(rate, 1e-6)
            print(f"[channel_sim] {i+1}/{n_ues}  "
                  f"({rate:.1f} UE/s  ETA {eta/60:.1f} min)", flush=True)

    return dict(
        rsrp_dbm    = rsrp_dbm,
        aoa_az_rad  = aoa_az_rad,
        aoa_el_rad  = aoa_el_rad,
        path_type   = path_type,
        n_paths     = n_paths_arr,
        pos_east    = pos_east,
        pos_north   = pos_north,
        pos_up      = pos_up,
    )
