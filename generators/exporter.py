"""
exporter.py — Salva resultados da simulação em NPZ e opcionalmente MAT.

Mantém compatibilidade de campos com measurements_v2.npz e
measurements_rt.mat gerados pelos scripts originais.
"""

from __future__ import annotations

import os
import numpy as np

from .config import Config


def save_results(results: dict, cfg: Config, gnb_pos: "np.ndarray | None" = None) -> None:
    """
    Persiste resultados em NPZ (e MAT se configurado).

    Parâmetros
    ----------
    results : dict retornado por channel_sim.run_simulation
    cfg     : Config do cenário (fornece caminhos de saída)
    gnb_pos : posição ENU (3,) da gNB — salva como metadado de compatibilidade
    """
    import math
    out = cfg.output
    os.makedirs(os.path.dirname(os.path.abspath(out.measurements_npz)), exist_ok=True)

    p_tx_w = 10 ** ((cfg.gnb.power_dbm - 30) / 10)

    # Compatibilidade com measurements_v2.npz: inclui campos de metadado
    extra: dict = dict(
        freq_hz = np.float64(cfg.gnb.frequency_hz),
        p_tx_w  = np.float64(p_tx_w),
    )
    if gnb_pos is not None:
        extra["gnb_pos"] = np.asarray(gnb_pos, dtype=float)

    # — NPZ —
    np.savez(
        out.measurements_npz,
        rsrp_dbm   = results["rsrp_dbm"],
        aoa_az_rad = results["aoa_az_rad"],
        aoa_el_rad = results["aoa_el_rad"],
        path_type  = results["path_type"],
        n_paths    = results["n_paths"],
        pos_east   = results["pos_east"],
        pos_north  = results["pos_north"],
        pos_up     = results["pos_up"],
        **extra,
    )
    print(f"[exporter] NPZ salvo: {out.measurements_npz}")

    # — MAT (opcional) —
    if out.mat_file:
        from scipy.io import savemat
        from .gnb_placer import gnb_config_dict
        import math

        gnb_cfg = gnb_config_dict(cfg)
        gnb_pos = gnb_cfg["pos_enu"]

        # Posições relativas à gNB (convenção MATLAB: Down = -Up)
        pos_east_rel  = results["pos_east"]  - gnb_pos[0]
        pos_north_rel = results["pos_north"] - gnb_pos[1]

        pt = results["path_type"]
        path_type_list = [str(v) if v is not None else "none" for v in pt]

        mat_dict = dict(
            n_ues        = float(len(results["rsrp_dbm"])),
            rsrp_dBm     = results["rsrp_dbm"].astype(float),
            aoa_rad      = results["aoa_az_rad"].astype(float),
            pos_east     = pos_east_rel.astype(float),
            pos_north    = pos_north_rel.astype(float),
            path_type    = path_type_list,
            gnb_pos_matlab = [0., 0., -float(gnb_pos[2])],   # Down convention
        )
        os.makedirs(os.path.dirname(os.path.abspath(out.mat_file)), exist_ok=True)
        savemat(out.mat_file, mat_dict)
        print(f"[exporter] MAT salvo:  {out.mat_file}")
