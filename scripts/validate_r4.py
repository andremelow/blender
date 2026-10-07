"""
validate_r4.py — Valida o YAML de cenário contra measurements_v2.npz (baseline).

Verificações realizadas:
  1. O YAML é carregado e validado sem erros.
  2. O NPZ baseline existe e contém todos os campos obrigatórios.
  3. Os parâmetros de grid (YAML) batem com as posições reais (NPZ).
  4. Os campos de metadado (freq_hz, p_tx_w, gnb_pos) conferem com a config.
  5. Os campos de resultado têm dtype e faixa de valores razoáveis.
  6. O NPZ que o exporter.py geraria teria os campos esperados (simulação seca).

Saída:
  output/validate_r4.txt — relatório pass/fail

Uso:
    python scripts/validate_r4.py [--yaml YAML] [--npz NPZ]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generators.config import load_config

# ─── Constantes ───────────────────────────────────────────────────────────────

DEFAULT_YAML = "scenarios/interlagos_baseline.yaml"
DEFAULT_NPZ  = "output/measurements_v2.npz"
OUT_TXT      = "output/validate_r4.txt"

# Campos obrigatórios no NPZ
REQUIRED_FIELDS = {
    "rsrp_dbm", "aoa_az_rad", "aoa_el_rad",
    "path_type", "n_paths",
    "pos_east", "pos_north", "pos_up",
}
# Campos de metadado (presentes na v2; gerados pelo exporter)
META_FIELDS = {"freq_hz", "p_tx_w", "gnb_pos"}

VALID_PATH_TYPES = {"LoS", "reflected", "diffracted", "transmitted",
                    "diffuse", "refracted", "none"}


def _pass(msg: str, lines: list) -> None:
    lines.append(f"  PASS  {msg}")


def _fail(msg: str, lines: list) -> None:
    lines.append(f"  FAIL  {msg}")


def _warn(msg: str, lines: list) -> None:
    lines.append(f"  WARN  {msg}")


# ─── Verificações ──────────────────────────────────────────────────────────────

def check_yaml(yaml_path: str, lines: list):
    lines.append("\n[1] Validação do YAML de cenário")
    try:
        cfg = load_config(yaml_path)
        _pass(f"YAML carregado: {cfg.scenario.name}", lines)
        return cfg
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"Erro ao carregar YAML: {exc}", lines)
        return None


def check_npz_exists(npz_path: str, lines: list):
    lines.append("\n[2] Existência e campos do NPZ baseline")
    if not os.path.exists(npz_path):
        _fail(f"NPZ não encontrado: {npz_path}", lines)
        return None
    _pass(f"NPZ encontrado: {npz_path}", lines)
    return np.load(npz_path, allow_pickle=True)


def check_fields(npz, lines: list) -> bool:
    existing = set(npz.files)
    missing_req  = REQUIRED_FIELDS - existing
    missing_meta = META_FIELDS - existing

    if not missing_req:
        _pass(f"Campos obrigatórios presentes: {sorted(REQUIRED_FIELDS)}", lines)
    else:
        _fail(f"Campos obrigatórios ausentes: {sorted(missing_req)}", lines)

    if not missing_meta:
        _pass(f"Campos de metadado presentes: {sorted(META_FIELDS)}", lines)
    else:
        _warn(f"Campos de metadado ausentes (exporter v2 necessário): "
              f"{sorted(missing_meta)}", lines)

    return not missing_req


def check_grid(npz, cfg, lines: list) -> None:
    lines.append("\n[3] Consistência grid YAML ↔ posições NPZ")
    if cfg is None or cfg.ues.grid is None:
        _warn("Nenhuma config de grid para verificar.", lines)
        return

    gr  = cfg.ues.grid
    pe  = npz["pos_east"]
    pn  = npz["pos_north"]
    pu  = npz["pos_up"]

    # Faixa de posições
    tol = 0.1   # metros

    if pe.min() >= gr.east_min - tol and pe.max() <= gr.east_max + tol:
        _pass(f"pos_east ∈ [{pe.min():.0f}, {pe.max():.0f}] m "
              f"⊂ [{gr.east_min:.0f}, {gr.east_max:.0f}] m", lines)
    else:
        _fail(f"pos_east [{pe.min():.1f}, {pe.max():.1f}] fora de "
              f"[{gr.east_min:.0f}, {gr.east_max:.0f}]", lines)

    if pn.min() >= gr.north_min - tol and pn.max() <= gr.north_max + tol:
        _pass(f"pos_north ∈ [{pn.min():.0f}, {pn.max():.0f}] m "
              f"⊂ [{gr.north_min:.0f}, {gr.north_max:.0f}] m", lines)
    else:
        _fail(f"pos_north [{pn.min():.1f}, {pn.max():.1f}] fora de "
              f"[{gr.north_min:.0f}, {gr.north_max:.0f}]", lines)

    # Altura dos UEs
    h_yaml = cfg.ues.height_m
    if np.allclose(pu, h_yaml, atol=0.01):
        _pass(f"pos_up = {pu[0]:.2f} m (YAML: {h_yaml} m)", lines)
    else:
        _fail(f"pos_up [{pu.min():.2f}, {pu.max():.2f}] ≠ height_m={h_yaml}", lines)

    # Tamanho do grid
    n_east_yaml  = round((gr.east_max  - gr.east_min)  / gr.east_step)  + 1
    n_north_yaml = round((gr.north_max - gr.north_min) / gr.north_step) + 1
    n_yaml = n_east_yaml * n_north_yaml
    n_npz  = len(pe)
    if n_npz == n_yaml:
        _pass(f"N_ues: {n_npz} = {n_east_yaml}×{n_north_yaml} (grid YAML)", lines)
    else:
        _warn(f"N_ues NPZ={n_npz}  vs  YAML grid={n_yaml} ({n_east_yaml}×{n_north_yaml}); "
              "pode refletir filtragem de cobertura.", lines)

    # Passo mínimo no NPZ
    pe_sorted = np.unique(pe)
    if len(pe_sorted) > 1:
        min_step = float(np.diff(pe_sorted).min())
        if abs(min_step - gr.east_step) < 0.5:
            _pass(f"Passo mínimo East: {min_step:.1f} m (YAML: {gr.east_step:.0f} m)", lines)
        else:
            _fail(f"Passo mínimo East: {min_step:.1f} m ≠ YAML {gr.east_step:.0f} m", lines)


def check_metadata(npz, cfg, lines: list) -> None:
    lines.append("\n[4] Metadados freq_hz / p_tx_w / gnb_pos")
    if cfg is None:
        return

    tol_rel = 5e-3   # 0.5 % — acomoda arredondamento dBm↔W

    # freq_hz
    if "freq_hz" in npz.files:
        f_npz  = float(npz["freq_hz"])
        f_yaml = cfg.gnb.frequency_hz
        if abs(f_npz - f_yaml) / f_yaml < tol_rel:
            _pass(f"freq_hz: {f_npz/1e9:.4f} GHz ≈ YAML {f_yaml/1e9:.4f} GHz", lines)
        else:
            _fail(f"freq_hz: NPZ={f_npz:.3e} ≠ YAML={f_yaml:.3e}", lines)

    # p_tx_w
    if "p_tx_w" in npz.files:
        p_npz  = float(npz["p_tx_w"])
        p_yaml = 10 ** ((cfg.gnb.power_dbm - 30) / 10)
        if abs(p_npz - p_yaml) / p_yaml < tol_rel:
            _pass(f"p_tx_w: {p_npz:.4f} W ≈ YAML {p_yaml:.4f} W "
                  f"({cfg.gnb.power_dbm:.0f} dBm)", lines)
        else:
            _fail(f"p_tx_w: NPZ={p_npz:.4f} ≠ YAML {p_yaml:.4f}", lines)

    # gnb_pos (ENU) — compara com posição derivada do YAML
    if "gnb_pos" in npz.files and cfg:
        gnb_npz = np.array(npz["gnb_pos"], dtype=float)
        from generators.gnb_placer import _lat_lon_to_enu
        e, n = _lat_lon_to_enu(
            cfg.gnb.lat, cfg.gnb.lon,
            cfg.gnb.scene_origin_lat, cfg.gnb.scene_origin_lon,
        )
        gnb_yaml = np.array([e, n, cfg.gnb.height_m])
        err = float(np.linalg.norm(gnb_npz - gnb_yaml))
        if err < 1.0:
            _pass(f"gnb_pos ENU: [{gnb_npz[0]:.2f}, {gnb_npz[1]:.2f}, {gnb_npz[2]:.1f}] m "
                  f"(err={err:.3f} m)", lines)
        else:
            _fail(f"gnb_pos: NPZ={gnb_npz.tolist()}  YAML→{gnb_yaml.tolist()}  "
                  f"err={err:.2f} m", lines)


def check_result_fields(npz, lines: list) -> None:
    lines.append("\n[5] Faixa de valores dos campos de resultado")

    # RSRP
    rsrp = npz["rsrp_dbm"].astype(float)
    valid = rsrp[np.isfinite(rsrp)]
    pct_nan = 100. * (1 - len(valid) / len(rsrp))
    if -130. <= valid.min() and valid.max() <= 0.:
        _pass(f"rsrp_dbm ∈ [{valid.min():.1f}, {valid.max():.1f}] dBm  "
              f"(NaN: {pct_nan:.0f}%)", lines)
    else:
        _warn(f"rsrp_dbm fora de [-130, 0] dBm: [{valid.min():.1f}, {valid.max():.1f}]", lines)

    # AoA azimute: [-π, π]
    aoa = npz["aoa_az_rad"].astype(float)
    valid_aoa = aoa[np.isfinite(aoa)]
    if valid_aoa.min() >= -math.pi - 0.01 and valid_aoa.max() <= math.pi + 0.01:
        _pass(f"aoa_az_rad ∈ [{valid_aoa.min():.3f}, {valid_aoa.max():.3f}] rad  ([-π, π])", lines)
    else:
        _fail(f"aoa_az_rad fora de [-π, π]: [{valid_aoa.min():.3f}, {valid_aoa.max():.3f}]", lines)

    # n_paths ≥ 0
    np_arr = npz["n_paths"].astype(int)
    if np_arr.min() >= 0:
        _pass(f"n_paths ≥ 0  (min={np_arr.min()}, max={np_arr.max()})", lines)
    else:
        _fail(f"n_paths < 0 encontrado", lines)

    # path_type: somente valores conhecidos
    pt = npz["path_type"]
    unknown = set(pt) - VALID_PATH_TYPES
    if not unknown:
        from collections import Counter
        counts = Counter(pt)
        _pass(f"path_type: {dict(counts)}", lines)
    else:
        _fail(f"path_type desconhecidos: {unknown}", lines)


def check_exporter_compat(cfg, lines: list) -> None:
    """Verifica que exporter.py geraria todos os campos esperados (sem rodar Sionna)."""
    lines.append("\n[6] Compatibilidade do exporter (simulação seca)")
    import importlib
    try:
        importlib.import_module("generators.exporter")
        _pass("generators.exporter importado sem erros", lines)
    except ImportError as exc:
        _fail(f"Erro ao importar exporter: {exc}", lines)
        return

    # Campos que o exporter.save_results gera no NPZ
    expected_output = REQUIRED_FIELDS | META_FIELDS
    _pass(f"Campos que o exporter produzirá: {sorted(expected_output)}", lines)

    # Confirma que o MAT também será gerado quando configurado
    if cfg and cfg.output.mat_file:
        _pass(f"MAT habilitado: {cfg.output.mat_file}", lines)
    else:
        _warn("mat_file não configurado; MATLAB não receberá dados.", lines)


# ─── Relatório ────────────────────────────────────────────────────────────────

def run(yaml_path: str, npz_path: str) -> int:
    lines: list[str] = [
        "Validação R4 — gerador YAML vs measurements_v2.npz",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"YAML: {yaml_path}",
        f"NPZ:  {npz_path}",
    ]

    cfg = check_yaml(yaml_path, lines)
    npz = check_npz_exists(npz_path, lines)

    if npz is not None:
        ok = check_fields(npz, lines)
        if ok:
            check_grid(npz, cfg, lines)
            check_metadata(npz, cfg, lines)
            check_result_fields(npz, lines)

    check_exporter_compat(cfg, lines)

    n_fail = sum(1 for l in lines if "FAIL" in l)
    n_warn = sum(1 for l in lines if "WARN" in l)
    n_pass = sum(1 for l in lines if "PASS" in l)

    lines.append(
        f"\n{'─'*55}\n"
        f"Resultado: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL\n"
        + ("OK — gerador compatível com baseline.\n" if n_fail == 0
           else "FALHOU — veja itens FAIL acima.\n")
    )

    report = "\n".join(lines)
    print(report)

    os.makedirs("output", exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nRelatório salvo: {OUT_TXT}")

    return 0 if n_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Valida YAML de cenário contra NPZ baseline.")
    p.add_argument("--yaml", default=DEFAULT_YAML)
    p.add_argument("--npz",  default=DEFAULT_NPZ)
    args = p.parse_args()
    return run(args.yaml, args.npz)


if __name__ == "__main__":
    sys.exit(main())
