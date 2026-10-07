"""
generate.py — CLI principal do gerador de cenários.

Uso:
    python -m generators.generate scenarios/interlagos_baseline.yaml [opções]

    --resume        retoma simulação de checkpoint existente
    --dry-run       valida config e imprime resumo sem rodar simulação
    --limit N       simula apenas os primeiros N UEs do grid (teste rápido)
    --checkpoint    caminho do checkpoint (padrão: output/<nome>_checkpoint.npz)
    --save-gnb-pkl  salva gnb_config.pkl para compatibilidade com scripts legados
"""

from __future__ import annotations

import argparse
import os
import sys

# Garante que a raiz do projeto está no path quando rodado como script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generators.config import load_config
from generators.scene_loader import load_scene
from generators.gnb_placer import place_gnb, save_gnb_config
from generators.ue_sampler import sample_ue_positions
from generators.channel_sim import run_simulation
from generators.exporter import save_results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m generators.generate",
        description="Gerador YAML-driven de cenários Sionna RT.",
    )
    p.add_argument("yaml", help="Caminho para o arquivo YAML do cenário.")
    p.add_argument("--resume",      action="store_true",
                   help="Retoma simulação de checkpoint existente.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Valida config e imprime resumo sem rodar simulação.")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Simula apenas os primeiros N UEs do grid (teste rápido).")
    p.add_argument("--checkpoint",  default=None,
                   help="Caminho do arquivo de checkpoint (.npz).")
    p.add_argument("--save-gnb-pkl", action="store_true",
                   help="Persiste gnb_config.pkl para compatibilidade com scripts legados.")
    return p


def _checkpoint_path(cfg, cli_arg: str | None) -> str:
    if cli_arg:
        return cli_arg
    name = cfg.scenario.name.replace(" ", "_")
    return os.path.join("output", f"{name}_checkpoint.npz")


def _print_summary(cfg, n_ues_total: int, n_limit: int | None) -> None:
    """Imprime tabela resumo do cenário antes de rodar."""
    g = cfg.gnb
    s = cfg.simulation
    u = cfg.ues
    n_run = n_limit if n_limit else n_ues_total

    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print(f"│  Cenário : {cfg.scenario.name:<45s}│")
    print(f"│  Cena    : {cfg.scenario.scene_xml:<45s}│")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  gNB  lat={g.lat:.6f}  lon={g.lon:.6f}  h={g.height_m:.1f} m{'':<5s}│")
    print(f"│       {g.array.rows}×{g.array.cols} UPA  {g.array.pattern}  {g.frequency_hz/1e9:.2f} GHz  {g.power_dbm:.0f} dBm{'':<10s}│")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  UEs  mode={u.mode}  h={u.height_m} m  total={n_ues_total}{'':<18s}│")
    if u.grid:
        gr = u.grid
        print(f"│       E[{gr.east_min:.0f}..{gr.east_max:.0f} Δ{gr.east_step:.0f}]  "
              f"N[{gr.north_min:.0f}..{gr.north_max:.0f} Δ{gr.north_step:.0f}]{'':<4s}│")
    if n_limit:
        print(f"│       ⚠  --limit {n_limit}: simulando {n_run}/{n_ues_total} UEs{'':<16s}│")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Sim  max_depth={s.max_depth}  LoS={s.los}  specular={s.specular_reflection}{'':<11s}│")
    print(f"│       diffr={s.diffraction}  diffuse={s.diffuse_reflection}  seed={s.random_seed}{'':<12s}│")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Out  {cfg.output.measurements_npz:<51s}│")
    if cfg.output.mat_file:
        print(f"│       {cfg.output.mat_file:<51s}│")
    print("└─────────────────────────────────────────────────────────┘")
    print()


def main() -> int:
    args = _build_parser().parse_args()

    # ── Valida config ──────────────────────────────────────────────────────────
    print(f"[generate] Carregando cenário: {args.yaml}")
    try:
        cfg = load_config(args.yaml)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1

    # Calcula tamanho total do grid para o resumo
    from generators.ue_sampler import sample_ue_positions as _smp
    ue_positions_all = _smp(cfg)
    n_total = len(ue_positions_all)

    _print_summary(cfg, n_total, args.limit)

    if args.dry_run:
        print("[generate] --dry-run: validação OK, simulação pulada.")
        return 0

    # Aplica --limit
    ue_positions = ue_positions_all if args.limit is None else ue_positions_all[: args.limit]

    # ── Carrega cena ──────────────────────────────────────────────────────────
    print("[generate] Carregando cena Sionna RT...")
    scene = load_scene(cfg)

    # ── Posiciona gNB ─────────────────────────────────────────────────────────
    gnb_pos = place_gnb(scene, cfg)
    print(f"[generate] gNB ENU: E={gnb_pos[0]:.2f}  N={gnb_pos[1]:.2f}  "
          f"Z={gnb_pos[2]:.1f} m")

    if args.save_gnb_pkl:
        save_gnb_config(cfg)
        print("[generate] gnb_config.pkl salvo.")

    print(f"[generate] Simulando {len(ue_positions)} UEs...")

    # ── Simulação ─────────────────────────────────────────────────────────────
    chk = _checkpoint_path(cfg, args.checkpoint)
    results = run_simulation(
        scene=scene,
        gnb_pos=gnb_pos,
        ue_positions=ue_positions,
        cfg=cfg,
        checkpoint_path=chk,
        resume=args.resume,
    )

    # ── Exporta ───────────────────────────────────────────────────────────────
    save_results(results, cfg, gnb_pos=gnb_pos)
    print("[generate] Concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
