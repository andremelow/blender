"""
summary_stats.py — Tabela de resumo estatístico das medidas Sionna RT v2.

Gera output/summary_stats.md com colunas:
  Categoria | n | % | RSRP med (dBm) | AoA err med (°) | AoA err p95 (°)
           | Pos err med (m) | Pos err p95 (m)

Estimador de posição (igual a analyze_bad_regime.py):
  d_hat = 10^((EIRP - RSRP_dBm) / (10 * n))
  EIRP = -7 dBm  (P_tx 23 dBm + ganho 8 dBi − PL0 38 dB)
  n    = 2.7

Uso:
    python scripts/summary_stats.py [--npz output/measurements_v2.npz]
"""

import os, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default="output/measurements_v2.npz")
parser.add_argument("--out", default="output/summary_stats.md")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

EIRP_DBM = -7.0
N_PL     = 2.7
CATS     = ["LoS", "diffracted", "reflected", "transmitted", "none"]

info(f"Carregando: {args.npz}")
d       = np.load(args.npz, allow_pickle=True)
pos_e   = d["pos_east"];   pos_n  = d["pos_north"]
rsrp    = d["rsrp_dbm"];   ptype  = d["path_type"]
aoa_az  = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb     = d["gnb_pos"]
n_ues   = len(rsrp)
freq_hz = float(d["freq_hz"])

# ─── Erro AoA 3D ──────────────────────────────────────────────────────────────
delta_e = pos_e - gnb[0]
delta_n = pos_n - gnb[1]
delta_u = np.full(n_ues, 1.5) - gnb[2]
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)
phi_true   = np.arctan2(delta_n, delta_e)
theta_true = np.arccos(np.clip(delta_u / np.maximum(dist_3d, 1e-6), -1, 1))
el_true    = np.pi / 2 - theta_true

def sph2vec(el_rad, az_rad):
    theta = np.pi / 2 - el_rad
    return np.column_stack([
        np.sin(theta) * np.cos(az_rad),
        np.sin(theta) * np.sin(az_rad),
        np.cos(theta),
    ])

valid_mask = (ptype != 'none') & np.isfinite(aoa_az) & np.isfinite(aoa_el)
err_3d     = np.full(n_ues, np.nan)
if valid_mask.sum() > 0:
    mv  = sph2vec(aoa_el[valid_mask], aoa_az[valid_mask])
    tv  = sph2vec(el_true[valid_mask], phi_true[valid_mask])
    dot = np.clip((mv * tv).sum(axis=1), -1., 1.)
    err_3d[valid_mask] = np.degrees(np.arccos(dot))

# ─── Estimativa de posição ────────────────────────────────────────────────────
d_hat     = 10 ** ((EIRP_DBM - rsrp) / (10 * N_PL))
theta_aoa = np.pi / 2 - aoa_el
east_hat  = gnb[0] + d_hat * np.sin(theta_aoa) * np.cos(aoa_az)
north_hat = gnb[1] + d_hat * np.sin(theta_aoa) * np.sin(aoa_az)
err_pos   = np.sqrt((east_hat - pos_e)**2 + (north_hat - pos_n)**2)

# ─── Monta tabela ─────────────────────────────────────────────────────────────
rows = []
for t in CATS:
    mask = ptype == t
    n    = int(mask.sum())
    if n == 0:
        continue
    pct       = 100 * n / n_ues
    rsrp_med  = float(np.nanmedian(rsrp[mask]))
    aoa_med   = float(np.nanmedian(err_3d[mask]))   if np.isfinite(err_3d[mask]).any() else float("nan")
    aoa_p95   = float(np.nanpercentile(err_3d[mask], 95)) if np.isfinite(err_3d[mask]).any() else float("nan")
    pos_med   = float(np.nanmedian(err_pos[mask]))  if np.isfinite(err_pos[mask]).any() else float("nan")
    pos_p95   = float(np.nanpercentile(err_pos[mask], 95)) if np.isfinite(err_pos[mask]).any() else float("nan")
    rows.append((t, n, pct, rsrp_med, aoa_med, aoa_p95, pos_med, pos_p95))

# Total
rows.append(("**Total**", n_ues, 100.0,
             float(np.nanmedian(rsrp)),
             float(np.nanmedian(err_3d[valid_mask])) if valid_mask.any() else float("nan"),
             float(np.nanpercentile(err_3d[valid_mask], 95)) if valid_mask.any() else float("nan"),
             float(np.nanmedian(err_pos[valid_mask])) if valid_mask.any() else float("nan"),
             float(np.nanpercentile(err_pos[valid_mask], 95)) if valid_mask.any() else float("nan")))

# ─── Escreve Markdown ─────────────────────────────────────────────────────────
header = (
    f"# Resumo Estatístico — Sionna RT Interlagos\n\n"
    f"**Frequência:** {freq_hz/1e9:.1f} GHz | "
    f"**UEs simulados:** {n_ues} | "
    f"**gNB:** 62 m AGL\n\n"
    f"**Estimador de posição:** "
    f"d̂ = 10^((EIRP − RSRP) / (10·n)) "
    f"com EIRP = {EIRP_DBM:.0f} dBm (P_tx 23 dBm + ganho 8 dBi − PL₀ 38 dB), "
    f"n = {N_PL}\n\n"
)

col_names = [
    "Categoria", "n", "%",
    "RSRP med (dBm)",
    "AoA err med (°)", "AoA err p95 (°)",
    "Pos err med (m)", "Pos err p95 (m)",
]
sep = [":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]

def fmt(v, decimals=1):
    if isinstance(v, float) and np.isnan(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)

table_lines = []
table_lines.append("| " + " | ".join(col_names) + " |")
table_lines.append("| " + " | ".join(sep) + " |")
for (t, n, pct, rsrp_med, aoa_med, aoa_p95, pos_med, pos_p95) in rows:
    row_vals = [
        t,
        fmt(n, 0),
        fmt(pct, 1),
        fmt(rsrp_med, 1),
        fmt(aoa_med, 1),
        fmt(aoa_p95, 1),
        fmt(pos_med, 0),
        fmt(pos_p95, 0),
    ]
    table_lines.append("| " + " | ".join(row_vals) + " |")

notes = (
    "\n\n## Notas\n\n"
    "- **LoS**: nenhuma interação no caminho dominante (linha de visada livre).\n"
    "- **diffracted**: algum hop de difração no caminho dominante. "
      "AoA preservado, RSRP reduzido pela difração.\n"
    "- **reflected**: algum hop especular/difuso (sem difração). "
      "AoA pode ser desviado para a superfície refletora.\n"
    "- **transmitted**: somente hops de refração — travessia de parede (modo O2I). "
      "AoA preservado com boa precisão, RSRP atenuado ~17 dB vs LoS.\n"
    "- **none**: nenhum caminho válido encontrado pelo PathSolver.\n"
    "- AoA err = erro angular 3D entre AoA medido e direção geométrica real "
      "(produto interno de vetores unitários esféricos).\n"
    "- Pos err = distância horizontal entre posição estimada e verdadeira.\n"
    "- Bad regime (reflected + diffracted): UEs com maior probabilidade de "
      "erro AoA elevado, candidatos a outlier no estimador de posição.\n"
)

os.makedirs("output", exist_ok=True)
content = header + "\n".join(table_lines) + notes
with open(args.out, "w") as f:
    f.write(content)
ok(f"Tabela salva: {args.out}")

# ─── Imprime na stdout ────────────────────────────────────────────────────────
print()
print("=" * 80)
print("RESUMO")
print("=" * 80)
for (t, n, pct, rsrp_med, aoa_med, aoa_p95, pos_med, pos_p95) in rows:
    print(f"  {t:<14}: n={n:4d} ({pct:5.1f}%) | RSRP={rsrp_med:6.1f} dBm | "
          f"AoA med={aoa_med:5.1f}° p95={aoa_p95:5.1f}° | "
          f"Pos med={pos_med:5.0f} m p95={pos_p95:5.0f} m")
print("=" * 80)
