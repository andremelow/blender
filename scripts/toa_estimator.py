"""
toa_estimator.py — Estimador de posição baseado em ToA (Time of Arrival).

d_hat_toa = c × tau_0

Dois modos de tau_0:
  dom : delay do caminho dominante (maior potência) — mesmo critério do RSRP
  min : delay da primeira chegada (menor delay entre caminhos válidos)

A primeira chegada corresponde ao caminho LoS mesmo em UEs classificados como
"reflected" ou "diffracted" — a componente LoS existe mas com menor potência.
Isso implica que tau_min é quase sempre ≈ d_true/c e err_radial ≈ 0.

O erro de posição residual vem da componente tangencial (erro de AoA):
  err_tang = d_true × sin(min(|err_aoa|, π − |err_aoa|))

Saídas:
  output/toa_estimator.png  — 4 painéis
  output/toa_summary.md     — tabela markdown
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz",     default="output/measurements_v2.npz")
parser.add_argument("--delays",  default="output/delays_v2.npz")
parser.add_argument("--params",  default="output/pathloss_calibrated.json")
parser.add_argument("--comp",    default="output/pathloss_compensated.json")
parser.add_argument("--out",     default="output/toa_estimator.png")
parser.add_argument("--md",      default="output/toa_summary.md")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

C_LIGHT = 299_792_458.   # m/s
N_OLD, A_OLD = 2.7, -7.0

CATS   = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# ─── Carrega dados ─────────────────────────────────────────────────────────────
info(f"Carregando medidas: {args.npz}")
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]; ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)
freq_ghz = float(d["freq_hz"]) / 1e9

info(f"Carregando delays: {args.delays}")
dl = np.load(args.delays, allow_pickle=True)
tau_dom = dl["tau_dom"]   # delay do caminho dominante (s)
tau_min = dl["tau_min"]   # menor delay = primeira chegada (s)
d_true  = dl["d_true"]    # distância geométrica 3D (m)

with open(args.params) as f: params_calib = json.load(f)
with open(args.comp)   as f: params_comp  = json.load(f)

n_g = params_calib["global"]["n"]; A_g = params_calib["global"]["A"]

# ─── Distâncias estimadas pelos diferentes observáveis ────────────────────────
# ToA caminho dominante
d_hat_toa_dom = C_LIGHT * tau_dom

# ToA primeira chegada
d_hat_toa_min = C_LIGHT * tau_min

# RSRP naive (n=2.7, A=-7 dBm)
d_hat_rsrp_old = 10 ** ((A_OLD - rsrp) / (10 * N_OLD))

# RSRP calibrado global
d_hat_rsrp_cal = 10 ** ((A_g - rsrp) / (10 * n_g))

# ─── Função de estimativa de posição 2D ───────────────────────────────────────
def pos_2d_de_d_hat(d_hat_vals):
    """Usa d_hat + AoA para estimar posição 2D (East, North)."""
    theta_aoa = np.pi / 2 - aoa_el       # ângulo zenital
    e_hat = gnb[0] + d_hat_vals * np.sin(theta_aoa) * np.cos(aoa_az)
    n_hat = gnb[1] + d_hat_vals * np.sin(theta_aoa) * np.sin(aoa_az)
    return e_hat, n_hat

def erro_pos_2d(e_hat, n_hat):
    return np.sqrt((e_hat - pos_e)**2 + (n_hat - pos_n)**2)

# ─── Erro de posição para cada estimador ──────────────────────────────────────
e_old, n_old = pos_2d_de_d_hat(d_hat_rsrp_old)
err_rsrp_old = erro_pos_2d(e_old, n_old)

e_cal, n_cal = pos_2d_de_d_hat(d_hat_rsrp_cal)
err_rsrp_cal = erro_pos_2d(e_cal, n_cal)

e_tdom, n_tdom = pos_2d_de_d_hat(d_hat_toa_dom)
err_toa_dom = erro_pos_2d(e_tdom, n_tdom)

e_tmin, n_tmin = pos_2d_de_d_hat(d_hat_toa_min)
err_toa_min = erro_pos_2d(e_tmin, n_tmin)

# Erro de distância (radial)
err_d_dom = np.abs(d_hat_toa_dom - d_true)
err_d_min = np.abs(d_hat_toa_min - d_true)

# ─── Resumo estatístico ───────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"{'Estimador':<28} {'med (m)':>9} {'p75 (m)':>9} {'p95 (m)':>9}")
print("-" * 80)
for nome, err in [
    ("RSRP naive n=2.7",    err_rsrp_old),
    ("RSRP global calib.",  err_rsrp_cal),
    ("ToA dom. path",       err_toa_dom),
    ("ToA 1ª chegada",      err_toa_min),
]:
    valid = np.isfinite(err)
    med = float(np.nanmedian(err[valid]))
    p75 = float(np.nanpercentile(err[valid], 75))
    p95 = float(np.nanpercentile(err[valid], 95))
    print(f"  {nome:<26}  {med:>9.1f}  {p75:>9.1f}  {p95:>9.1f}")
print("=" * 80)

# Por categoria
print()
print("─── Erro de posição por categoria (mediana / p95, em metros) ───")
header = f"{'Cat':<14}"
for nome in ["RSRP naive", "RSRP calib", "ToA dom", "ToA min"]:
    header += f"  {nome:>10}"
print(header)
for t in CATS:
    mask = (ptype == t)
    if not mask.any(): continue
    row = f"  {t:<12}"
    for err_arr in [err_rsrp_old, err_rsrp_cal, err_toa_dom, err_toa_min]:
        valid = mask & np.isfinite(err_arr)
        if valid.any():
            row += f"  {np.nanmedian(err_arr[valid]):>10.1f}"
        else:
            row += f"  {'—':>10}"
    print(row)

# ─── Plot: 4 painéis ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"Estimador de Posição: RSRP vs ToA — {freq_ghz:.1f} GHz | {n_ues} UEs",
    fontsize=13
)

# Mapeamento de estilos de linha por estimador
estilos = [
    (err_rsrp_old, "#7f8c8d", ":", 1.5, f"RSRP naive (n={N_OLD}, A={A_OLD:.0f} dBm)"),
    (err_rsrp_cal, "#2980b9", "--", 2.0, f"RSRP global calib. (n={n_g:.2f}, A={A_g:.1f} dBm)"),
    (err_toa_dom,  "#e74c3c", "-.", 2.0, "ToA dom. path (τ maior potência)"),
    (err_toa_min,  "#27ae60", "-",  2.5, "ToA 1ª chegada (τ mínimo)"),
]

# (a) CDF agregado — todos os UEs
ax = axes[0, 0]
ax.set_title("(a) CDF agregado — todos os UEs")
for (err_arr, col, ls, lw, lbl) in estilos:
    valid = np.isfinite(err_arr)
    vals  = np.sort(err_arr[valid])
    cdf   = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, color=col, ls=ls, lw=lw, label=lbl)
ax.axhline(0.5,  color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Erro de posição (m)"); ax.set_ylabel("CDF")
ax.set_xlim(0, 500); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

# (b) CDF por categoria — ToA 1ª chegada vs RSRP naive
ax = axes[0, 1]
ax.set_title("(b) ToA 1ª chegada vs RSRP naive — por categoria")
for t in CATS:
    mask = ptype == t
    if not mask.any(): continue
    col = COLORS[t]
    # RSRP naive (tracejado)
    v1 = np.sort(err_rsrp_old[mask & np.isfinite(err_rsrp_old)])
    if len(v1) > 0:
        ax.plot(v1, np.arange(1,len(v1)+1)/len(v1), color=col, ls=":", lw=1.5, alpha=0.7)
    # ToA min (sólido)
    v2 = np.sort(err_toa_min[mask & np.isfinite(err_toa_min)])
    if len(v2) > 0:
        ax.plot(v2, np.arange(1,len(v2)+1)/len(v2), color=col, ls="-", lw=2.2,
                label=f"{t} (n={mask.sum()})")

ax.axhline(0.5,  color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Erro de posição (m)"); ax.set_ylabel("CDF")
ax.set_xlim(0, 500); ax.set_ylim(0, 1)
ax.text(0.03, 0.97, "— ToA 1ª chegada\n··· RSRP naive",
        transform=ax.transAxes, fontsize=8, va="top")
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

# (c) Erro radial: d_hat vs d_true (separado entre ToA dom e min)
ax = axes[1, 0]
ax.set_title("(c) Erro radial |d̂ − d_true| — ToA dom vs min")
for (err_d, col, ls, lw, lbl) in [
    (err_d_dom, "#e74c3c", "-.", 2.0, "ToA dom. path"),
    (err_d_min, "#27ae60", "-",  2.5, "ToA 1ª chegada"),
]:
    for t in CATS:
        mask  = (ptype == t) & np.isfinite(err_d)
        if not mask.any(): continue
        vals  = np.sort(err_d[mask])
        cdf   = np.arange(1, len(vals)+1)/len(vals)
        ax.plot(vals, cdf, color=COLORS[t], ls=ls, lw=lw, alpha=0.85)

# Legenda manual
from matplotlib.lines import Line2D
leg_handles = [Line2D([0],[0], color="k", ls="-.", lw=2, label="ToA dom."),
               Line2D([0],[0], color="k", ls="-",  lw=2, label="ToA 1ª chegada")]
leg_handles += [Line2D([0],[0], color=COLORS[t], lw=2, label=t) for t in CATS]
ax.legend(handles=leg_handles, fontsize=7, loc="lower right")
ax.set_xlabel("|d̂ − d_true| (m)"); ax.set_ylabel("CDF")
ax.set_xlim(0, 200); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# (d) Scatter: erro radial vs erro de posição 2D (ToA 1ª chegada)
ax = axes[1, 1]
ax.set_title("(d) Erro radial vs posição 2D — ToA 1ª chegada")
for t in CATS:
    mask = (ptype == t) & np.isfinite(err_d_min) & np.isfinite(err_toa_min)
    if not mask.any(): continue
    ax.scatter(err_d_min[mask], err_toa_min[mask], color=COLORS[t],
               s=15, alpha=0.6, label=f"{t} (n={mask.sum()})")

lim = min(np.nanpercentile(err_d_min[np.isfinite(err_d_min)], 99) * 1.1, 300)
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.4, label="err_2D = err_rad")
ax.set_xlabel("|d̂ − d_true| (m)")
ax.set_ylabel("Erro de posição 2D (m)")
ax.set_xlim(0, lim); ax.set_ylim(0, min(600, err_toa_min[np.isfinite(err_toa_min)].max() * 1.05))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.97, 0.05,
    "Pontos acima da bissetriz:\nerro 2D dominado por AoA\n(radial ≈ 0 mas AoA errado)",
    transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")

# ─── Tabela Markdown ──────────────────────────────────────────────────────────
md  = f"# Estimador ToA vs RSRP — {freq_ghz:.1f} GHz\n\n"
md += f"**c × τ_0**: velocidade da luz × delay de propagação absoluto\n"
md += f"**τ_dom**: caminho dominante (maior potência)\n"
md += f"**τ_min**: primeira chegada (menor delay)\n\n"
md += "| Estimador | Todos med (m) | Todos p95 (m) |\n"
md += "|:---|---:|---:|\n"
for nome, err in [
    ("RSRP naive n=2.7", err_rsrp_old),
    ("RSRP global calib.", err_rsrp_cal),
    ("ToA dom. path", err_toa_dom),
    ("ToA 1ª chegada", err_toa_min),
]:
    valid = np.isfinite(err)
    med = float(np.nanmedian(err[valid]))
    p95 = float(np.nanpercentile(err[valid], 95))
    md += f"| {nome} | {med:.1f} | {p95:.1f} |\n"

md += "\n## Por categoria — mediana do erro de posição (m)\n\n"
md += "| Categoria | n | RSRP naive | RSRP calib | ToA dom | **ToA min** |\n"
md += "|:---|---:|---:|---:|---:|---:|\n"
for t in CATS:
    mask = (ptype == t)
    if not mask.any(): continue
    row = f"| {t} | {mask.sum()} |"
    for err_arr in [err_rsrp_old, err_rsrp_cal, err_toa_dom, err_toa_min]:
        valid = mask & np.isfinite(err_arr)
        if valid.any():
            row += f" {np.nanmedian(err_arr[valid]):.1f} |"
        else:
            row += " — |"
    md += row + "\n"

md += "\n## Achado central\n\n"
md += ("τ_min (primeira chegada) corresponde ao caminho LoS em quase todos os UEs "
       "— mesmo naqueles classificados como 'reflected' ou 'diffracted', a componente LoS "
       "existe e chega primeiro, apenas com menor potência que o caminho refletido/difratado. "
       "Resultado: `d_hat_toa = c × τ_min ≈ d_true` para todos os tipos de caminho. "
       "O erro de posição residual é dominado pelo erro de AoA (direção do caminho dominante).\n")

os.makedirs("output", exist_ok=True)
with open(args.md, "w") as f:
    f.write(md)
ok(f"Markdown salvo: {args.md}")
