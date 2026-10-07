"""
final_position_estimation.py — Figura matriz do paper: comparação de estimadores.

4 painéis organizados em 2×2:
  (a) CDF erro de posição — todos os estimadores, todos os UEs
  (b) Mediana do erro por categoria — gráfico de barras agrupadas
  (c) CDF LoS — estimadores comparados (cenário ideal)
  (d) Decomposição: erro radial vs tangencial para cada estimador

Estimadores comparados:
  1. RSRP naive      : d_hat = 10^((−7 − RSRP) / 27)   [baseline]
  2. RSRP compensado : d_hat = 10^((A_c − RSRP_comp) / (10·n_c))  [por categoria]
  3. ToA dom. path   : d_hat = c × τ_dom                [dominante]
  4. ToA 1ª chegada  : d_hat = c × τ_min                [primeira chegada]

Saída: output/final_position_estimation.png
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz",    default="output/measurements_v2.npz")
parser.add_argument("--delays", default="output/delays_v2.npz")
parser.add_argument("--params", default="output/pathloss_calibrated.json")
parser.add_argument("--comp",   default="output/pathloss_compensated.json")
parser.add_argument("--out",    default="output/final_position_estimation.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

C_LIGHT  = 299_792_458.
N_OLD, A_OLD = 2.7, -7.0

CATS   = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# Paleta dos estimadores
EST_COLORS = {
    "naive":    "#7f8c8d",
    "comp_los": "#27ae60",
    "toa_dom":  "#e74c3c",
    "toa_min":  "#8e44ad",
}
EST_LABELS = {
    "naive":    "RSRP naive (n=2.7)",
    "comp_los": "RSRP compensado LoS (n≈2)",
    "toa_dom":  "ToA dominante",
    "toa_min":  "ToA 1ª chegada",
}
EST_LS = {"naive": ":", "comp_los": "--", "toa_dom": "-.", "toa_min": "-"}
EST_LW = {"naive": 1.5, "comp_los": 2.0, "toa_dom": 2.0, "toa_min": 2.5}

# ─── Carrega dados ─────────────────────────────────────────────────────────────
info("Carregando dados...")
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]; ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)
freq_ghz = float(d["freq_hz"]) / 1e9

dl = np.load(args.delays, allow_pickle=True)
tau_dom = dl["tau_dom"]; tau_min = dl["tau_min"]
d_true  = dl["d_true"]

with open(args.params) as f: params_calib = json.load(f)
with open(args.comp)   as f: params_comp  = json.load(f)

# ─── Ganho elemento TR38901 (para RSRP compensado) ───────────────────────────
def ganho_elemento_tr38901_dbi(theta_zenital, phi_az):
    THETA_3DB = np.radians(65.); PHI_3DB = np.radians(65.)
    A_M = SLA_V = 30.; G_MAX = 8.
    phi_norm  = (phi_az + np.pi) % (2*np.pi) - np.pi
    theta_dev = theta_zenital - np.pi/2
    A_EV = -np.minimum(12.*(theta_dev/THETA_3DB)**2, SLA_V)
    A_EH = -np.minimum(12.*(phi_norm/PHI_3DB)**2, A_M)
    return G_MAX - np.minimum(-(A_EH + A_EV), A_M)

valid_aoa = np.isfinite(aoa_az) & np.isfinite(aoa_el)
theta_t = np.pi/2 - aoa_el; phi_t = aoa_az
G_elem  = np.full(n_ues, np.nan)
G_elem[valid_aoa] = ganho_elemento_tr38901_dbi(theta_t[valid_aoa], phi_t[valid_aoa])
rsrp_comp_all = rsrp - G_elem   # RSRP compensado (todos os UEs)

# ─── d_hat por estimador ──────────────────────────────────────────────────────
def d_hat_rsrp_comp_por_cat(rsrp_comp_vals):
    """Usa parâmetros calibrados por categoria para RSRP compensado."""
    d_hat = np.full(n_ues, np.nan)
    for t in CATS:
        if t not in params_comp: continue
        p = params_comp[t]
        mask = ptype == t
        d_hat[mask] = 10 ** ((p["A"] - rsrp_comp_vals[mask]) / (10 * p["n"]))
    # UEs sem categoria: usa global compensado
    sem_cat = ~np.isin(ptype, CATS)
    if sem_cat.any() and "global" in params_comp:
        p = params_comp["global"]
        d_hat[sem_cat] = 10 ** ((p["A"] - rsrp_comp_vals[sem_cat]) / (10 * p["n"]))
    return d_hat

# RSRP compensado — usa parâmetros LoS para TODOS os UEs (pois é o regime bem calibrado)
# Justificativa: queremos mostrar o melhor caso RSRP-compensado sem oracle de categoria
p_los = params_comp["LoS"]
d_hat_comp_los_all = 10 ** ((p_los["A"] - rsrp_comp_all) / (10 * p_los["n"]))

d_hat_naive   = 10 ** ((A_OLD - rsrp) / (10 * N_OLD))
d_hat_toa_dom = C_LIGHT * tau_dom
d_hat_toa_min = C_LIGHT * tau_min

# ─── Posição estimada 2D para cada estimador ─────────────────────────────────
def pos_2d(d_hat_vals):
    theta_aoa = np.pi/2 - aoa_el
    e_h = gnb[0] + d_hat_vals * np.sin(theta_aoa) * np.cos(aoa_az)
    n_h = gnb[1] + d_hat_vals * np.sin(theta_aoa) * np.sin(aoa_az)
    return e_h, n_h

def err_2d(e_h, n_h):
    return np.sqrt((e_h - pos_e)**2 + (n_h - pos_n)**2)

err = {}
for key, dh in [("naive",    d_hat_naive),
                ("comp_los", d_hat_comp_los_all),
                ("toa_dom",  d_hat_toa_dom),
                ("toa_min",  d_hat_toa_min)]:
    e_h, n_h = pos_2d(dh)
    err[key] = err_2d(e_h, n_h)

# ─── Decomposição err_radial e err_tang (para estimadores de distância) ───────
def decomp(d_hat_vals):
    """Retorna (err_radial, err_tang_aoa_3d)."""
    err_rad = np.abs(d_hat_vals - d_true)
    # Erro AoA 3D
    delta_e = pos_e - gnb[0]; delta_n = pos_n - gnb[1]
    delta_u = np.full(n_ues, 1.5) - gnb[2]
    d3 = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)
    phi_tr  = np.arctan2(delta_n, delta_e)
    th_tr   = np.arccos(np.clip(delta_u/np.maximum(d3,1e-6), -1, 1))
    el_tr   = np.pi/2 - th_tr

    def sph2vec(el, az):
        th = np.pi/2 - el
        return np.column_stack([np.sin(th)*np.cos(az), np.sin(th)*np.sin(az), np.cos(th)])

    vm = valid_aoa & (ptype != 'none')
    err_aoa = np.full(n_ues, np.nan)
    if vm.sum() > 0:
        mv = sph2vec(aoa_el[vm], aoa_az[vm])
        tv = sph2vec(el_tr[vm],  phi_tr[vm])
        dot = np.clip((mv*tv).sum(axis=1), -1., 1.)
        err_aoa[vm] = np.arccos(dot)

    err_tang = d3 * np.sin(np.minimum(np.abs(err_aoa), np.pi - np.abs(err_aoa)))
    return err_rad, err_tang

err_rad_naive,   err_tang_all   = decomp(d_hat_naive)
err_rad_comp,    _              = decomp(d_hat_comp_los_all)
err_rad_toa_dom, _              = decomp(d_hat_toa_dom)
err_rad_toa_min, _              = decomp(d_hat_toa_min)

# ─── Log resumo ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"{'Estimador':<24} {'Todos med':>9} {'Todos p95':>9} {'LoS med':>9}")
print("-" * 70)
for key in ["naive", "comp_los", "toa_dom", "toa_min"]:
    e = err[key]
    vt = np.isfinite(e)
    vl = np.isfinite(e) & (ptype == "LoS")
    med = np.nanmedian(e[vt]); p95 = np.nanpercentile(e[vt], 95)
    med_los = np.nanmedian(e[vl]) if vl.any() else float("nan")
    info(f"  {EST_LABELS[key]:<22}  {med:>9.1f}  {p95:>9.1f}  {med_los:>9.1f}")
print("=" * 70)

# ─── Plot 2×2 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"Estimação de Posição: RSRP vs ToA — {freq_ghz:.1f} GHz | {n_ues} UEs",
    fontsize=13, fontweight="bold"
)

# (a) CDF agregado — todos os UEs, todos os estimadores
ax = axes[0, 0]
ax.set_title("(a) CDF erro de posição — todos os UEs")
for key in ["naive", "comp_los", "toa_dom", "toa_min"]:
    e = err[key]; valid = np.isfinite(e)
    vals = np.sort(e[valid])
    ax.plot(vals, np.arange(1,len(vals)+1)/len(vals),
            color=EST_COLORS[key], ls=EST_LS[key], lw=EST_LW[key],
            label=EST_LABELS[key])
ax.axhline(0.50, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Erro de posição (m)"); ax.set_ylabel("CDF")
ax.set_xlim(0, 600); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

# (b) Mediana por categoria — barras agrupadas
ax = axes[0, 1]
ax.set_title("(b) Mediana do erro por categoria")
x = np.arange(len(CATS)); w = 0.18
for ki, key in enumerate(["naive", "comp_los", "toa_dom", "toa_min"]):
    meds = []
    for t in CATS:
        mask = (ptype == t) & np.isfinite(err[key])
        meds.append(float(np.nanmedian(err[key][mask])) if mask.any() else 0)
    bars = ax.bar(x + (ki - 1.5) * w, meds, w,
                  color=EST_COLORS[key], alpha=0.85, edgecolor="k", lw=0.5,
                  label=EST_LABELS[key])

ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9)
ax.set_ylabel("Mediana do erro (m)")
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3, axis="y")

# Anota n por categoria
for xi, t in enumerate(CATS):
    n_cat = int((ptype == t).sum())
    ax.text(xi, -25, f"n={n_cat}", ha="center", fontsize=7, color="gray")

# (c) CDF LoS — detalhe do regime mais favorável
ax = axes[1, 0]
ax.set_title("(c) CDF LoS — zoom no regime favorável")
mask_los = ptype == "LoS"
for key in ["naive", "comp_los", "toa_dom", "toa_min"]:
    e = err[key]; valid = mask_los & np.isfinite(e)
    if not valid.any(): continue
    vals = np.sort(e[valid])
    ax.plot(vals, np.arange(1,len(vals)+1)/len(vals),
            color=EST_COLORS[key], ls=EST_LS[key], lw=EST_LW[key],
            label=f"{EST_LABELS[key]} (med={np.nanmedian(e[valid]):.1f} m)")
ax.axhline(0.50, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Erro de posição (m)"); ax.set_ylabel("CDF")
ax.set_xlim(0, 300); ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
ax.text(0.97, 0.03,
    f"LoS: n={mask_los.sum()} UEs ({100*mask_los.mean():.1f}%)",
    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

# (d) Decomposição: barras empilhadas radial + tang por estimador e categoria
ax = axes[1, 1]
ax.set_title("(d) Decomposição radial vs tangencial — estimador vs categoria")

# Para cada estimador, calcula mediana radial e tangencial por categoria
n_ests = 4
n_cats = len(CATS)
x = np.arange(n_cats)
w = 0.18

rad_data  = {
    "naive":    [float(np.nanmedian(err_rad_naive[   (ptype==t) & np.isfinite(err_rad_naive)]))    if (ptype==t).any() else 0 for t in CATS],
    "comp_los": [float(np.nanmedian(err_rad_comp[    (ptype==t) & np.isfinite(err_rad_comp)]))     if (ptype==t).any() else 0 for t in CATS],
    "toa_dom":  [float(np.nanmedian(err_rad_toa_dom[ (ptype==t) & np.isfinite(err_rad_toa_dom)])) if (ptype==t).any() else 0 for t in CATS],
    "toa_min":  [float(np.nanmedian(err_rad_toa_min[ (ptype==t) & np.isfinite(err_rad_toa_min)])) if (ptype==t).any() else 0 for t in CATS],
}
tang_meds = [float(np.nanmedian(err_tang_all[(ptype==t) & np.isfinite(err_tang_all)])) if (ptype==t).any() else 0 for t in CATS]

for ki, key in enumerate(["naive", "comp_los", "toa_dom", "toa_min"]):
    xpos = x + (ki - 1.5) * w
    rad_v = rad_data[key]
    # Barra sólida = radial, hatch = tangencial (igual para todos)
    ax.bar(xpos, rad_v, w, color=EST_COLORS[key], alpha=0.9, edgecolor="k", lw=0.5)
    ax.bar(xpos, tang_meds, w, bottom=rad_v, color=EST_COLORS[key], alpha=0.35,
           edgecolor="k", lw=0.5, hatch="//")

ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9)
ax.set_ylabel("Mediana (m)")
ax.grid(True, alpha=0.3, axis="y")

# Legenda manual: estilos de estimador + radial/tang
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
hand = [Patch(fc=EST_COLORS[k], label=EST_LABELS[k]) for k in ["naive","comp_los","toa_dom","toa_min"]]
hand += [Patch(fc="gray", alpha=0.9, label="Radial |d̂−d|"),
         Patch(fc="gray", alpha=0.35, hatch="//", label="Tangencial (AoA)")]
ax.legend(handles=hand, fontsize=6.5, loc="upper right", ncol=2)

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")
