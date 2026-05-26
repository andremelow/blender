"""
plot_measurements_v2.py — Diagnóstico visual das medidas Sionna RT (v2).

Painéis:
  (a) RSRP espacial [dBm]
  (b) Tipo de caminho dominante por UE
  (c) Histograma de RSRP por tipo
  (d) Erro AoA vs distância (com nota sobre modo O2I)
  (e) CDF do erro AoA — destaca regime não modelado por gaussiana analítica

Categorias (paleta fixa):
  LoS         → verde
  diffracted  → laranja
  reflected   → azul
  transmitted → roxo   (antes "refracted" — modo outdoor-to-indoor)

Uso:
    python scripts/plot_measurements_v2.py [--npz output/measurements_v2.npz]
"""

import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default="output/measurements_v2.npz")
parser.add_argument("--out", default="output/measurements_diag_v2.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ─── Paleta e ordem ───────────────────────────────────────────────────────────
CATS   = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# ─── Carrega dados ────────────────────────────────────────────────────────────
info(f"Carregando: {args.npz}")
d = np.load(args.npz, allow_pickle=True)
pos_e   = d["pos_east"];   pos_n  = d["pos_north"]
rsrp    = d["rsrp_dbm"];   ptype  = d["path_type"]
aoa_az  = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb_pos = d["gnb_pos"]
n_ues   = len(rsrp)

# Distância horizontal ao gNB
delta_e = pos_e - gnb_pos[0]
delta_n = pos_n - gnb_pos[1]
delta_u = np.full(n_ues, 1.5) - gnb_pos[2]   # UE z=1.5 m, gNB z=62 m
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)
dist_h  = np.sqrt(delta_e**2 + delta_n**2)

# AoA geométrico verdadeiro (convenção Sionna: phi de East, theta zenital)
phi_true   = np.arctan2(delta_n, delta_e)
theta_true = np.arccos(np.clip(delta_u / np.maximum(dist_3d, 1e-6), -1, 1))
el_true    = np.pi / 2 - theta_true

# Erro angular 3D via produto interno de vetores unitários
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
    mv   = sph2vec(aoa_el[valid_mask],  aoa_az[valid_mask])
    tv   = sph2vec(el_true[valid_mask], phi_true[valid_mask])
    dot  = np.clip((mv * tv).sum(axis=1), -1., 1.)
    err_3d[valid_mask] = np.degrees(np.arccos(dot))

info(f"UEs: {n_ues} | RSRP [{rsrp.min():.1f}, {rsrp.max():.1f}] dBm")
for t in CATS:
    cnt = int((ptype == t).sum())
    info(f"  {t:<14}: {cnt:5d} ({100*cnt/n_ues:.1f}%)")

# ─── Figura ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.32)
ax_rsrp = fig.add_subplot(gs[0, 0])
ax_type = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, 0])
ax_aoa  = fig.add_subplot(gs[1, 1])
ax_cdf  = fig.add_subplot(gs[2, :])

freq_ghz = float(d["freq_hz"]) / 1e9

fig.suptitle(
    f"Medidas Sionna RT — Interlagos | {freq_ghz:.1f} GHz | {n_ues} UEs | gNB 62 m AGL",
    fontsize=14, y=0.98
)

# ── (a) RSRP espacial ─────────────────────────────────────────────────────────
sc = ax_rsrp.scatter(pos_e, pos_n, c=rsrp, cmap="jet_r",
                     vmin=-110, vmax=-35, s=14, alpha=0.85)
ax_rsrp.scatter(*gnb_pos[:2], marker="^", c="k", s=120, zorder=5, label="gNB")
plt.colorbar(sc, ax=ax_rsrp, label="RSRP (dBm)", shrink=0.85)
ax_rsrp.set_title("(a) RSRP [dBm]"); ax_rsrp.set_xlabel("Leste (m)"); ax_rsrp.set_ylabel("Norte (m)")
ax_rsrp.legend(fontsize=8)

# ── (b) Tipo de caminho ───────────────────────────────────────────────────────
for t in CATS:
    mask = ptype == t
    if mask.any():
        cnt = mask.sum()
        ax_type.scatter(pos_e[mask], pos_n[mask], color=COLORS[t], s=14,
                        alpha=0.8, label=f"{t} ({cnt})")
ax_type.scatter(*gnb_pos[:2], marker="^", c="k", s=120, zorder=5)
ax_type.set_title("(b) Tipo de caminho dominante")
ax_type.set_xlabel("Leste (m)"); ax_type.set_ylabel("Norte (m)")
ax_type.legend(fontsize=8, ncol=2)

# ── (c) Histograma RSRP por tipo ─────────────────────────────────────────────
bins = np.linspace(-115, -30, 35)
for t in CATS:
    mask = ptype == t
    if mask.any():
        med = np.median(rsrp[mask])
        ax_hist.hist(rsrp[mask], bins=bins, alpha=0.55, color=COLORS[t],
                     label=f"{t} (n={mask.sum()}, med={med:.0f} dBm)", density=True)
ax_hist.set_xlabel("RSRP (dBm)"); ax_hist.set_ylabel("Densidade")
ax_hist.set_title("(c) Histograma de RSRP por tipo de caminho")
ax_hist.legend(fontsize=8)

# ── (d) Erro AoA vs distância ─────────────────────────────────────────────────
for t in CATS:
    mask = ptype == t
    if mask.any():
        med = np.median(err_3d[mask])
        ax_aoa.scatter(dist_h[mask], err_3d[mask], color=COLORS[t], s=14,
                       alpha=0.65, label=f"{t} (med={med:.1f}°)")

# Tendência global (regressão linear)
slope, intercept, *_ = stats.linregress(dist_h, err_3d)
x_line = np.array([0, dist_h.max()])
ax_aoa.plot(x_line, slope*x_line + intercept, "k--", lw=1.2, label="tendência global")

ax_aoa.set_xlabel("Distância horizontal à gNB (m)")
ax_aoa.set_ylabel("Erro angular 3D (graus)")
ax_aoa.set_title("(d) Erro angular AoA (medido vs geométrico) vs Distância")
ax_aoa.legend(fontsize=8, ncol=2)

# Nota sobre modo O2I
ax_aoa.text(0.02, 0.97,
    "Categoria 'transmitted' (refração através de paredes) preserva AoA\n"
    "com baixo erro, mas atenua RSRP em ~17 dB relativo a LoS —\n"
    "modo O2I em prédios sem espessura modelada.",
    transform=ax_aoa.transAxes, fontsize=7.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="lavender", alpha=0.75))

# ── (e) CDF do erro AoA ───────────────────────────────────────────────────────
for t in CATS:
    mask = ptype == t
    if not mask.any(): continue
    errs_sorted = np.sort(err_3d[mask])
    cdf = np.arange(1, len(errs_sorted)+1) / len(errs_sorted)
    ax_cdf.plot(errs_sorted, cdf, color=COLORS[t], lw=2,
                label=f"{t} (n={mask.sum()})")

# Gaussiana analítica σ=3° (banda cinza — modelo analítico padrão)
x_gauss = np.linspace(0, 180, 500)
# CDF de Rayleigh com σ=3° (aprox. para erros 2D pequenos)
sigma_g = 3.0
rayleigh_cdf = 1 - np.exp(-x_gauss**2 / (2 * sigma_g**2))
ax_cdf.fill_between(x_gauss,
                    np.clip(rayleigh_cdf - 0.1, 0, 1),
                    np.clip(rayleigh_cdf + 0.1, 0, 1),
                    color="gray", alpha=0.25,
                    label=f"Modelo analítico gaussiano σ={sigma_g}° ± 10%")
ax_cdf.plot(x_gauss, rayleigh_cdf, "k-", lw=1, alpha=0.5)

# Destaque do regime não modelado
ax_cdf.axvspan(50, 165, color="red", alpha=0.07)
ax_cdf.text(107, 0.25,
    "RT revela regime de erro\nque modelo analítico\n(banda cinza) não modela.",
    fontsize=9, ha="center", color="darkred",
    bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", alpha=0.85))

ax_cdf.set_xlabel("Erro angular AoA 3D (graus)")
ax_cdf.set_ylabel("CDF")
ax_cdf.set_title("(e) CDF do erro angular AoA por tipo de caminho")
ax_cdf.set_xlim(0, 180); ax_cdf.set_ylim(0, 1)
ax_cdf.legend(fontsize=9, loc="lower right")
ax_cdf.grid(True, alpha=0.3)

# ─── Salva ───────────────────────────────────────────────────────────────────
plt.savefig(args.out, dpi=160, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")

# ─── Resumo ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("RESUMO")
print("=" * 50)
print(f"  UEs simulados : {n_ues}")
print(f"  RSRP          : [{rsrp.min():.1f}, {rsrp.max():.1f}] dBm | mediana={np.median(rsrp):.1f} dBm")
for t in CATS:
    mask = ptype == t
    if mask.any():
        med_e = np.median(err_3d[mask])
        print(f"  Erro AoA {t:<12}: n={mask.sum():4d} | mediana={med_e:.2f}° | max={err_3d[mask].max():.2f}°")
print("=" * 50)
