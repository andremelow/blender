"""
analyze_bad_regime.py — Análise do regime "bad AoA" (reflected + diffracted).

Painéis:
  (a) Mapa espacial: todos UEs em cinza, reflected/diffracted destacados
  (b) RSRP vs erro AoA (UEs bad regime), linha de tendência por categoria
  (c) Distância vs erro AoA (UEs bad regime), linha de tendência por categoria
  (d) n_paths vs erro AoA (UEs bad regime)
  (e) Mapa de erros de posição (setas true→estimado) para UEs bad regime
  (f) CDF do erro de posição por categoria (com linhas mediana e p95)

Estimador de posição:
  d_hat = 10^((EIRP - RSRP_dBm) / (10 * n))
  EIRP = -7 dBm  (P_tx 23 dBm + ganho 8 dBi - PL0 38 dB)
  n    = 2.7     (expoente de path loss urbano)
  Direção: AoA medido (phi_t, theta_t) por reciprocidade

Uso:
    python scripts/analyze_bad_regime.py [--npz output/measurements_v2.npz]
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
parser.add_argument("--out", default="output/bad_regime_analysis.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ─── Parâmetros do estimador de posição ───────────────────────────────────────
EIRP_DBM = -7.0    # 23 dBm (P_tx) + 8 dBi (ganho) − 38 dB (PL0 @ 1 m)
N_PL     = 2.7     # expoente de path loss urbano

# ─── Paleta (consistente com plot_measurements_v2.py) ────────────────────────
CATS   = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}
BAD_CATS = ["reflected", "diffracted"]

# ─── Carrega dados ─────────────────────────────────────────────────────────────
info(f"Carregando: {args.npz}")
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"];   pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"];   ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
n_paths_arr = d["n_paths"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)
info(f"UEs: {n_ues}")

# ─── Geometria verdadeira ─────────────────────────────────────────────────────
delta_e = pos_e - gnb[0]
delta_n = pos_n - gnb[1]
delta_u = np.full(n_ues, 1.5) - gnb[2]
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)
dist_h  = np.sqrt(delta_e**2 + delta_n**2)

phi_true   = np.arctan2(delta_n, delta_e)
theta_true = np.arccos(np.clip(delta_u / np.maximum(dist_3d, 1e-6), -1, 1))
el_true    = np.pi / 2 - theta_true

# ─── Erro AoA 3D ──────────────────────────────────────────────────────────────
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

# ─── Estimativa de posição para todos os UEs ─────────────────────────────────
d_hat      = 10 ** ((EIRP_DBM - rsrp) / (10 * N_PL))   # distância estimada (m)
theta_aoa  = np.pi / 2 - aoa_el                          # ângulo zenital
east_hat   = gnb[0] + d_hat * np.sin(theta_aoa) * np.cos(aoa_az)
north_hat  = gnb[1] + d_hat * np.sin(theta_aoa) * np.sin(aoa_az)
err_pos    = np.sqrt((east_hat - pos_e)**2 + (north_hat - pos_n)**2)  # (m)

for t in CATS:
    mask = ptype == t
    cnt  = int(mask.sum())
    if cnt:
        med_e = float(np.nanmedian(err_3d[mask]))
        med_p = float(np.nanmedian(err_pos[mask]))
        p95_p = float(np.nanpercentile(err_pos[mask], 95))
        info(f"  {t:<12}: n={cnt:4d} | AoA err med={med_e:.1f}° | pos err med={med_p:.0f} m | p95={p95_p:.0f} m")

bad_mask = np.isin(ptype, BAD_CATS)
info(f"Bad regime (reflected+diffracted): {bad_mask.sum()} UEs ({100*bad_mask.sum()/n_ues:.1f}%)")

# ─── Figura ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
ax_map  = fig.add_subplot(gs[0, 0])
ax_rsrp = fig.add_subplot(gs[0, 1])
ax_dist = fig.add_subplot(gs[1, 0])
ax_np   = fig.add_subplot(gs[1, 1])
ax_arr  = fig.add_subplot(gs[2, 0])
ax_cdf  = fig.add_subplot(gs[2, 1])

freq_ghz = float(d["freq_hz"]) / 1e9
fig.suptitle(
    f"Análise do Regime Bad-AoA — Interlagos | {freq_ghz:.1f} GHz | {n_ues} UEs | gNB 62 m AGL",
    fontsize=13, y=0.99
)

# ── (a) Mapa espacial ─────────────────────────────────────────────────────────
ax_map.scatter(pos_e, pos_n, color="lightgray", s=10, alpha=0.6, label="outros")
for t in BAD_CATS:
    mask = ptype == t
    if mask.any():
        ax_map.scatter(pos_e[mask], pos_n[mask], color=COLORS[t], s=24,
                       alpha=0.9, label=f"{t} (n={mask.sum()})", zorder=3)
ax_map.scatter(*gnb[:2], marker="^", c="k", s=130, zorder=5, label="gNB")
ax_map.set_title("(a) Mapa espacial — bad regime destacado")
ax_map.set_xlabel("Leste (m)"); ax_map.set_ylabel("Norte (m)")
ax_map.legend(fontsize=8)

# ── (b) RSRP vs erro AoA ──────────────────────────────────────────────────────
for t in BAD_CATS:
    mask = ptype == t
    if not mask.any(): continue
    valid = mask & np.isfinite(err_3d)
    ax_rsrp.scatter(rsrp[valid], err_3d[valid], color=COLORS[t], s=18,
                    alpha=0.7, label=f"{t} (n={valid.sum()})")
    if valid.sum() > 2:
        sl, ic, *_ = stats.linregress(rsrp[valid], err_3d[valid])
        xr = np.array([rsrp[valid].min(), rsrp[valid].max()])
        ax_rsrp.plot(xr, sl*xr + ic, color=COLORS[t], lw=1.8, ls="--")

ax_rsrp.set_xlabel("RSRP (dBm)"); ax_rsrp.set_ylabel("Erro AoA 3D (graus)")
ax_rsrp.set_title("(b) RSRP vs Erro AoA — bad regime")
ax_rsrp.legend(fontsize=9)

# ── (c) Distância vs erro AoA ─────────────────────────────────────────────────
for t in BAD_CATS:
    mask = ptype == t
    if not mask.any(): continue
    valid = mask & np.isfinite(err_3d)
    ax_dist.scatter(dist_h[valid], err_3d[valid], color=COLORS[t], s=18,
                    alpha=0.7, label=f"{t} (n={valid.sum()})")
    if valid.sum() > 2:
        sl, ic, *_ = stats.linregress(dist_h[valid], err_3d[valid])
        xr = np.array([dist_h[valid].min(), dist_h[valid].max()])
        ax_dist.plot(xr, sl*xr + ic, color=COLORS[t], lw=1.8, ls="--")

ax_dist.set_xlabel("Distância horizontal à gNB (m)"); ax_dist.set_ylabel("Erro AoA 3D (graus)")
ax_dist.set_title("(c) Distância vs Erro AoA — bad regime")
ax_dist.legend(fontsize=9)

# ── (d) n_paths vs erro AoA ───────────────────────────────────────────────────
for t in BAD_CATS:
    mask = ptype == t
    if not mask.any(): continue
    valid = mask & np.isfinite(err_3d)
    # jitter horizontal para legibilidade
    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, valid.sum())
    ax_np.scatter(n_paths_arr[valid].astype(float) + jitter, err_3d[valid],
                  color=COLORS[t], s=18, alpha=0.7, label=f"{t} (n={valid.sum()})")

ax_np.set_xlabel("Número de caminhos válidos"); ax_np.set_ylabel("Erro AoA 3D (graus)")
ax_np.set_title("(d) n_paths vs Erro AoA — bad regime")
ax_np.legend(fontsize=9)

# ── (e) Setas de erro de posição (bad regime) ─────────────────────────────────
ax_arr.scatter(pos_e, pos_n, color="lightgray", s=8, alpha=0.5)
ax_arr.scatter(*gnb[:2], marker="^", c="k", s=130, zorder=5)

for t in BAD_CATS:
    mask = ptype == t
    if not mask.any(): continue
    valid = mask & np.isfinite(east_hat) & np.isfinite(north_hat)
    for i in np.where(valid)[0]:
        ax_arr.annotate("",
            xy=(east_hat[i], north_hat[i]), xytext=(pos_e[i], pos_n[i]),
            arrowprops=dict(arrowstyle="->", color=COLORS[t], lw=0.8, alpha=0.7))
    # ponto de origem (posição verdadeira)
    ax_arr.scatter(pos_e[valid], pos_n[valid], color=COLORS[t], s=14,
                   alpha=0.9, zorder=3)

patches = [mpatches.Patch(color=COLORS[t], label=t) for t in BAD_CATS]
ax_arr.legend(handles=patches, fontsize=8)
ax_arr.set_title("(e) Erros de posição — setas true→estimado (bad regime)")
ax_arr.set_xlabel("Leste (m)"); ax_arr.set_ylabel("Norte (m)")

# ── (f) CDF do erro de posição por categoria ──────────────────────────────────
for t in CATS:
    mask = ptype == t
    if not mask.any(): continue
    valid = mask & np.isfinite(err_pos)
    if not valid.any(): continue
    errs  = np.sort(err_pos[valid])
    cdf   = np.arange(1, len(errs)+1) / len(errs)
    ax_cdf.plot(errs, cdf, color=COLORS[t], lw=2, label=f"{t} (n={valid.sum()})")

    med = float(np.nanmedian(err_pos[mask]))
    p95 = float(np.nanpercentile(err_pos[mask], 95))
    ax_cdf.axvline(med, color=COLORS[t], lw=0.9, ls=":")
    ax_cdf.axvline(p95, color=COLORS[t], lw=0.9, ls="--")

# Linhas de referência globais
ax_cdf.axhline(0.50, color="gray", lw=0.7, ls=":", alpha=0.7)
ax_cdf.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.7)
ax_cdf.text(ax_cdf.get_xlim()[0] if ax_cdf.get_xlim()[0] > 0 else 5,
            0.52, "mediana", fontsize=7.5, color="gray")
ax_cdf.text(ax_cdf.get_xlim()[0] if ax_cdf.get_xlim()[0] > 0 else 5,
            0.97, "p95", fontsize=7.5, color="gray")

ax_cdf.set_xlabel("Erro de posição (m)")
ax_cdf.set_ylabel("CDF")
ax_cdf.set_title(f"(f) CDF do erro de posição — estimador: EIRP={EIRP_DBM:.0f} dBm, n={N_PL}")
ax_cdf.set_ylim(0, 1); ax_cdf.legend(fontsize=9, loc="lower right")
ax_cdf.grid(True, alpha=0.3)

# ─── Anotação do estimador ────────────────────────────────────────────────────
ax_cdf.text(0.02, 0.97,
    f"d̂ = 10^((EIRP − RSRP)/(10·n))  |  EIRP={EIRP_DBM:.0f} dBm  |  n={N_PL}",
    transform=ax_cdf.transAxes, fontsize=8.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))

# ─── Salva ────────────────────────────────────────────────────────────────────
plt.savefig(args.out, dpi=160, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")
