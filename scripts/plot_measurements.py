"""
plot_measurements.py — Diagnóstico das medidas simuladas pelo Sionna RT.

Gera output/measurements_diag.png com 4 painéis:
  (a) Scatter dos UEs colorido por RSRP [dBm]
  (b) Scatter colorido pelo tipo de caminho dominante
  (c) Histograma de RSRP separado por tipo de caminho
  (d) Erro angular AoA (medido vs geométrico verdadeiro) vs distância à gNB

O painel (d) é o mais relevante: mostra onde o multipath enviesa o AoA
em relação ao valor geométrico, que é o que modelos analíticos não capturam.

Uso:
    python scripts/plot_measurements.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

NPZ_IN  = "output/measurements.npz"
PNG_OUT = "output/measurements_diag.png"

def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

# ─── Carrega dados ────────────────────────────────────────────────────────────
data     = np.load(NPZ_IN, allow_pickle=True)
rsrp     = data["rsrp_dbm"].astype(float)
aoa_az   = data["aoa_az_rad"].astype(float)
aoa_el   = data["aoa_el_rad"].astype(float)
ptype    = data["path_type"].astype(str)
n_paths  = data["n_paths"].astype(int)
pos_e    = data["pos_east"].astype(float)
pos_n    = data["pos_north"].astype(float)
gnb      = data["gnb_pos"].astype(float)
freq_hz  = float(data["freq_hz"]) if "freq_hz" in data else 3.5e9
n_ues    = len(rsrp)

print(f"[INFO] {n_ues} UEs carregados de {NPZ_IN}")
for t in ['LoS', 'reflected', 'refracted', 'diffracted', 'diffuse', 'none', 'unknown']:
    cnt = int((ptype == t).sum())
    if cnt > 0:
        print(f"[INFO]   {t:<12s}: {cnt} UEs ({100*cnt/n_ues:.1f}%)")

# ─── Geometria ────────────────────────────────────────────────────────────────
# Vetor da gNB ao UE (em ENU)
d_east  = pos_e - gnb[0]
d_north = pos_n - gnb[1]
d_up    = 1.5 - gnb[2]  # UEs @ 1.5 m, gNB @ cfg["h_gnb"] m → negativo

dist_3d = np.sqrt(d_east**2 + d_north**2 + d_up**2)
dist_2d = np.sqrt(d_east**2 + d_north**2)

# AoA geométrico verdadeiro na gNB (ângulos de partida da gNB → UE)
# phi: azimute do X-axis (East); theta: ângulo zenital do Z-axis (Up)
phi_true   = np.arctan2(d_north, d_east)                       # azimute verdadeiro
theta_true = np.arccos(np.clip(d_up / np.maximum(dist_3d, 1e-6), -1, 1))
el_true    = np.pi / 2 - theta_true                            # elevação verdadeira

# ─── Erro angular 3D ──────────────────────────────────────────────────────────
# Converte ângulos esféricos → vetores unitários (convenção Sionna/ENU)
# theta: zenital (0=up, π=down); phi: azimute (0=East, π/2=North)
def sph2vec(el_rad, az_rad):
    theta = np.pi / 2 - el_rad  # elevação → zenital
    return np.column_stack([
        np.sin(theta) * np.cos(az_rad),   # East component
        np.sin(theta) * np.sin(az_rad),   # North component
        np.cos(theta),                     # Up component
    ])

# Só UEs com caminhos válidos e ângulos finitos
valid = (ptype != 'none') & (ptype != 'unknown') & np.isfinite(aoa_az) & np.isfinite(aoa_el)

if valid.sum() > 0:
    meas_vecs = sph2vec(aoa_el[valid],  aoa_az[valid])
    true_vecs = sph2vec(el_true[valid], phi_true[valid])
    dot       = np.clip((meas_vecs * true_vecs).sum(axis=1), -1., 1.)
    ang_err   = np.degrees(np.arccos(dot))  # erro angular total [graus]
else:
    ang_err = np.array([])
    warn("Nenhum UE com caminhos válidos para calcular erro angular.")

# ─── Cores por tipo ───────────────────────────────────────────────────────────
TYPE_COLORS = {
    'LoS':        '#2ecc71',
    'reflected':  '#3498db',
    'refracted':  '#1abc9c',
    'diffracted': '#e67e22',
    'diffuse':    '#9b59b6',
    'none':       '#e74c3c',
    'unknown':    '#95a5a6',
}
# Ordem de renderização (LoS por cima)
TYPE_ORDER = ['none', 'unknown', 'diffuse', 'diffracted', 'refracted', 'reflected', 'LoS']

# ─── Figura principal ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    f"Medidas Sionna RT — Interlagos | {freq_hz/1e9:.1f} GHz | "
    f"{n_ues} UEs | P_tx = 23 dBm",
    fontsize=13, fontweight="bold"
)

# ── (a) Scatter RSRP ─────────────────────────────────────────────────────────
ax = axes[0, 0]
rsrp_fin = rsrp[np.isfinite(rsrp)]
if len(rsrp_fin) > 0:
    sc = ax.scatter(pos_e, pos_n, c=rsrp, cmap="jet", s=25, alpha=0.85,
                    vmin=np.nanpercentile(rsrp_fin, 5),
                    vmax=np.nanpercentile(rsrp_fin, 99))
    plt.colorbar(sc, ax=ax, label="RSRP (dBm)")
ax.plot(gnb[0], gnb[1], "w^", ms=12, markeredgecolor="k", label="gNB", zorder=5)
ax.set_title("(a) RSRP [dBm]")
ax.set_xlabel("Leste (m)"); ax.set_ylabel("Norte (m)")
ax.legend(fontsize=9); ax.set_aspect("equal"); ax.grid(alpha=0.25)

# ── (b) Scatter tipo de caminho ───────────────────────────────────────────────
ax = axes[0, 1]
for t in TYPE_ORDER:
    mask = ptype == t
    if mask.sum() == 0:
        continue
    ax.scatter(pos_e[mask], pos_n[mask], color=TYPE_COLORS[t],
               s=25, alpha=0.85, label=f"{t} ({mask.sum()})", zorder=3)
ax.plot(gnb[0], gnb[1], "w^", ms=12, markeredgecolor="k", zorder=5)
ax.set_title("(b) Tipo de caminho dominante")
ax.set_xlabel("Leste (m)"); ax.set_ylabel("Norte (m)")
ax.legend(fontsize=8, ncol=2); ax.set_aspect("equal"); ax.grid(alpha=0.25)

# ── (c) Histograma de RSRP por tipo ──────────────────────────────────────────
ax = axes[1, 0]
rsrp_types = [t for t in TYPE_ORDER if t not in ('none', 'unknown')]
if len(rsrp_fin) > 5:
    bins = np.linspace(np.nanpercentile(rsrp_fin, 1),
                       np.nanpercentile(rsrp_fin, 99), 35)
    for t in rsrp_types:
        mask = (ptype == t) & np.isfinite(rsrp)
        if mask.sum() < 2:
            continue
        ax.hist(rsrp[mask], bins=bins, alpha=0.55, color=TYPE_COLORS[t],
                label=f"{t} (n={mask.sum()}, med={np.median(rsrp[mask]):.0f} dBm)",
                density=True, histtype="stepfilled", edgecolor="none")
ax.set_title("(c) Histograma de RSRP por tipo de caminho")
ax.set_xlabel("RSRP (dBm)"); ax.set_ylabel("Densidade")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── (d) Erro angular AoA vs distância horizontal ─────────────────────────────
ax = axes[1, 1]
if valid.sum() > 0:
    ptype_valid = ptype[valid]
    dist2d_valid = dist_2d[valid]
    for t in TYPE_ORDER:
        if t in ('none', 'unknown'):
            continue
        mask_t = ptype_valid == t
        if mask_t.sum() == 0:
            continue
        ax.scatter(dist2d_valid[mask_t], ang_err[mask_t],
                   color=TYPE_COLORS[t], s=18, alpha=0.7,
                   label=f"{t} (med={np.median(ang_err[mask_t]):.1f}°)", zorder=3)

    # Linha de tendência global
    if len(ang_err) > 5:
        from numpy.polynomial.polynomial import polyfit as polyfit_np
        try:
            coef = polyfit_np(dist2d_valid, ang_err, 1)
            x_l  = np.linspace(dist2d_valid.min(), dist2d_valid.max(), 200)
            ax.plot(x_l, coef[0] + coef[1]*x_l, "k--", lw=1.5,
                    alpha=0.7, label="tendência global", zorder=4)
        except Exception:
            pass

    ax.set_ylim(bottom=0)
ax.set_title("(d) Erro angular AoA (medido vs geométrico) vs Distância")
ax.set_xlabel("Distância horizontal à gNB (m)")
ax.set_ylabel("Erro angular 3D (graus)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Diagnóstico salvo: {PNG_OUT}")

# ─── Estatísticas resumidas ───────────────────────────────────────────────────
print()
print("=" * 50)
print("RESUMO")
print("=" * 50)
print(f"  UEs simulados : {n_ues}")
print(f"  UEs com paths : {int(valid.sum())}")
if len(rsrp_fin) > 0:
    print(f"  RSRP          : [{rsrp_fin.min():.1f}, {rsrp_fin.max():.1f}] dBm "
          f"| mediana={np.median(rsrp_fin):.1f} dBm")
if len(ang_err) > 0:
    print(f"  Erro AoA      : [{ang_err.min():.2f}°, {ang_err.max():.2f}°] "
          f"| mediana={np.median(ang_err):.2f}°")
    ptype_v = ptype[valid]
    for t in ['LoS', 'reflected', 'refracted', 'diffracted', 'diffuse']:
        tmask = ptype_v == t
        if tmask.sum() > 0:
            print(f"  Erro AoA {t:<10s}: n={tmask.sum():3d} | "
                  f"mediana={np.median(ang_err[tmask]):.2f}° | "
                  f"max={ang_err[tmask].max():.2f}°")
print("=" * 50)
print()
ok("=== plot_measurements.py concluído ===")
