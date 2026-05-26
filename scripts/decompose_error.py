"""
decompose_error.py — Decomposição do erro de posição em componentes radial e tangencial.

Para cada UE:
  err_radial    = |d_hat - d_true|
  err_tangencial = d_true · sin(min(|err_aoa|, π - |err_aoa|))

A correção min(|err|, π - |err|) é necessária porque o seno é simétrico em π/2:
sin(|err|) = sin(π - |err|) em qualquer caso, mas o ângulo efetivo entre o
vetor estimado e o eixo gNB→UE é sempre o menor dos dois. Para err_aoa → 180°,
o erro dominante passa a ser radial (UE projetado no lado oposto da gNB), e
o componente tangencial volta a zero — o que min(·,π-·) torna explícito.

Usa o estimador global calibrado (n=1.362, A=−33.0 dBm) para d_hat.

Saídas:
  output/error_decomposition.png  — 4 painéis
  output/error_decomposition.md   — tabela markdown
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz",    default="output/measurements_v2.npz")
parser.add_argument("--params", default="output/pathloss_calibrated.json")
parser.add_argument("--out",    default="output/error_decomposition.png")
parser.add_argument("--md",     default="output/error_decomposition.md")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

CATS     = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS   = {"LoS": "#2ecc71", "diffracted": "#e67e22",
            "reflected": "#3498db", "transmitted": "#9b59b6"}
BAD_CATS = ["reflected", "diffracted"]

# ─── Carrega dados ─────────────────────────────────────────────────────────────
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]; ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)

with open(args.params) as f:
    params = json.load(f)

n_g = params["global"]["n"]; A_g = params["global"]["A"]
info(f"Estimador global: n={n_g:.3f}, A={A_g:.2f} dBm")

# ─── Geometria verdadeira ─────────────────────────────────────────────────────
delta_e = pos_e - gnb[0]
delta_n = pos_n - gnb[1]
delta_u = np.full(n_ues, 1.5) - gnb[2]
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)   # distância real

# ─── Erro AoA 3D ──────────────────────────────────────────────────────────────
phi_true   = np.arctan2(delta_n, delta_e)
theta_true = np.arccos(np.clip(delta_u / np.maximum(dist_3d, 1e-6), -1, 1))
el_true    = np.pi / 2 - theta_true

def sph2vec(el_rad, az_rad):
    theta = np.pi / 2 - el_rad
    return np.column_stack([np.sin(theta)*np.cos(az_rad),
                            np.sin(theta)*np.sin(az_rad),
                            np.cos(theta)])

valid_mask = (ptype != 'none') & np.isfinite(aoa_az) & np.isfinite(aoa_el)
err_aoa_rad = np.full(n_ues, np.nan)
if valid_mask.sum() > 0:
    mv  = sph2vec(aoa_el[valid_mask], aoa_az[valid_mask])
    tv  = sph2vec(el_true[valid_mask], phi_true[valid_mask])
    dot = np.clip((mv * tv).sum(axis=1), -1., 1.)
    err_aoa_rad[valid_mask] = np.arccos(dot)     # em radianos, [0, π]

# ─── d_hat (modo global calibrado) ────────────────────────────────────────────
d_hat = 10 ** ((A_g - rsrp) / (10 * n_g))

# ─── Decomposição do erro ─────────────────────────────────────────────────────
err_radial = np.abs(d_hat - dist_3d)

# Componente tangencial: d_true · sin(min(|err_aoa|, π − |err_aoa|))
# Equivalente a d_true · sin(|err_aoa|) numericamente (sin é simétrico em π/2),
# mas a forma min(·, π-·) deixa explícito que para err→180°, o ângulo efetivo
# entre o vetor estimado e o eixo gNB→UE volta a diminuir, e o erro dominante
# passa a ser radial (projeção no lado oposto). Quando err_aoa=π, o sin=0 e
# err_tangencial=0 mesmo que a posição estimada esteja completamente errada —
# pois nesse caso TUDO é radial (estimou exatamente 180° errado).
err_tang = dist_3d * np.sin(np.minimum(np.abs(err_aoa_rad),
                                        np.pi - np.abs(err_aoa_rad)))

err_total_pred = np.sqrt(err_radial**2 + err_tang**2)

# Erro de posição 2D real (para verificação)
theta_aoa = np.pi / 2 - aoa_el
east_hat  = gnb[0] + d_hat * np.sin(theta_aoa) * np.cos(aoa_az)
north_hat = gnb[1] + d_hat * np.sin(theta_aoa) * np.sin(aoa_az)
err_real  = np.sqrt((east_hat - pos_e)**2 + (north_hat - pos_n)**2)

# ─── Log ──────────────────────────────────────────────────────────────────────
info("Decomposição — mediana (m):")
info(f"  {'Cat':<14} {'radial':>8} {'tang':>8} {'pred':>8} {'real 2D':>8} {'dominante':>10}")
rows_md = []
for t in CATS:
    mask  = ptype == t
    valid = mask & np.isfinite(err_radial) & np.isfinite(err_tang)
    if not valid.any(): continue
    med_r = float(np.nanmedian(err_radial[valid]))
    med_t = float(np.nanmedian(err_tang[valid]))
    med_p = float(np.nanmedian(err_total_pred[valid]))
    med_e = float(np.nanmedian(err_real[valid & np.isfinite(err_real)]))
    dom   = "radial" if med_r > med_t else "tangencial"
    info(f"  {t:<14}  {med_r:>8.1f}  {med_t:>8.1f}  {med_p:>8.1f}  {med_e:>8.1f}  {dom:>10}")
    rows_md.append((t, mask.sum(), med_r, med_t, dom))

# ─── Gráfico 2×2 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
freq_ghz  = float(d["freq_hz"]) / 1e9
fig.suptitle(
    f"Decomposição do Erro de Posição — {freq_ghz:.1f} GHz | Global n={n_g:.2f}",
    fontsize=13
)

# (a) CDF err_radial
ax = axes[0, 0]
for t in CATS:
    mask  = ptype == t
    valid = mask & np.isfinite(err_radial)
    if not valid.any(): continue
    vals = np.sort(err_radial[valid])
    ax.plot(vals, np.arange(1, len(vals)+1)/len(vals),
            color=COLORS[t], lw=2, label=f"{t} (n={valid.sum()})")
ax.set_xlabel("Erro radial |d̂ − d_true| (m)"); ax.set_ylabel("CDF")
ax.set_title("(a) CDF erro radial por categoria")
ax.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# (b) CDF err_tangencial
ax = axes[0, 1]
for t in CATS:
    mask  = ptype == t
    valid = mask & np.isfinite(err_tang)
    if not valid.any(): continue
    vals = np.sort(err_tang[valid])
    ax.plot(vals, np.arange(1, len(vals)+1)/len(vals),
            color=COLORS[t], lw=2, label=f"{t} (n={valid.sum()})")
ax.set_xlabel("Erro tangencial d_true·sin(min(ε,π−ε)) (m)"); ax.set_ylabel("CDF")
ax.set_title("(b) CDF erro tangencial por categoria")
ax.axhline(0.5, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

# (c) Scatter (err_radial, err_tangencial) — bad regime
ax = axes[1, 0]
lim = 0
for t in BAD_CATS:
    mask  = ptype == t
    valid = mask & np.isfinite(err_radial) & np.isfinite(err_tang)
    if not valid.any(): continue
    ax.scatter(err_radial[valid], err_tang[valid], color=COLORS[t],
               s=22, alpha=0.75, label=f"{t} (n={valid.sum()})", zorder=3)
    lim = max(lim, err_radial[valid].max(), err_tang[valid].max())

# Bissetriz (radial = tangencial)
lim = min(lim * 1.1, 2000)
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="radial = tang.")
ax.set_xlabel("Erro radial (m)"); ax.set_ylabel("Erro tangencial (m)")
ax.set_title("(c) Scatter radial × tangencial — bad regime")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, "Acima da bissetriz → dominante tangencial",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.85))

# (d) Barras: mediana radial vs tangencial por categoria
ax = axes[1, 1]
cats_present = [r[0] for r in rows_md]
x  = np.arange(len(cats_present))
w  = 0.35
med_rs = [r[2] for r in rows_md]
med_ts = [r[3] for r in rows_md]
cols   = [COLORS[t] for t in cats_present]

bars_r = ax.bar(x - w/2, med_rs, w, label="radial med",
                color=cols, alpha=0.9, edgecolor="k", linewidth=0.6)
bars_t = ax.bar(x + w/2, med_ts, w, label="tangencial med",
                color=cols, alpha=0.45, edgecolor="k", linewidth=0.6, hatch="//")
for i, (r2, dom) in enumerate([(r[2], r[4]) for r in rows_md]):
    ax.text(i, max(med_rs[i], med_ts[i]) + 3, dom,
            ha="center", fontsize=8, fontweight="bold",
            color="darkred" if dom == "tangencial" else "navy")

ax.set_xticks(x); ax.set_xticklabels(cats_present, fontsize=9)
ax.set_ylabel("Mediana do erro (m)"); ax.set_title("(d) Radial vs Tangencial por categoria")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")

# ─── Markdown ─────────────────────────────────────────────────────────────────
freq_ghz = float(d["freq_hz"]) / 1e9
md  = f"# Decomposição do Erro de Posição\n\n"
md += f"**Frequência:** {freq_ghz:.1f} GHz | **Estimador global:** n={n_g:.3f}, A={A_g:.2f} dBm\n\n"
md += f"**err_tangencial** = d_true · sin(min(|ε|, π − |ε|)) — ângulo efetivo entre "
md +=  f"vetor estimado e eixo gNB→UE\n\n"
md += "| Categoria | n | Radial med (m) | Tang. med (m) | Dominante |\n"
md += "|:---|---:|---:|---:|:---:|\n"
for (t, n_cat, med_r, med_t, dom) in rows_md:
    md += f"| {t} | {n_cat} | {med_r:.1f} | {med_t:.1f} | **{dom}** |\n"
md += "\n## Verificação\n\n"
md += "err_total_predito = √(radial² + tang²) deve aproximar o erro 2D real.\n"
md += "Discrepância esperada: projeção 3D→2D e linearização d_hat ≈ d_true.\n"

with open(args.md, "w") as f:
    f.write(md)
ok(f"Markdown salvo: {args.md}")
