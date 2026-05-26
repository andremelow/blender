"""
reestimate_positions.py — Re-estimativa de posição com path loss calibrado.

Dois modos:
  oracle : usa (n, A) calibrados da categoria real do UE.
           Limite superior — o que seria possível sabendo o regime.
  global : usa (n, A) ajustados em todos os UEs juntos.
           Limite inferior realista — o que a gNB faz sem saber o regime.

Comparação com estimador anterior (n=2.7, A=−7 dBm) no mesmo gráfico.

Saída: output/cdf_pos_error_calibrated.png
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
parser.add_argument("--out",    default="output/cdf_pos_error_calibrated.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

CATS   = ["LoS", "diffracted", "reflected", "transmitted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# ─── Carrega dados e parâmetros ───────────────────────────────────────────────
info(f"Carregando: {args.npz}")
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]; ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)

with open(args.params) as f:
    params = json.load(f)
info(f"Parâmetros calibrados carregados de: {args.params}")

# ─── Função de estimativa de posição ─────────────────────────────────────────
def estimar_posicao(rsrp_vals, aoa_az_vals, aoa_el_vals, n_pl, A_dBm):
    """
    Estima posição do UE a partir de RSRP e AoA.
    d_hat = 10^((A - RSRP) / (10·n))
    Direção: ângulos phi_t, theta_t (ângulos de partida da gNB = AoA por reciprocidade).
    """
    d_hat     = 10 ** ((A_dBm - rsrp_vals) / (10 * n_pl))
    theta_aoa = np.pi / 2 - aoa_el_vals          # ângulo zenital
    east_hat  = gnb[0] + d_hat * np.sin(theta_aoa) * np.cos(aoa_az_vals)
    north_hat = gnb[1] + d_hat * np.sin(theta_aoa) * np.sin(aoa_az_vals)
    return east_hat, north_hat

def erro_pos(east_hat, north_hat):
    return np.sqrt((east_hat - pos_e)**2 + (north_hat - pos_n)**2)

# ─── Modo anterior (n=2.7, A=−7 dBm) ─────────────────────────────────────────
N_OLD, A_OLD = 2.7, -7.0
eh_old, nh_old = estimar_posicao(rsrp, aoa_az, aoa_el, N_OLD, A_OLD)
err_old = erro_pos(eh_old, nh_old)

# ─── Modo global (parâmetros calibrados únicos) ───────────────────────────────
n_g = params["global"]["n"]; A_g = params["global"]["A"]
info(f"Global calibrado: n={n_g:.3f}  A={A_g:.2f} dBm")
eh_glob, nh_glob = estimar_posicao(rsrp, aoa_az, aoa_el, n_g, A_g)
err_glob = erro_pos(eh_glob, nh_glob)

# ─── Modo oracle (parâmetro da categoria real de cada UE) ─────────────────────
eh_orac = np.full(n_ues, np.nan)
nh_orac = np.full(n_ues, np.nan)
for t in CATS:
    mask = ptype == t
    if not mask.any() or t not in params:
        continue
    n_t, A_t = params[t]["n"], params[t]["A"]
    # Nota: para reflected, n<0 — o estimador inverte a relação d×RSRP.
    # O modo oracle ainda usa esses parâmetros para mostrar o limite real
    # (que será PIOR que global para categorias com R² ≈ 0).
    eh_orac[mask], nh_orac[mask] = estimar_posicao(
        rsrp[mask], aoa_az[mask], aoa_el[mask], n_t, A_t)
err_orac = erro_pos(eh_orac, nh_orac)

# ─── Resumo estatístico ───────────────────────────────────────────────────────
info("Erro de posição — mediana / p95 (m):")
info(f"  {'Modo':<20} {'med':>8} {'p95':>8}")
for nome, err in [("anterior n=2.7", err_old),
                  ("global calibrado", err_glob),
                  ("oracle", err_orac)]:
    valid = np.isfinite(err)
    med = float(np.nanmedian(err[valid]))
    p95 = float(np.nanpercentile(err[valid], 95))
    info(f"  {nome:<20}  {med:>8.0f}  {p95:>8.0f}")

# ─── Plot: CDF do erro de posição ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
freq_ghz  = float(d["freq_hz"]) / 1e9

fig.suptitle(
    f"CDF Erro de Posição — Modos Oracle vs Global vs Anterior | {freq_ghz:.1f} GHz",
    fontsize=13
)

# Painel esquerdo: por categoria, 3 modos cada
ax = axes[0]
ax.set_title("Por categoria (—— oracle  ·· global  -- anterior)")
for t in CATS:
    mask = ptype == t
    if not mask.any(): continue
    col = COLORS[t]
    for (err_arr, ls, lw, label_sfx) in [
        (err_orac, "-",  2.2, "oracle"),
        (err_glob, ":",  1.8, "global"),
        (err_old,  "--", 1.2, "ant."),
    ]:
        vals  = np.sort(err_arr[mask & np.isfinite(err_arr)])
        if len(vals) == 0: continue
        cdf   = np.arange(1, len(vals)+1) / len(vals)
        lbl   = f"{t} {label_sfx}" if ls == "-" else None   # legenda só no oracle
        ax.plot(vals, cdf, color=col, ls=ls, lw=lw, label=lbl, alpha=0.85)

ax.axhline(0.50, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax.set_xlabel("Erro de posição (m)"); ax.set_ylabel("CDF")
ax.set_ylim(0, 1); ax.set_xlim(0, 2000)
ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

# Adiciona nota sobre oracle para categorias com R²≈0
ax.text(0.98, 0.05,
    "Nota: oracle piora reflected/diffracted\n"
    "pois n<0 ou R²≈0 na calibração.",
    transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

# Painel direito: agregado (todos UEs juntos), 3 modos
ax2 = axes[1]
ax2.set_title("Agregado — todos os UEs")
modos_agg = [
    (err_orac, "#e74c3c", "-",  2.5, f"oracle (n por categoria)",    0),
    (err_glob, "#2980b9", "--", 2.0, f"global (n={n_g:.2f}, A={A_g:.1f} dBm)", 1),
    (err_old,  "#7f8c8d", ":",  1.5, f"anterior (n={N_OLD}, A={A_OLD:.0f} dBm)", 2),
]
for (err_arr, col, ls, lw, lbl, offset_idx) in modos_agg:
    valid = np.isfinite(err_arr)
    vals  = np.sort(err_arr[valid])
    cdf   = np.arange(1, len(vals)+1) / len(vals)
    ax2.plot(vals, cdf, color=col, ls=ls, lw=lw, label=lbl)

    med = float(np.nanmedian(err_arr[valid]))
    p95 = float(np.nanpercentile(err_arr[valid], 95))
    ax2.axvline(min(med, 2000), color=col, lw=0.9, ls=ls, alpha=0.5)
    ax2.text(min(med, 1970) + 5, 0.55 - 0.07 * offset_idx,
        f"p50={med:.0f} m", fontsize=7.5, color=col, ha="left")
    # p95 pode estar fora do eixo (>2000 m) — anota na borda direita com seta →
    x_p95 = min(p95, 1970)
    sfx   = "→" if p95 > 2000 else ""
    ax2.text(x_p95, 0.98 - 0.07 * offset_idx,
        f"p95={p95:.0f} m{sfx}", fontsize=7.5, color=col, ha="left", va="top")

ax2.axhline(0.50, color="gray", lw=0.7, ls=":", alpha=0.6)
ax2.axhline(0.95, color="gray", lw=0.7, ls="--", alpha=0.6)
ax2.set_xlabel("Erro de posição (m)"); ax2.set_ylabel("CDF")
ax2.set_ylim(0, 1); ax2.set_xlim(0, 2000)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.text(0.98, 0.05,
    "Eixo cortado em 2000 m.\nOutliers extremos (calib. ruim n=1.36):\nglobal p95=7570 m, oracle p95=2620 m.",
    transform=ax2.transAxes, fontsize=7.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.85))

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")
