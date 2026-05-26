"""
calibrate_pathloss.py — Calibração honesta do modelo de path loss por categoria.

Regressão linear:
    RSRP_dBm = A - 10·n·log10(d_true)

onde A absorve EIRP e a perda de penetração (se houver).
Usa numpy.polyfit(log10(d_true), RSRP, 1), grau 1.

Saídas:
  output/pathloss_calibrated.json  — parâmetros (n, A, R², n_amostras)
  output/pathloss_calibration.png  — scatter RSRP vs log10(d) + retas
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default="output/measurements_v2.npz")
parser.add_argument("--out-json", default="output/pathloss_calibrated.json")
parser.add_argument("--out-png",  default="output/pathloss_calibration.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

CATS   = ["LoS", "transmitted", "reflected", "diffracted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# ─── Carrega dados ─────────────────────────────────────────────────────────────
info(f"Carregando: {args.npz}")
d     = np.load(args.npz, allow_pickle=True)
pos_e = d["pos_east"]; pos_n = d["pos_north"]
rsrp  = d["rsrp_dbm"]; ptype = d["path_type"]
gnb   = d["gnb_pos"]
n_ues = len(rsrp)

# ─── Distância 3D real (UE z=1.5 m, gNB z conforme gnb_pos) ──────────────────
delta_e = pos_e - gnb[0]
delta_n = pos_n - gnb[1]
delta_u = np.full(n_ues, 1.5) - gnb[2]
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)

# ─── Regressão por categoria ──────────────────────────────────────────────────
def fit_pathloss(rsrp_vals, dist_vals):
    """
    Ajusta RSRP = A - 10·n·log10(d) via polyfit grau 1 em log10(d).
    Retorna dict {n, A, r2, n_samples}.
    """
    mask = np.isfinite(rsrp_vals) & (dist_vals > 0)
    if mask.sum() < 3:
        return None
    x = np.log10(dist_vals[mask])
    y = rsrp_vals[mask]
    coeffs = np.polyfit(x, y, 1)          # coeffs[0]=slope, coeffs[1]=intercept
    slope, intercept = coeffs
    n_pl = -slope / 10.0                   # slope = -10·n
    A    = intercept                       # RSRP @ d=1 m (= EIRP - PL0)
    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": round(n_pl, 3), "A": round(float(A), 2),
            "r2": round(r2, 4), "n_samples": int(mask.sum())}

params = {}
for t in CATS:
    mask = ptype == t
    res  = fit_pathloss(rsrp[mask], dist_3d[mask])
    if res:
        params[t] = res
        info(f"  {t:<12}: n={res['n']:.3f}  A={res['A']:+.1f} dBm  R²={res['r2']:.3f}  n={res['n_samples']}")
    else:
        info(f"  {t:<12}: amostras insuficientes")

# Ajuste global (todos os UEs com RSRP finito)
res_global = fit_pathloss(rsrp, dist_3d)
if res_global:
    params["global"] = res_global
    info(f"  {'global':<12}: n={res_global['n']:.3f}  A={res_global['A']:+.1f} dBm  "
         f"R²={res_global['r2']:.3f}  n={res_global['n_samples']}")

# ─── Salva JSON ───────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump(params, f, indent=2)
ok(f"Params salvos: {args.out_json}")

# ─── Tabela de calibração ─────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Categoria':<14} {'n calib':>9} {'A (dBm)':>9} {'R²':>7} {'n amostras':>11}")
print("-" * 65)
for t in CATS + ["global"]:
    if t not in params: continue
    p = params[t]
    print(f"  {t:<12}  {p['n']:>9.3f}  {p['A']:>9.2f}  {p['r2']:>7.4f}  {p['n_samples']:>11}")
print("=" * 65)

# ─── Plot: scatter RSRP vs log10(d) com retas ajustadas ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
axes = axes.flatten()

freq_ghz = float(d["freq_hz"]) / 1e9
fig.suptitle(
    f"Calibração do Path Loss por Categoria — {freq_ghz:.1f} GHz | {n_ues} UEs",
    fontsize=13
)

for idx, t in enumerate(CATS):
    ax   = axes[idx]
    mask = ptype == t
    if not mask.any():
        ax.set_title(f"{t} — sem dados"); continue

    x_all = np.log10(dist_3d[mask])
    y_all = rsrp[mask]
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    ax.scatter(x_all[valid], y_all[valid], color=COLORS[t],
               s=10, alpha=0.5, label=f"dados (n={valid.sum()})")

    if t in params:
        p  = params[t]
        xr = np.linspace(x_all[valid].min(), x_all[valid].max(), 200)
        # reta: RSRP = A - 10·n·log10(d) = A + slope·x
        slope = -10 * p["n"]
        yr = p["A"] + slope * xr
        ax.plot(xr, yr, "k-", lw=2,
                label=f"ajuste: n={p['n']:.2f}  A={p['A']:+.1f} dBm  R²={p['r2']:.2f}")

        # Referência: n=2.7 / A=-7 (estimador anterior)
        yr_old = -7.0 - 27.0 * xr
        ax.plot(xr, yr_old, "r--", lw=1.2, alpha=0.6, label="n=2.7 A=−7 (anterior)")

    # Eixo x como distância real no topo
    ax.set_xlabel("log₁₀(d_true) — distância 3D (m)")
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title(f"({chr(97+idx)}) {t}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Marcas de distância no eixo x superior
    ax2 = ax.twiny()
    ticks_d = [50, 100, 200, 400]
    ticks_x = [np.log10(dd) for dd in ticks_d if dd <= 10**x_all[valid].max()]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks_x)
    ax2.set_xticklabels([f"{dd} m" for dd in ticks_d[:len(ticks_x)]], fontsize=7)

plt.tight_layout()
plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out_png}")
