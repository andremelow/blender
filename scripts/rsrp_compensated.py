"""
rsrp_compensated.py — Compensação do RSRP pelo padrão de elemento TR38901.

O RSRP medido inclui o ganho direcional do UPA 8×8 da gNB, que varia até
~30 dB com o ângulo — dominando a variação de path loss (~17 dB no grid).
Esta variação angular explica o R²≈0.05 na calibração direta.

Compensação:
    RSRP_comp = RSRP_meas − G_elemento(θ_t, φ_t)   [dBm]

onde G_elemento é o ganho TR38901 do elemento isolado calculado analiticamente
nos ângulos de partida da gNB (φ_t = aoa_az_rad, θ_zenital = π/2 − aoa_el_rad).

Saída: output/rsrp_compensation.png
       Tabela R² comparativa no stdout
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz",      default="output/measurements_v2.npz")
parser.add_argument("--params",   default="output/pathloss_calibrated.json")
parser.add_argument("--out-png",  default="output/rsrp_compensation.png")
parser.add_argument("--out-json", default="output/pathloss_compensated.json")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

CATS   = ["LoS", "transmitted", "reflected", "diffracted"]
COLORS = {"LoS": "#2ecc71", "diffracted": "#e67e22",
          "reflected": "#3498db", "transmitted": "#9b59b6"}

# ─── Carrega dados ─────────────────────────────────────────────────────────────
info(f"Carregando: {args.npz}")
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]; ptype  = d["path_type"]
aoa_az = d["aoa_az_rad"]; aoa_el = d["aoa_el_rad"]
gnb    = d["gnb_pos"]
n_ues  = len(rsrp)
freq_ghz = float(d["freq_hz"]) / 1e9

with open(args.params) as f:
    params_orig = json.load(f)

# ─── Ganho do elemento TR38901 (analítico, numpy) ─────────────────────────────
# Fórmula conforme 3GPP TR38901 Tabela 7.3-1 / código-fonte Sionna rt.antenna_pattern.
# Parâmetros padrão UPA 8×8: θ_3dB = φ_3dB = 65°, SLA_v = A_m = 30 dB, G_max = 8 dBi.
def ganho_elemento_tr38901_dbi(theta_zenital, phi_az):
    """
    Parâmetros:
        theta_zenital : ângulo zenital (rad), 0=cima, π/2=horizontal, π=baixo
        phi_az        : azimute (rad), 0=boresight (Leste), ±π = atrás

    Retorna ganho em dBi (array numpy).
    """
    THETA_3DB = np.radians(65.)
    PHI_3DB   = np.radians(65.)
    A_M       = 30.0   # atenuação máxima (dB)
    SLA_V     = 30.0   # side-lobe level vertical (dB)
    G_MAX     = 8.0    # ganho máximo do elemento (dBi)

    # Ajusta φ para [−π, π]  (igual ao código Sionna: phi += π, wrap, phi -= π)
    phi_norm = (phi_az + np.pi) % (2 * np.pi) - np.pi

    # Atenuação vertical: 0 dB no horizontal (θ = π/2)
    theta_dev = theta_zenital - np.pi / 2
    A_EV = -np.minimum(12. * (theta_dev / THETA_3DB) ** 2, SLA_V)

    # Atenuação horizontal: 0 dB no boresight (φ = 0)
    A_EH = -np.minimum(12. * (phi_norm / PHI_3DB) ** 2, A_M)

    # Ganho combinado
    G = G_MAX - np.minimum(-(A_EH + A_EV), A_M)
    return G   # dBi

# ─── Computa ganho para cada UE ───────────────────────────────────────────────
# theta_t é o ângulo zenital de partida da gNB: π/2 − aoa_el_rad
# phi_t  é o azimute de partida da gNB:          aoa_az_rad
# (gNB orientation=[0,0,0] → frame local = frame global; sem rotação necessária)
theta_t = np.pi / 2 - aoa_el    # zenital: π/2 no horizontal
phi_t   = aoa_az                 # azimute: 0 = Leste = boresight

valid_aoa = np.isfinite(aoa_az) & np.isfinite(aoa_el)
G_elem_dbi = np.full(n_ues, np.nan)
G_elem_dbi[valid_aoa] = ganho_elemento_tr38901_dbi(theta_t[valid_aoa], phi_t[valid_aoa])

info(f"Ganho de elemento: mín={np.nanmin(G_elem_dbi):.1f} dBi, "
     f"máx={np.nanmax(G_elem_dbi):.1f} dBi, "
     f"med={np.nanmedian(G_elem_dbi):.1f} dBi")

rsrp_comp = rsrp - G_elem_dbi

# ─── Distância 3D real ────────────────────────────────────────────────────────
delta_e = pos_e - gnb[0]
delta_n = pos_n - gnb[1]
delta_u = np.full(n_ues, 1.5) - gnb[2]
dist_3d = np.sqrt(delta_e**2 + delta_n**2 + delta_u**2)

# ─── Regressão log10(d) vs RSRP_comp ─────────────────────────────────────────
def fit_pathloss(rsrp_vals, dist_vals):
    mask = np.isfinite(rsrp_vals) & (dist_vals > 0)
    if mask.sum() < 3:
        return None
    x = np.log10(dist_vals[mask])
    y = rsrp_vals[mask]
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    n_pl = -slope / 10.
    A    = intercept
    y_hat  = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1. - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": round(n_pl, 3), "A": round(float(A), 2),
            "r2": round(r2, 4), "n_samples": int(mask.sum())}

params_comp = {}
for t in CATS:
    mask = ptype == t
    res  = fit_pathloss(rsrp_comp[mask], dist_3d[mask])
    if res:
        params_comp[t] = res

res_global = fit_pathloss(rsrp_comp, dist_3d)
if res_global:
    params_comp["global"] = res_global

# ─── Tabela comparativa R² ────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"{'Categoria':<14} {'n':>6}  {'n_comp':>7}  {'A_comp':>8}  {'R²_orig':>8}  {'R²_comp':>8}")
print("-" * 72)
for t in CATS + ["global"]:
    if t not in params_comp: continue
    p_c = params_comp[t]
    p_o = params_orig.get(t, {})
    r2_o = p_o.get("r2", float("nan"))
    n_o  = p_o.get("n", float("nan"))
    delta_r2 = p_c["r2"] - r2_o
    flag = " ←" if t == "LoS" or t == "global" else ""
    print(f"  {t:<12}  {n_o:>6.3f}  {p_c['n']:>7.3f}  {p_c['A']:>8.2f}  "
          f"{r2_o:>8.4f}  {p_c['r2']:>8.4f}  (Δ{delta_r2:+.4f}){flag}")
print("=" * 72)

# ─── Salva parâmetros compensados ─────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
with open(args.out_json, "w") as f:
    json.dump(params_comp, f, indent=2)
ok(f"Parâmetros compensados salvos: {args.out_json}")

# ─── Plot: 3 painéis ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Compensação RSRP pelo Padrão de Elemento TR38901 — {freq_ghz:.1f} GHz",
    fontsize=13
)

# (a) RSRP original vs log10(d)
ax = axes[0]
ax.set_title("(a) RSRP original vs distância")
for t in CATS:
    mask  = (ptype == t) & np.isfinite(rsrp) & (dist_3d > 0)
    if not mask.any(): continue
    x_vals = np.log10(dist_3d[mask])
    ax.scatter(x_vals, rsrp[mask], color=COLORS[t], s=8, alpha=0.5,
               label=f"{t} (n={mask.sum()})")
    if t in params_orig:
        p = params_orig[t]
        xr = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(xr, p["A"] - 10*p["n"]*xr, "--", color=COLORS[t], lw=1.8,
                label=f"n={p['n']:.2f} R²={p['r2']:.3f}")
ax.set_xlabel("log₁₀(d_true) (m)")
ax.set_ylabel("RSRP (dBm)")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# (b) RSRP compensado vs log10(d)
ax = axes[1]
ax.set_title("(b) RSRP compensado vs distância\n(RSRP − G_elem)")
for t in CATS:
    mask  = (ptype == t) & np.isfinite(rsrp_comp) & (dist_3d > 0)
    if not mask.any(): continue
    x_vals = np.log10(dist_3d[mask])
    ax.scatter(x_vals, rsrp_comp[mask], color=COLORS[t], s=8, alpha=0.5,
               label=f"{t} (n={mask.sum()})")
    if t in params_comp:
        p = params_comp[t]
        xr = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(xr, p["A"] - 10*p["n"]*xr, "--", color=COLORS[t], lw=1.8,
                label=f"n={p['n']:.2f} R²={p['r2']:.3f}")
ax.set_xlabel("log₁₀(d_true) (m)")
ax.set_ylabel("RSRP − G_elem (dBm)")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# (c) Histograma de G_elem(θ, φ)
ax = axes[2]
ax.set_title("(c) Distribuição do ganho de elemento G(θ,φ)")

# Histograma por categoria
bins = np.linspace(-32, 10, 50)
for t in CATS:
    mask = (ptype == t) & np.isfinite(G_elem_dbi)
    if not mask.any(): continue
    ax.hist(G_elem_dbi[mask], bins=bins, color=COLORS[t], alpha=0.5,
            label=f"{t} (n={mask.sum()})", density=True)

ax.axvline(8.0, color="black", ls="--", lw=1.2, label="G_max=8 dBi")
ax.axvline(-22.0, color="gray", ls=":", lw=1.0, label="G_max−A_m=−22 dBi")
ax.set_xlabel("G_elemento (dBi)")
ax.set_ylabel("Densidade")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Adiciona nota de resultado R²
los_r2_orig = params_orig.get("LoS", {}).get("r2", float("nan"))
los_r2_comp = params_comp.get("LoS", {}).get("r2", float("nan"))
msg = (f"LoS R² original: {los_r2_orig:.4f}\n"
       f"LoS R² compensado: {los_r2_comp:.4f}\n"
       f"Δ = {los_r2_comp - los_r2_orig:+.4f}")
ax.text(0.03, 0.97, msg, transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

plt.tight_layout()
plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out_png}")

# ─── Diagnóstico: amplitude do efeito angular ─────────────────────────────────
info("Variação angular do ganho de elemento:")
for t in CATS:
    mask = (ptype == t) & np.isfinite(G_elem_dbi)
    if not mask.any(): continue
    info(f"  {t:<14}: G min={np.nanmin(G_elem_dbi[mask]):.1f} dBi "
         f"max={np.nanmax(G_elem_dbi[mask]):.1f} dBi "
         f"range={np.nanmax(G_elem_dbi[mask])-np.nanmin(G_elem_dbi[mask]):.1f} dB")
