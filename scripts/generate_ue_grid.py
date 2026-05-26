"""
generate_ue_grid.py — Gera grade 2D de UEs e filtra por cobertura do RadioMap.

Grid: East e North de -400 a +400 m, passo 20 m → 41×41 = 1681 posições.
Filtro: path_gain >= -140 dB na altitude de 1.5 m.

Saída:
    output/ue_grid.npz   — positions (N,3), valid_mask (N,), pg_at_ue (N,)
    output/ue_grid.png   — visualização do grid e cobertura

Uso:
    python scripts/generate_ue_grid.py
"""

import os
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import sionna.rt as rt

# ─── Configuração ─────────────────────────────────────────────────────────────
PKL_CFG  = "output/gnb_config.pkl"
XML_IN   = "output/interlagos.xml"
NPZ_OUT  = "output/ue_grid.npz"
PNG_OUT  = "output/ue_grid.png"

GRID_EAST  = (-400., 400.)  # (min, max) em metros
GRID_NORTH = (-400., 400.)
GRID_STEP  = 20.            # passo da grade em metros
UE_HEIGHT  = 1.5            # altitude dos UEs em metros
PG_THRESH  = -140.          # limiar de cobertura em dB (path_gain)

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

os.makedirs("output", exist_ok=True)

# ─── Carrega config da gNB ───────────────────────────────────────────────────
if not os.path.exists(PKL_CFG):
    raise FileNotFoundError(f"Rode 'make gnb' primeiro — {PKL_CFG} não encontrado.")

with open(PKL_CFG, "rb") as f:
    cfg = pickle.load(f)

pos_gnb = cfg["pos_enu"]
info(f"gNB ENU: E={pos_gnb[0]:.2f} m, N={pos_gnb[1]:.2f} m, Z={pos_gnb[2]:.1f} m")

# ─── Carrega e configura cena ────────────────────────────────────────────────
info(f"Carregando cena: {XML_IN}")
scene = rt.load_scene(XML_IN)
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]

scene.tx_array = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"],
)
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(rt.Transmitter(name="gnb", position=pos_gnb, orientation=cfg["orientacao"]))
ok("Cena e gNB configurados.")

# ─── Calcula RadioMap para área do grid ──────────────────────────────────────
# Centro do grid em ENU (0, 0, 1.5) → Mitsuba Y-up (X=0, Y=1.5, Z=0)
info("Calculando RadioMap (840×840 m, cell=5 m, 1M amostras, max_depth=3)...")
rm_solver = rt.RadioMapSolver()
rm = rm_solver(
    scene=scene,
    center=[0., UE_HEIGHT, 0.],        # Mitsuba Y-up: X=East, Y=Up, Z=-North
    orientation=[0., 0., math.pi / 2], # plano horizontal East×North
    size=[840., 840.],                 # cobre -420 a +420 m → margem extra
    cell_size=[5., 5.],
    samples_per_tx=1_000_000,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    diffraction=True,
)
ok("RadioMap calculado.")

# Converte cell_centers (Mitsuba Y-up) → ENU
cc      = np.array(rm.cell_centers)    # (rows, cols, 3)
cc_east = cc[..., 0]                   # X_mitsuba = East
cc_north = -cc[..., 2]                 # -Z_mitsuba = North
pg      = np.array(rm.path_gain)       # (num_tx, rows, cols)
pg_db   = 10 * np.log10(pg[0] + 1e-30)  # dB — apenas a gNB (TX 0)

info(f"RadioMap: {pg_db.shape[0]}×{pg_db.shape[1]} células | "
     f"East [{cc_east.min():.0f}, {cc_east.max():.0f}] m | "
     f"North [{cc_north.min():.0f}, {cc_north.max():.0f}] m")

# ─── Gera grade de UEs ────────────────────────────────────────────────────────
east_vals  = np.arange(GRID_EAST[0],  GRID_EAST[1]  + 0.1, GRID_STEP)
north_vals = np.arange(GRID_NORTH[0], GRID_NORTH[1] + 0.1, GRID_STEP)
grid_east, grid_north = np.meshgrid(east_vals, north_vals)  # (N_north, N_east)

n_east  = len(east_vals)
n_north = len(north_vals)
n_total = n_east * n_north

positions = np.column_stack([
    grid_east.flatten(),
    grid_north.flatten(),
    np.full(n_total, UE_HEIGHT),
])
info(f"Grid: {n_east}×{n_north} = {n_total} UEs | passo={GRID_STEP:.0f} m")

# ─── Interpola path_gain nas posições do grid ────────────────────────────────
info("Interpolando path_gain nas posições do grid...")

pts_rm  = np.column_stack([cc_east.flatten(), cc_north.flatten()])
vals_rm = pg_db.flatten()
finite  = np.isfinite(vals_rm)

interp = LinearNDInterpolator(pts_rm[finite], vals_rm[finite], fill_value=-300.)
pg_at_ue = interp(positions[:, 0], positions[:, 1])  # (N,)

valid_mask = pg_at_ue >= PG_THRESH
n_valid = int(valid_mask.sum())
pct = 100. * n_valid / n_total

ok(f"UEs com cobertura (path_gain ≥ {PG_THRESH:.0f} dB): "
   f"{n_valid}/{n_total}  ({pct:.1f}%)")
ok(f"UEs sem cobertura: {n_total - n_valid}")

# ─── Salva NPZ ────────────────────────────────────────────────────────────────
np.savez(NPZ_OUT,
         positions=positions,          # (N, 3) — ENU em metros
         valid_mask=valid_mask,        # (N,) bool
         pg_at_ue=pg_at_ue,           # (N,) path_gain interpolado em dB
         east_vals=east_vals,
         north_vals=north_vals,
         gnb_pos=np.array(pos_gnb),
         grid_step=np.float64(GRID_STEP),
         pg_threshold=np.float64(PG_THRESH))
ok(f"Grid salvo: {NPZ_OUT}")

# ─── Gera visualização ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Painel esquerdo: RadioMap + posições dos UEs
e0, e1 = cc_east.min(),  cc_east.max()
n0, n1 = cc_north.min(), cc_north.max()
vmin = np.nanpercentile(pg_db[pg_db > -250], 5)
vmax = np.nanpercentile(pg_db[pg_db > -250], 99)
im = axes[0].imshow(pg_db, origin="lower", cmap="jet",
                    vmin=vmin, vmax=vmax,
                    extent=[e0, e1, n0, n1], aspect="equal")
plt.colorbar(im, ax=axes[0], label="Path Gain (dB)")

# UEs válidos (branco) e bloqueados (vermelho)
axes[0].scatter(positions[valid_mask,  0], positions[valid_mask,  1],
                c="white", s=4, alpha=0.6, label=f"válido ({n_valid})")
axes[0].scatter(positions[~valid_mask, 0], positions[~valid_mask, 1],
                c="red", s=4, alpha=0.6, label=f"sem cobertura ({n_total - n_valid})")
axes[0].plot(pos_gnb[0], pos_gnb[1], "w^", ms=12, markeredgecolor="k", label="gNB")
axes[0].set_title(f"Path Gain + grade UEs (threshold={PG_THRESH:.0f} dB)")
axes[0].set_xlabel("Leste (m)"); axes[0].set_ylabel("Norte (m)")
axes[0].legend(fontsize=8)

# Painel direito: máscara de cobertura
valid_img = valid_mask.reshape(n_north, n_east).astype(float)
axes[1].imshow(valid_img, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[east_vals[0], east_vals[-1], north_vals[0], north_vals[-1]],
               aspect="equal")
axes[1].plot(pos_gnb[0], pos_gnb[1], "w^", ms=12, markeredgecolor="k", label="gNB")
axes[1].set_title(f"Cobertura (verde=válido | {n_valid}/{n_total} UEs = {pct:.0f}%)")
axes[1].set_xlabel("Leste (m)"); axes[1].set_ylabel("Norte (m)")
axes[1].legend()

plt.tight_layout()
plt.suptitle(f"Grade de UEs — Interlagos | {cfg['freq_hz']/1e9:.1f} GHz | "
             f"step={GRID_STEP:.0f} m | UE height={UE_HEIGHT:.1f} m",
             fontsize=11, y=1.01)
plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Visualização salva: {PNG_OUT}")

print()
ok(f"=== generate_ue_grid.py concluído — {n_valid} UEs válidos para simulação ===")
ok("Próximo: make simulate")
