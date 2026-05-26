"""
smoke_test_gnb.py — Testa propagação com a gNB na cena Interlagos.

Carrega a cena XML, adiciona a gNB nas coordenadas ENU convertidas,
roda PathSolver e RadioMapSolver, e salva imagens de resultado.

Uso:
    python scripts/smoke_test_gnb.py
"""

import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # sem display
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import sionna.rt as rt

# ─── Caminhos ─────────────────────────────────────────────────────────────────
XML_IN       = "output/interlagos.xml"
PKL_CFG      = "output/gnb_config.pkl"
OUT_RADIOMAP = "output/radiomap_gnb.png"
OUT_RENDER   = "output/scene_with_gnb.png"

R_EARTH = 6_378_137

def lat_lon_to_enu(lat, lon, lat0, lon0):
    m = np.pi / 180.0 * R_EARTH
    return (lon - lon0) * m * np.cos(np.radians(lat0)), (lat - lat0) * m

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

os.makedirs("output", exist_ok=True)

# ─── Carrega configuração da gNB ─────────────────────────────────────────────
if os.path.exists(PKL_CFG):
    with open(PKL_CFG, "rb") as f:
        cfg = pickle.load(f)
    info(f"Configuração carregada de: {PKL_CFG}")
else:
    warn(f"{PKL_CFG} não encontrado — usando defaults.")
    cfg = {
        "lat0": -23.7019, "lon0": -46.7009,
        "lat_gnb": -23.701926, "lon_gnb": -46.700974,
        "h_gnb": 25.0, "freq_hz": 3.5e9, "bw_hz": 100e6,
        "num_rows": 8, "num_cols": 8, "spacing": 0.5,
        "pattern": "tr38901", "polariz": "VH",
        "orientacao": [0., 0., 0.],
    }
    east, north = lat_lon_to_enu(cfg["lat_gnb"], cfg["lon_gnb"], cfg["lat0"], cfg["lon0"])
    cfg["pos_enu"] = [east, north, cfg["h_gnb"]]

pos_gnb = cfg["pos_enu"]
info(f"gNB ENU: E={pos_gnb[0]:.2f} m, N={pos_gnb[1]:.2f} m, Z={pos_gnb[2]:.1f} m")

# ─── Carrega e configura cena ─────────────────────────────────────────────────
info(f"Carregando cena: {XML_IN}")
scene = rt.load_scene(XML_IN)
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]

scene.tx_array = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"],
)
scene.rx_array = rt.PlanarArray(
    num_rows=1, num_cols=1, pattern="iso", polarization="V",
)

scene.add(rt.Transmitter(name="gnb", position=pos_gnb, orientation=cfg["orientacao"]))
scene.add(rt.Receiver(name="rx_test", position=[100., 0., 1.5]))
ok(f"gNB e RX de teste adicionados.")

# ─── PathSolver ───────────────────────────────────────────────────────────────
print()
info("Rodando PathSolver (max_depth=3, LOS + reflexão + difração, synthetic_array=True)...")
solver = rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffraction=True,
    diffuse_reflection=False,
    synthetic_array=True,
)

tau_arr   = np.array(paths.tau)    # shape [num_rx, num_tx, num_paths]
a_arr     = np.array(paths.a)      # shape [..., num_paths]
valid_arr = np.array(paths.valid)  # shape [num_rx, num_tx, num_paths]
theta_r   = np.array(paths.theta_r)
phi_r     = np.array(paths.phi_r)

# Extrai apenas caminhos válidos
valid_flat = valid_arr.flatten()
tau_flat   = tau_arr.flatten()[valid_flat]
a_flat     = a_arr.reshape(-1, a_arr.shape[-1])  # colapsa dimensões de antena

# Potência por caminho: soma |a|² sobre dimensões de antena
# a tem shape [num_rx_ant, num_tx_ant, num_rx, num_tx, num_paths] em synthetic_array
a_sq = np.abs(np.array(paths.a)) ** 2
pow_per_path = a_sq.sum(axis=tuple(range(a_sq.ndim - 1)))  # soma tudo exceto num_paths
pow_valid     = pow_per_path[valid_arr.flatten()[:len(pow_per_path)]]

num_caminhos = int(valid_flat.sum())
ok(f"Número de caminhos encontrados: {num_caminhos}")

if num_caminhos > 0:
    tau_sorted_idx = np.argsort(tau_flat)
    tau_1 = tau_flat[tau_sorted_idx[0]] * 1e9
    tau_esperado = np.sqrt(100**2 + (pos_gnb[2] - 1.5)**2) / 3e8 * 1e9
    ok(f"Atraso 1º caminho : {tau_1:.2f} ns  (LoS esperado: {tau_esperado:.2f} ns)")

    pot_total_db = 10 * np.log10(float(pow_per_path.sum()) + 1e-30)
    ok(f"Potência total    : {pot_total_db:.2f} dB rel.")

    # Caminho dominante (maior potência)
    dom_idx = int(np.argmax(pow_per_path))
    th_r = float(np.array(paths.theta_r).flatten()[dom_idx]) * 180 / np.pi
    ph_r = float(np.array(paths.phi_r).flatten()[dom_idx]) * 180 / np.pi
    ok(f"Caminho dominante : θ_r={th_r:.1f}°  φ_r={ph_r:.1f}°")
else:
    warn("Nenhum caminho válido encontrado.")
    pot_total_db = float("-inf")

# ─── RadioMapSolver ───────────────────────────────────────────────────────────
print()
info("Rodando RadioMapSolver (cell_size=5m, samples_per_tx=1e6, max_depth=5)...")
# Plano horizontal a 1.5 m de altitude.
# Centro em Mitsuba Y-up: X=East=-76, Y=1.5m(Up), Z=-North=126
# Orientação (0,0,π/2) → superfície no plano ENU (East×North), normal = Up
# Tamanho cobre toda a cena: East 1880 m × North 1860 m
import math as _math
rm_solver = rt.RadioMapSolver()
rm = rm_solver(
    scene=scene,
    center=[-76., 1.5, 126.],          # Mitsuba Y-up (X=E, Y=alt, Z=-N)
    orientation=[0., 0., _math.pi/2],  # plano horizontal
    size=[1880., 1860.],               # East × North (m)
    cell_size=[5., 5.],
    samples_per_tx=1_000_000,
    max_depth=5,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    diffraction=True,
)
ok("RadioMap calculado.")

# Salva radio map como PNG via matplotlib
# cell_centers shape: (rows, cols, 3) — coordenadas Mitsuba Y-up
cc  = np.array(rm.cell_centers)                        # (rows, cols, 3)
# converte Mitsuba(X,Y,Z) → ENU(East, North, Up)
cc_east  = cc[..., 0]                                  # (rows, cols)
cc_north = -cc[..., 2]                                 # (rows, cols)

pg    = np.array(rm.path_gain)                         # (num_tx, rows, cols)
pg_db = 10 * np.log10(pg[0] + 1e-30)                  # primeira TX (gNB)

# Extent para imshow: [x_min, x_max, y_min, y_max] em ENU
e0, e1 = cc_east.min(),  cc_east.max()
n0, n1 = cc_north.min(), cc_north.max()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(
    pg_db,
    origin="lower",
    cmap="jet",
    vmin=np.nanpercentile(pg_db[pg_db > -200], 5),
    vmax=np.nanpercentile(pg_db[pg_db > -200], 99),
    extent=[e0, e1, n0, n1],
    aspect="equal",
)
plt.colorbar(im, ax=ax, label="Path Gain (dB)")
ax.plot(pos_gnb[0], pos_gnb[1], "w^", ms=10, label=f"gNB ({pos_gnb[0]:.1f},{pos_gnb[1]:.1f})")
ax.plot(100., 0., "wo", ms=8, label="RX teste (100,0)")
ax.set_xlabel("Leste (m)")
ax.set_ylabel("Norte (m)")
ax.set_title(f"Radio Map — Path Gain [dB]  |  {cfg['freq_hz']/1e9:.1f} GHz  |  h=1.5 m")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_RADIOMAP, dpi=150)
plt.close()
ok(f"Radio map salvo: {OUT_RADIOMAP}")

# ─── Render da cena com a gNB ────────────────────────────────────────────────
print()
info("Renderizando cena com gNB (1280×720)...")
try:
    cam = rt.Camera(
        position=[pos_gnb[0] + 200, pos_gnb[1] + 200, 150],
        look_at=pos_gnb,
    )
    bmp = scene.render_to_file(
        camera=cam,
        filename=OUT_RENDER,
        resolution=(1280, 720),
        num_samples=512,
        radio_map=rm,
        rm_metric="path_gain",
        show_devices=True,
    )
    ok(f"Render salvo: {OUT_RENDER}")
except Exception as e:
    warn(f"Render falhou: {e}")
    warn("Continuando sem render.")

print()
ok("=== smoke_test_gnb.py concluído ===")
