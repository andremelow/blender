"""
place_gnb.py — Converte coordenadas GPS da gNB para ENU local e salva
               a configuração em output/gnb_config.pkl.

A API do Sionna RT não expõe serialização de cena, então salvamos um
pickle com os parâmetros necessários para recriar a gNB em outros scripts.

Uso:
    python scripts/place_gnb.py
"""

import os
import sys
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import sionna.rt as rt

# ─── Parâmetros da cena ───────────────────────────────────────────────────────
XML_IN   = "output/interlagos.xml"
PKL_OUT  = "output/gnb_config.pkl"

FREQ_HZ    = 3.5e9    # n78
BW_HZ      = 100e6

# ─── Parâmetros da gNB ────────────────────────────────────────────────────────
LAT0     = -23.7019   # latitude da origem da cena
LON0     = -46.7009   # longitude da origem da cena
LAT_GNB  = -23.701926
LON_GNB  = -46.700974
# Torre trelicada autoportante, seção triangular 3 pernas, sobre prédio 2 pavimentos (~8.5 m).
# Balizamento ANAC diurno (branco/laranja). Altura confirmada pelo operador.
H_GNB    = 62.0       # altura total AGL (prédio + torre); confirmada
R_EARTH  = 6_378_137  # raio médio da Terra (m)

# Array UPA 8×8 dual-pol, spacing 0.5λ, padrão 3GPP TR 38.901
NUM_ROWS  = 8
NUM_COLS  = 8
SPACING   = 0.5       # em comprimentos de onda
PATTERN   = "tr38901"
POLARIZ   = "VH"      # dual-pol

ORIENTACAO = [0., 0., 0.]   # (alpha, beta, gamma) em radianos — ajustar depois

# ─── Conversão GPS → ENU ─────────────────────────────────────────────────────
def lat_lon_to_enu(lat, lon, lat0, lon0, r_earth=R_EARTH):
    """Aproximação plana (boa para distâncias < 10 km)."""
    m_per_deg = np.pi / 180.0 * r_earth
    north = (lat - lat0) * m_per_deg
    east  = (lon - lon0) * m_per_deg * np.cos(np.radians(lat0))
    return float(east), float(north)

def info(msg):  print(f"[INFO] {msg}", flush=True)
def ok(msg):    print(f"[ OK ] {msg}", flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
info(f"Carregando cena: {XML_IN}")
scene = rt.load_scene(XML_IN)
ok(f"Cena carregada — {len(scene.objects)} objeto(s).")

# Frequência e banda
scene.frequency  = FREQ_HZ
scene.bandwidth  = BW_HZ
info(f"Frequência: {FREQ_HZ/1e9:.1f} GHz  |  Banda: {BW_HZ/1e6:.0f} MHz")

# Array de transmissão UPA 8×8
scene.tx_array = rt.PlanarArray(
    num_rows=NUM_ROWS,
    num_cols=NUM_COLS,
    vertical_spacing=SPACING,
    horizontal_spacing=SPACING,
    pattern=PATTERN,
    polarization=POLARIZ,
)
# Array de recepção isotrópico (UE genérico)
scene.rx_array = rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="iso",
    polarization="V",
)
ok(f"TX array: UPA {NUM_ROWS}×{NUM_COLS} dual-pol ({PATTERN}), spacing={SPACING}λ")

# Conversão GPS → ENU
east_gnb, north_gnb = lat_lon_to_enu(LAT_GNB, LON_GNB, LAT0, LON0)
pos_gnb = [east_gnb, north_gnb, H_GNB]

info(f"Origem da cena : lat={LAT0}°, lon={LON0}°")
info(f"GPS da gNB     : lat={LAT_GNB}°, lon={LON_GNB}°")
info(f"Delta           : Δlat={LAT_GNB-LAT0:.6f}°, Δlon={LON_GNB-LON0:.6f}°")
ok(f"Posição ENU local: E={east_gnb:.2f} m, N={north_gnb:.2f} m, Z={H_GNB:.1f} m")

# Cria e adiciona o Transmitter
gnb = rt.Transmitter(name="gnb", position=pos_gnb, orientation=ORIENTACAO)
scene.add(gnb)
ok(f"Transmitter 'gnb' adicionado à cena.")

# ─── Salva configuração em pickle ────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
config = {
    "lat0":      LAT0,
    "lon0":      LON0,
    "lat_gnb":   LAT_GNB,
    "lon_gnb":   LON_GNB,
    "h_gnb":     H_GNB,
    "pos_enu":   pos_gnb,
    "freq_hz":   FREQ_HZ,
    "bw_hz":     BW_HZ,
    "num_rows":  NUM_ROWS,
    "num_cols":  NUM_COLS,
    "spacing":   SPACING,
    "pattern":   PATTERN,
    "polariz":   POLARIZ,
    "orientacao": ORIENTACAO,
}
with open(PKL_OUT, "wb") as f:
    pickle.dump(config, f)
ok(f"Configuração salva em: {PKL_OUT}")

print()
ok("=== place_gnb.py concluído — próximo: make smoke-gnb ===")
