"""
export_to_matlab.py — Converte saídas do Sionna RT para formato MATLAB.

Modos:
  --source grid        (padrão) measurements_v2.npz  → measurements_rt.mat
  --source spectators            measurements_spectators.npz → measurements_spectators_v2.mat

Campos exportados (ambos os modos):
  rsrp_dBm    (1×N double) : potência recebida (dBm)
  aoa_az_rad  (1×N double) : azimute CCW do +East, conv. Sionna (gNB→UE)
  aoa_el_rad  (1×N double) : elevação Sionna (negativa = gNB acima do UE)
  pos_east    (1×N double) : East verdadeiro relativo à gNB (m)
  pos_north   (1×N double) : North verdadeiro relativo à gNB (m)
  d_true      (1×N double) : distância 3D real gNB→UE (m)
  path_type   (1×N cell)   : categoria do caminho dominante
  n_paths     (1×N double) : número de caminhos detectados
  zone_name   (1×N cell)   : setor/zona do UE (spectators) ou '' (grid)
  gnb_pos     (1×3 double) : [0, 0, -gnb_up] frame MATLAB [East, North, Down]
  n_ues       (1×1 double) : número de UEs

Parâmetros de calibração embutidos (para uso direto no MATLAB):
  calib_n_naive    (1×1) : expoente n naive (2.7)
  calib_A_naive    (1×1) : offset A naive em dBm
  calib_n_global   (1×1) : n global calibrado
  calib_A_global   (1×1) : A global calibrado
  calib_n_los      (1×1) : n LoS calibrado
  calib_A_los      (1×1) : A LoS calibrado
"""

import os, argparse
import numpy as np
from scipy.io import savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["grid", "spectators"], default="grid",
                    help="Fonte dos dados Sionna RT")
parser.add_argument("--npz",    default=None,
                    help="Override do arquivo npz de entrada")
parser.add_argument("--out",    default=None,
                    help="Override do arquivo .mat de saída")
args = parser.parse_args()

SOURCE_DEFAULTS = {
    "grid":        ("output/measurements_v2.npz",          "output/measurements_rt.mat"),
    "spectators":  ("output/measurements_spectators.npz",  "output/measurements_spectators_v2.mat"),
}
npz_path, out_path = SOURCE_DEFAULTS[args.source]
if args.npz: npz_path = args.npz
if args.out: out_path = args.out

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ── Parâmetros de calibração (fitados em measurements_v2, 1681 UEs) ────────────
CALIB = {
    "naive":  {"n": 2.70,  "A": -7.0},    # baseline: EIRP = Pt+Gt+Gr-PL0 = 23+8+0-38
    "global": {"n": 1.362, "A": -32.95},  # regressão global
    "los":    {"n": 1.525, "A": -23.92},  # regressão LoS
}

# ── Carrega medidas Sionna RT ──────────────────────────────────────────────────
info(f"Fonte: {args.source}  |  Arquivo: {npz_path}")
d      = np.load(npz_path, allow_pickle=True)
pos_e  = d["pos_east"]
pos_n  = d["pos_north"]
rsrp   = d["rsrp_dbm"]
aoa_az = d["aoa_az_rad"]
aoa_el = d["aoa_el_rad"]
ptype  = d["path_type"]
n_paths_arr = d["n_paths"].astype(float)
gnb    = d["gnb_pos"]          # ENU: [East, North, Up]
n_ues  = len(rsrp)

# Posições relativas à gNB (gNB = (0,0) no frame MATLAB)
pos_east_rel  = pos_e - gnb[0]
pos_north_rel = pos_n - gnb[1]

# Distância 3D real gNB→UE
ue_up = float(d["pos_up"].flat[0]) if "pos_up" in d.files else 1.5
delta_h = ue_up - float(gnb[2])           # negativo (UE abaixo da gNB)
d_true  = np.sqrt(pos_east_rel**2 + pos_north_rel**2 + delta_h**2)

# Zone names (espectadores) ou string vazia (grid)
if args.source == "spectators":
    ue_sp = np.load("output/ue_spectators.npz", allow_pickle=True)
    zone_names = ue_sp["zone_names"].astype(str)
else:
    zone_names = np.array([""] * n_ues)

info(f"gNB ENU: East={gnb[0]:.2f} m, North={gnb[1]:.2f} m, Up={gnb[2]:.1f} m")
info(f"UEs: {n_ues}  |  válidos (RSRP finito): {np.isfinite(rsrp).sum()}")
info(f"East rel gNB: [{np.nanmin(pos_east_rel):.0f}, {np.nanmax(pos_east_rel):.0f}] m")
info(f"North rel gNB: [{np.nanmin(pos_north_rel):.0f}, {np.nanmax(pos_north_rel):.0f}] m")
info(f"d_true: [{np.nanmin(d_true):.0f}, {np.nanmax(d_true):.0f}] m")
info(f"RSRP: [{np.nanmin(rsrp):.1f}, {np.nanmax(rsrp):.1f}] dBm")

# Cell arrays para MATLAB
def to_cell(arr):
    c = np.empty((1, len(arr)), dtype=object)
    for i, s in enumerate(arr):
        c[0, i] = str(s)
    return c

pt_cell   = to_cell(ptype)
zone_cell = to_cell(zone_names)

# Frame MATLAB: [East, North, Down]
gnb_pos_matlab = np.array([[0.0, 0.0, -float(gnb[2])]])

os.makedirs("output", exist_ok=True)
savemat(out_path, {
    "rsrp_dBm":   rsrp.reshape(1, -1),
    "aoa_az_rad": aoa_az.reshape(1, -1),
    "aoa_el_rad": aoa_el.reshape(1, -1),
    "pos_east":   pos_east_rel.reshape(1, -1),
    "pos_north":  pos_north_rel.reshape(1, -1),
    "d_true":     d_true.reshape(1, -1),
    "n_paths":    n_paths_arr.reshape(1, -1),
    "path_type":  pt_cell,
    "zone_name":  zone_cell,
    "gnb_pos":    gnb_pos_matlab,
    "n_ues":      np.float64(n_ues),
    # Parâmetros de calibração embutidos
    "calib_n_naive":  np.float64(CALIB["naive"]["n"]),
    "calib_A_naive":  np.float64(CALIB["naive"]["A"]),
    "calib_n_global": np.float64(CALIB["global"]["n"]),
    "calib_A_global": np.float64(CALIB["global"]["A"]),
    "calib_n_los":    np.float64(CALIB["los"]["n"]),
    "calib_A_los":    np.float64(CALIB["los"]["A"]),
})
ok(f"MAT salvo: {out_path}  ({n_ues} UEs)")

# ── Resumo por categoria ───────────────────────────────────────────────────────
print()
print(f"{'Categoria':<14} {'N':>6}  {'RSRP med':>9}  {'d_true med':>10}  {'AoA span':>9}")
print("-" * 55)
for t in ["LoS", "reflected", "diffracted", "transmitted", "none"]:
    mask = (ptype == t) & np.isfinite(rsrp)
    if not mask.any(): continue
    print(f"  {t:<12}: {mask.sum():>6}  "
          f"{np.nanmedian(rsrp[mask]):>8.1f} dBm  "
          f"{np.nanmedian(d_true[mask]):>9.0f} m  "
          f"{np.degrees(np.nanmax(aoa_az[mask])-np.nanmin(aoa_az[mask])):>7.1f}°")
