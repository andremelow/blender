"""
simulate_ues.py — Simula canal Sionna RT para cada UE do grid.

Para cada UE válido:
  - Adiciona Receiver na posição do UE
  - Roda PathSolver (max_depth=3, LoS + reflexão especular + difração)
  - Extrai RSRP [dBm], AoA na gNB (phi_t, theta_t via reciprocidade),
    tipo do caminho dominante e número de caminhos válidos
  - Remove Receiver e salva checkpoint a cada 50 UEs

AoA na gNB: como a gNB é TX e o UE é RX, os ângulos de partida da gNB
(phi_t, theta_t) equivalem, por reciprocidade, ao AoA quando o UE
transmite em direção à gNB. Não é necessário reverse_direction=True.

Tipo do caminho: obtido de paths.interactions (InteractionType da Sionna):
  NONE=0 (LoS), SPECULAR=1 (reflexão), DIFFUSE=2, DIFFRACTION=8

Uso:
    python scripts/simulate_ues.py [--limit N]
"""

import os
import pickle
import math
import time
import argparse
import numpy as np
import sionna.rt as rt
from sionna.rt.constants import InteractionType
from scipy.io import savemat
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

# ─── Parâmetros ───────────────────────────────────────────────────────────────
PKL_CFG  = "output/gnb_config.pkl"
XML_IN   = "output/interlagos.xml"
NPZ_GRID = "output/ue_grid.npz"
NPZ_OUT  = "output/measurements.npz"
MAT_OUT  = "output/measurements.mat"
CHK_OUT  = "output/measurements_checkpoint.npz"

P_TX_W = 0.2              # 23 dBm de potência de TX
CHECKPOINT_EVERY = 50     # salva parcial a cada N UEs

# Mapeamento InteractionType → string
ITYPE_STR = {
    InteractionType.NONE:        'LoS',
    InteractionType.SPECULAR:    'reflected',
    InteractionType.DIFFUSE:     'diffuse',
    InteractionType.REFRACTION:  'refracted',
    InteractionType.DIFFRACTION: 'diffracted',
}

# ─── Argumentos ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simula UEs no Sionna RT.")
parser.add_argument("--limit", type=int, default=None,
                    help="Máximo de UEs (None = todos os válidos)")
parser.add_argument("--resume", action="store_true",
                    help="Retoma do checkpoint existente (pula UEs já simulados)")
parser.add_argument("--all", action="store_true",
                    help="Simula todos os 1681 pontos do grid, ignorando o filtro de cobertura")
args = parser.parse_args()

os.makedirs("output", exist_ok=True)

# ─── Carrega config da gNB ───────────────────────────────────────────────────
with open(PKL_CFG, "rb") as f:
    cfg = pickle.load(f)
pos_gnb = np.array(cfg["pos_enu"], dtype=float)
info(f"gNB ENU: E={pos_gnb[0]:.2f} m, N={pos_gnb[1]:.2f} m, Z={pos_gnb[2]:.1f} m")

# ─── Carrega cena ─────────────────────────────────────────────────────────────
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

# ─── Carrega grade de UEs ─────────────────────────────────────────────────────
grid = np.load(NPZ_GRID)
all_pos   = grid["positions"]   # (N_total, 3) ENU
all_valid = grid["valid_mask"]  # (N_total,) bool

if args.all:
    valid_pos = all_pos          # todos os 1681 pontos
    info(f"Modo --all: simulando todos os {len(valid_pos)} pontos do grid (ignora filtro de cobertura).")
else:
    valid_pos = all_pos[all_valid]  # (N_valid, 3)
    n_valid   = len(valid_pos)
    if args.limit is not None and args.limit < n_valid:
        valid_pos = valid_pos[:args.limit]
        info(f"Limite ativado: simulando {len(valid_pos)}/{n_valid} UEs válidos.")
    else:
        info(f"Simulando todos os {n_valid} UEs válidos.")

n_ues = len(valid_pos)

# ─── Arrays de resultado ──────────────────────────────────────────────────────
rsrp_dbm    = np.full(n_ues, np.nan)
aoa_az_rad  = np.full(n_ues, np.nan)   # azimute de partida da gNB (= AoA na gNB)
aoa_el_rad  = np.full(n_ues, np.nan)   # elevação de partida da gNB (= AoA na gNB)
path_type   = np.empty(n_ues, dtype=object)
n_paths_arr = np.zeros(n_ues, dtype=int)
pos_east    = valid_pos[:, 0].copy()
pos_north   = valid_pos[:, 1].copy()
pos_up      = valid_pos[:, 2].copy()

# ─── Retomada de checkpoint ───────────────────────────────────────────────────
resume_from = 0
if args.resume and os.path.exists(CHK_OUT):
    chk = np.load(CHK_OUT, allow_pickle=True)
    n_done = len(chk["rsrp_dbm"])
    rsrp_dbm[:n_done]    = chk["rsrp_dbm"]
    aoa_az_rad[:n_done]  = chk["aoa_az_rad"]
    aoa_el_rad[:n_done]  = chk["aoa_el_rad"]
    path_type[:n_done]   = chk["path_type"]
    n_paths_arr[:n_done] = chk["n_paths"]
    resume_from = n_done
    info(f"Retomando do checkpoint: {n_done} UEs já prontos, restam {n_ues - n_done}.")
elif args.resume:
    warn("--resume solicitado mas checkpoint não encontrado; simulando do zero.")

solver = rt.PathSolver()

# ─── Loop por UE ──────────────────────────────────────────────────────────────
info(f"Iniciando loop — checkpoint a cada {CHECKPOINT_EVERY} UEs...")
t_loop = time.time()

for i, pos_ue in enumerate(tqdm(valid_pos, desc="UEs", unit="UE", dynamic_ncols=True)):
    if i < resume_from:
        continue
    t0 = time.time()

    scene.add(rt.Receiver(name="rx_ue", position=pos_ue.tolist()))

    try:
        paths = solver(
            scene=scene,
            max_depth=3,
            los=True,
            specular_reflection=True,
            diffraction=True,
            diffuse_reflection=False,
            synthetic_array=True,
        )

        # --- valid shape: (1, 1, num_paths) com synthetic_array=True
        valid = np.array(paths.valid)       # (1, 1, num_paths)
        valid_flat = valid[0, 0, :]         # (num_paths,) bool

        n_valid_p = int(valid_flat.sum())
        n_paths_arr[i] = n_valid_p

        if n_valid_p == 0:
            path_type[i] = 'none'
        else:
            # Potência por caminho: |a_real|² + |a_imag|²
            # paths.a retorna (a_real, a_imag) — shape (1, num_rx_ant, 1, num_tx_ant, num_paths)
            a_r = np.array(paths.a[0])      # parte real
            a_i = np.array(paths.a[1])      # parte imaginária
            a_sq = a_r ** 2 + a_i ** 2     # |a|² por antena e caminho
            # Soma sobre todas as dimensões de antena: shape → (num_paths,)
            pow_per_path = a_sq.sum(axis=tuple(range(a_sq.ndim - 1)))

            # Potência apenas dos caminhos válidos
            pow_valid = pow_per_path * valid_flat.astype(float)

            # RSRP [dBm] = P_tx_W × soma das potências → dBm
            total_pwr = float(pow_valid.sum())
            rsrp_dbm[i] = 10 * math.log10(P_TX_W * total_pwr + 1e-30) + 30.

            # Caminho dominante (maior potência entre os válidos)
            dom_idx = int(np.argmax(pow_valid))

            # AoA na gNB = ângulos de partida do TX (gNB) por reciprocidade
            # phi_t, theta_t: shape (1, 1, num_paths) com synthetic_array=True
            phi_t_all   = np.array(paths.phi_t)[0, 0, :]    # (num_paths,)
            theta_t_all = np.array(paths.theta_t)[0, 0, :]  # (num_paths,)
            aoa_az_rad[i] = float(phi_t_all[dom_idx])
            # Converte ângulo zenital → elevação: el = π/2 - θ
            aoa_el_rad[i] = math.pi / 2 - float(theta_t_all[dom_idx])

            # Tipo do caminho dominante via interactions
            # interactions shape: (max_depth, 1, 1, num_paths) com synthetic_array=True
            inter = np.array(paths.interactions)  # (max_depth, 1, 1, num_paths)
            inter_dom = inter[:, 0, 0, dom_idx]   # (max_depth,) — tipo em cada salto

            # Classifica pelo CONJUNTO de hops (não pelo primeiro hop isolado).
            # Hierarquia: LoS > diffracted > reflected > transmitted > none
            # SPECULAR→REFRACTION é "reflected" (SPECULAR presente no conjunto).
            # REFRACTION puro (travessia de parede, modo O2I) → "transmitted".
            inter_set = set(inter_dom.tolist())
            inter_set.discard(int(InteractionType.NONE))  # remove padding NONE dos hops vazios
            if not inter_set:
                path_type[i] = 'LoS'
            elif int(InteractionType.DIFFRACTION) in inter_set:
                path_type[i] = 'diffracted'
            elif (int(InteractionType.SPECULAR) in inter_set or
                  int(InteractionType.DIFFUSE)  in inter_set):
                path_type[i] = 'reflected'
            elif int(InteractionType.REFRACTION) in inter_set:
                path_type[i] = 'transmitted'  # refração pura = travessia de parede (O2I)
            else:
                path_type[i] = 'none'

    except Exception as exc:
        warn(f"UE {i} pos={pos_ue.tolist()}: PathSolver falhou — {exc}")
        path_type[i] = 'none'
    finally:
        scene.remove("rx_ue")

    # Log da primeira iteração: estimativa de ETA
    if i == 0:
        dt = time.time() - t0
        eta = dt * n_ues / 60.
        info(f"1ª iteração: {dt:.2f} s | ETA estimado: {eta:.1f} min")

    # Checkpoint periódico
    if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == n_ues:
        np.savez(CHK_OUT,
                 rsrp_dbm   = rsrp_dbm[:i+1],
                 aoa_az_rad = aoa_az_rad[:i+1],
                 aoa_el_rad = aoa_el_rad[:i+1],
                 path_type  = np.array([str(t) for t in path_type[:i+1]]),
                 n_paths    = n_paths_arr[:i+1],
                 pos_east   = pos_east[:i+1],
                 pos_north  = pos_north[:i+1],
                 pos_up     = pos_up[:i+1],
                 gnb_pos    = pos_gnb)

t_total = time.time() - t_loop
ok(f"Loop concluído: {n_ues} UEs em {t_total:.1f} s ({t_total/max(n_ues,1):.2f} s/UE)")

# ─── Resumo de tipos ──────────────────────────────────────────────────────────
ptype_str = np.array([str(t) for t in path_type])
for t in ['LoS', 'reflected', 'transmitted', 'diffracted', 'none']:
    cnt = int((ptype_str == t).sum())
    if cnt > 0:
        ok(f"  {t:<12s}: {cnt:4d} UEs ({100*cnt/n_ues:.1f}%)")

# ─── Salva NPZ ────────────────────────────────────────────────────────────────
np.savez(NPZ_OUT,
         rsrp_dbm   = rsrp_dbm,
         aoa_az_rad = aoa_az_rad,
         aoa_el_rad = aoa_el_rad,
         path_type  = ptype_str,
         n_paths    = n_paths_arr,
         pos_east   = pos_east,
         pos_north  = pos_north,
         pos_up     = pos_up,
         gnb_pos    = pos_gnb,
         freq_hz    = np.float64(cfg["freq_hz"]),
         p_tx_w     = np.float64(P_TX_W))
ok(f"Medidas salvas: {NPZ_OUT}")

# ─── Salva MAT ────────────────────────────────────────────────────────────────
try:
    # path_type como cell array MATLAB (dtype=object)
    pt_matlab = np.empty((n_ues, 1), dtype=object)
    for j in range(n_ues):
        pt_matlab[j, 0] = ptype_str[j]

    savemat(MAT_OUT, {
        "rsrp_dBm":    rsrp_dbm.reshape(-1, 1),
        "aoa_az_rad":  aoa_az_rad.reshape(-1, 1),
        "aoa_el_rad":  aoa_el_rad.reshape(-1, 1),
        "path_type":   pt_matlab,
        "n_paths":     n_paths_arr.reshape(-1, 1).astype(float),
        "pos_east_m":  pos_east.reshape(-1, 1),
        "pos_north_m": pos_north.reshape(-1, 1),
        "pos_up_m":    pos_up.reshape(-1, 1),
        "gnb_pos_m":   pos_gnb.reshape(1, 3),
        "freq_Hz":     float(cfg["freq_hz"]),
        "P_tx_W":      float(P_TX_W),
    })
    ok(f"Medidas salvas (MATLAB): {MAT_OUT}")
except Exception as exc:
    warn(f"Falha ao salvar .mat: {exc}")

print()
ok(f"=== simulate_ues.py concluído — próximo: make plot-meas ===")
