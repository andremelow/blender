"""
extract_delays.py — Extrai delays de propagação (tau_0) via PathSolver.

Re-executa o PathSolver para cada UE em measurements_v2.npz com os
mesmos parâmetros de simulate_ues.py e salva os delays em delays_v2.npz.

Parâmetros de delay:
  tau_dom  : delay do caminho dominante (maior potência), em segundos
  tau_min  : delay mínimo (primeira chegada), em segundos
  d_toa_dom: distância estimada pelo ToA do caminho dominante = c × tau_dom [m]
  d_toa_min: distância estimada pelo ToA da primeira chegada = c × tau_min [m]
  d_true   : distância 3D real gNB→UE [m]

normalize_delays=False: os delays são tempos de propagação absolutos
(não relativos à primeira chegada). c=299792458 m/s.

Uso:
    python scripts/extract_delays.py [--limit N] [--resume]

Saída: output/delays_v2.npz
"""

import os, pickle, math, time, argparse
import numpy as np
import sionna.rt as rt
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

C_LIGHT = 299_792_458.  # m/s

parser = argparse.ArgumentParser()
parser.add_argument("--npz",    default="output/measurements_v2.npz")
parser.add_argument("--out",    default="output/delays_v2.npz")
parser.add_argument("--chk",    default="output/delays_checkpoint.npz")
parser.add_argument("--limit",  type=int, default=None)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

os.makedirs("output", exist_ok=True)

# ─── Carrega posições do NPZ de referência ────────────────────────────────────
info(f"Carregando posições de: {args.npz}")
ref  = np.load(args.npz, allow_pickle=True)
pos_e  = ref["pos_east"]
pos_n  = ref["pos_north"]
pos_up = np.full(len(pos_e), 1.5)   # UE a 1.5 m AGL (mesmo que simulate_ues.py)
gnb    = ref["gnb_pos"]
ptype  = ref["path_type"]
rsrp_ref = ref["rsrp_dbm"]
n_ues  = len(pos_e)

if args.limit is not None:
    n_ues = min(n_ues, args.limit)
    info(f"Limite: processando apenas os primeiros {n_ues} UEs")

# ─── Distância real 3D ────────────────────────────────────────────────────────
delta_e3 = pos_e - gnb[0]
delta_n3 = pos_n - gnb[1]
delta_u3 = pos_up - gnb[2]
d_true   = np.sqrt(delta_e3**2 + delta_n3**2 + delta_u3**2)

# ─── Configuração da cena (igual a simulate_ues.py) ───────────────────────────
with open("output/gnb_config.pkl", "rb") as f:
    cfg = pickle.load(f)
pos_gnb = np.array(cfg["pos_enu"], dtype=float)

info("Carregando cena Sionna RT...")
scene = rt.load_scene("output/interlagos.xml")
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]
scene.tx_array  = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"],
)
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso", polarization="V")
scene.add(rt.Transmitter(name="gnb", position=pos_gnb,
                          orientation=cfg["orientacao"]))
ok("Cena configurada.")

# ─── Arrays de saída ──────────────────────────────────────────────────────────
tau_dom  = np.full(n_ues, np.nan)   # delay do caminho dominante (s)
tau_min  = np.full(n_ues, np.nan)   # menor delay (primeira chegada) (s)
n_valid  = np.zeros(n_ues, dtype=int)

# ─── Retomada de checkpoint ───────────────────────────────────────────────────
resume_from = 0
if args.resume and os.path.exists(args.chk):
    chk = np.load(args.chk, allow_pickle=True)
    n_done = int(chk["n_done"])
    resume_from = n_done
    tau_dom[:n_done] = chk["tau_dom"]
    tau_min[:n_done] = chk["tau_min"]
    n_valid[:n_done] = chk["n_valid"]
    info(f"Retomando checkpoint: {n_done}/{n_ues} UEs já processados.")
elif args.resume:
    warn("--resume solicitado mas checkpoint não encontrado; começa do zero.")

solver = rt.PathSolver()
CHKPT_EVERY = 50

# ─── Loop por UE ──────────────────────────────────────────────────────────────
info(f"Extraindo delays para {n_ues} UEs (checkpoint a cada {CHKPT_EVERY})...")
t_loop = time.time()

for i in tqdm(range(n_ues), desc="delays", unit="UE", dynamic_ncols=True):
    if i < resume_from:
        continue

    pos_ue = [float(pos_e[i]), float(pos_n[i]), float(pos_up[i])]
    scene.add(rt.Receiver(name="rx_ue", position=pos_ue))

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

        valid_flat = np.array(paths.valid)[0, 0, :]   # (num_paths,)
        n_v = int(valid_flat.sum())
        n_valid[i] = n_v

        if n_v > 0:
            # tau: shape (num_rx, num_tx, num_paths) com synthetic_array=True
            # Tempo de propagação absoluto em segundos (não normalizado pela primeira chegada)
            delays_all = np.array(paths.tau)[0, 0, :]   # (num_paths,)

            # Delay da primeira chegada (mínimo entre caminhos válidos)
            valid_delays = delays_all[valid_flat.astype(bool)]
            tau_min[i] = float(valid_delays.min())

            # Delay do caminho dominante (maior potência, igual ao simulate_ues.py)
            a_r  = np.array(paths.a[0])
            a_i  = np.array(paths.a[1])
            a_sq = a_r**2 + a_i**2
            pow_per_path = a_sq.sum(axis=tuple(range(a_sq.ndim - 1)))
            pow_valid = pow_per_path * valid_flat.astype(float)
            dom_idx = int(np.argmax(pow_valid))
            tau_dom[i] = float(delays_all[dom_idx])

    except Exception as exc:
        warn(f"UE {i}: PathSolver falhou — {exc}")
    finally:
        scene.remove("rx_ue")

    if i == 0:
        dt = time.time() - t_loop
        eta_min = dt * n_ues / 60.
        info(f"1ª iter: {dt:.2f} s | ETA: {eta_min:.0f} min")

    # Checkpoint
    if (i + 1) % CHKPT_EVERY == 0 or (i + 1) == n_ues:
        np.savez(args.chk,
                 tau_dom  = tau_dom[:i+1],
                 tau_min  = tau_min[:i+1],
                 n_valid  = n_valid[:i+1],
                 n_done   = np.array(i + 1))

t_total = time.time() - t_loop
ok(f"Loop concluído: {n_ues} UEs em {t_total:.1f} s ({t_total/max(n_ues,1):.2f} s/UE)")

# ─── Salva resultado completo ─────────────────────────────────────────────────
d_toa_dom = C_LIGHT * tau_dom
d_toa_min = C_LIGHT * tau_min

np.savez(args.out,
         tau_dom   = tau_dom,
         tau_min   = tau_min,
         d_toa_dom = d_toa_dom,
         d_toa_min = d_toa_min,
         d_true    = d_true[:n_ues],
         n_valid   = n_valid,
         pos_east  = pos_e[:n_ues],
         pos_north = pos_n[:n_ues],
         path_type = ptype[:n_ues],
         gnb_pos   = gnb,
         c_light   = np.float64(C_LIGHT))
ok(f"Delays salvos: {args.out}")

# ─── Diagnóstico rápido ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("Diagnóstico ToA (caminho dominante)")
print("=" * 60)
valid_m = np.isfinite(tau_dom[:n_ues])
d_hat_v = d_toa_dom[valid_m]
d_true_v = d_true[:n_ues][valid_m]
err_v    = np.abs(d_hat_v - d_true_v)
info(f"Erro ToA (dom): med={np.nanmedian(err_v):.1f} m  "
     f"p95={np.nanpercentile(err_v,95):.1f} m  "
     f"max={err_v.max():.1f} m")
for t in ["LoS", "diffracted", "reflected", "transmitted"]:
    mt = (ptype[:n_ues] == t) & valid_m
    if not mt.any(): continue
    e = err_v[mt[valid_m]]
    info(f"  {t:<14}: med={np.nanmedian(e):.1f} m  p95={np.nanpercentile(e,95):.1f} m")
print("=" * 60)
