"""
relabel_paths.py — Patch retroativo nas labels de path_type do measurements.npz.

Para cada UE, re-roda PathSolver apenas para extrair interactions do caminho
dominante (RSRP e AoA são mantidos do measurements.npz original).
Aplica a hierarquia corrigida e salva como measurements_v2.npz / .mat.

Hierarquia:
  nenhum hop real                        → "LoS"
  algum hop DIFFRACTION                  → "diffracted"
  algum hop SPECULAR ou DIFFUSE          → "reflected"
  somente hops REFRACTION                → "transmitted"  (antes "refracted")
  fallback                               → "none"
"""

import os, pickle, math, time
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

NPZ_IN  = "output/measurements.npz"
NPZ_OUT = "output/measurements_v2.npz"
MAT_OUT = "output/measurements_v2.mat"
PKL_CFG = "output/gnb_config.pkl"
XML_IN  = "output/interlagos.xml"

# ─── Carrega medidas originais ────────────────────────────────────────────────
info(f"Carregando medidas originais: {NPZ_IN}")
meas = np.load(NPZ_IN, allow_pickle=True)
rsrp_dbm   = meas["rsrp_dbm"].copy()
aoa_az_rad = meas["aoa_az_rad"].copy()
aoa_el_rad = meas["aoa_el_rad"].copy()
path_type  = meas["path_type"].copy()        # labels originais (strings)
n_paths    = meas["n_paths"].copy()
pos_east   = meas["pos_east"].copy()
pos_north  = meas["pos_north"].copy()
pos_up     = meas["pos_up"].copy()
gnb_pos    = meas["gnb_pos"].copy()
freq_hz    = float(meas["freq_hz"])
p_tx_w     = float(meas["p_tx_w"])
n_ues      = len(rsrp_dbm)
info(f"{n_ues} UEs carregados | categorias originais:")
for t in ["LoS", "reflected", "refracted", "transmitted", "diffracted", "none"]:
    cnt = int((path_type == t).sum())
    if cnt: info(f"  {t:<12s}: {cnt}")

# ─── Carrega cena ─────────────────────────────────────────────────────────────
with open(PKL_CFG, "rb") as f:
    cfg = pickle.load(f)
pos_gnb = np.array(cfg["pos_enu"], dtype=float)

scene = rt.load_scene(XML_IN)
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]
scene.tx_array  = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"])
scene.rx_array  = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(rt.Transmitter(name="gnb", position=pos_gnb, orientation=cfg["orientacao"]))
ok("Cena carregada")

# ─── Função de classificação corrigida ────────────────────────────────────────
def classificar(inter_dom):
    """Aplica hierarquia sobre o CONJUNTO de InteractionTypes do caminho dominante."""
    inter_set = set(inter_dom.tolist())
    inter_set.discard(int(InteractionType.NONE))
    if not inter_set:
        return "LoS"
    elif int(InteractionType.DIFFRACTION) in inter_set:
        return "diffracted"
    elif (int(InteractionType.SPECULAR) in inter_set or
          int(InteractionType.DIFFUSE)  in inter_set):
        return "reflected"
    elif int(InteractionType.REFRACTION) in inter_set:
        return "transmitted"
    else:
        return "none"

# ─── Re-roda PathSolver para obter interactions ───────────────────────────────
solver      = rt.PathSolver()
# dtype object evita truncagem ao atribuir "transmitted" (11 chars) em
# array originalmente dimensionado para "refracted" (9 chars)
new_labels  = np.array(path_type.tolist(), dtype=object)
changed     = np.zeros(n_ues, dtype=bool)
# converte para string antes de comparar
path_type   = np.array(path_type.tolist(), dtype=object)

info(f"Re-classificando {n_ues} UEs...")
t0 = time.time()

for i in tqdm(range(n_ues), desc="relabel", unit="UE", dynamic_ncols=True):
    pos_ue = [float(pos_east[i]), float(pos_north[i]), float(pos_up[i])]
    scene.add(rt.Receiver(name="rx_ue", position=pos_ue))
    try:
        paths = solver(
            scene=scene, max_depth=5,
            los=True, specular_reflection=True,
            diffraction=True, diffuse_reflection=False,
            synthetic_array=True,
        )
        valid_flat = np.array(paths.valid)[0, 0, :]   # (num_paths,)
        n_v = int(valid_flat.sum())

        if n_v == 0:
            new_label = "none"
        else:
            # Potência para encontrar caminho dominante
            a_r = np.array(paths.a[0]); a_i = np.array(paths.a[1])
            pow_path  = (a_r**2 + a_i**2).sum(axis=tuple(range(a_r.ndim - 1)))
            pow_valid = pow_path * valid_flat.astype(float)
            dom_idx   = int(np.argmax(pow_valid))

            inter     = np.array(paths.interactions)   # (max_depth, 1, 1, num_paths)
            inter_dom = inter[:, 0, 0, dom_idx]
            new_label = classificar(inter_dom)

        if new_label != path_type[i]:
            changed[i] = True
        new_labels[i] = new_label

    except Exception as exc:
        warn(f"UE {i}: erro — {exc}")
    finally:
        scene.remove("rx_ue")

t_total = time.time() - t0
ok(f"Re-classificação concluída em {t_total:.1f} s")

# ─── Tabela de transição ──────────────────────────────────────────────────────
CATEGORIAS = ["LoS", "reflected", "refracted", "transmitted", "diffracted", "none"]
print()
print("=" * 65)
print("TABELA DE TRANSIÇÃO  (original → novo)")
print("=" * 65)
print(f"{'Original':<14} {'Novo':<14} {'Count':>6}  {'%Total':>7}")
print("-" * 65)

from collections import Counter
trans = Counter(zip(path_type.tolist(), new_labels.tolist()))
for (orig, novo), cnt in sorted(trans.items(), key=lambda x: -x[1]):
    marker = "  ← MUDOU" if orig != novo else ""
    print(f"  {orig:<12} → {novo:<12}  {cnt:>6}  ({100*cnt/n_ues:5.1f}%){marker}")

print("-" * 65)
n_changed = int(changed.sum())
print(f"  Total UEs que mudaram de label: {n_changed} / {n_ues} ({100*n_changed/n_ues:.1f}%)")
print()
print("Distribuição final:")
for t in ["LoS", "reflected", "transmitted", "diffracted", "none"]:
    cnt = int((new_labels == t).sum())
    if cnt:
        print(f"  {t:<14}: {cnt:5d}  ({100*cnt/n_ues:5.1f}%)")
print("=" * 65)

# ─── Salva NPZ v2 ─────────────────────────────────────────────────────────────
np.savez(NPZ_OUT,
         rsrp_dbm   = rsrp_dbm,
         aoa_az_rad = aoa_az_rad,
         aoa_el_rad = aoa_el_rad,
         path_type  = new_labels,
         n_paths    = n_paths,
         pos_east   = pos_east,
         pos_north  = pos_north,
         pos_up     = pos_up,
         gnb_pos    = gnb_pos,
         freq_hz    = np.float64(freq_hz),
         p_tx_w     = np.float64(p_tx_w))
ok(f"Salvo: {NPZ_OUT}")

# ─── Salva MAT v2 ─────────────────────────────────────────────────────────────
pt_mat = np.empty((n_ues, 1), dtype=object)
for j, t in enumerate(new_labels): pt_mat[j, 0] = str(t)
savemat(MAT_OUT, {
    "rsrp_dBm":    rsrp_dbm.reshape(-1,1),
    "aoa_az_rad":  aoa_az_rad.reshape(-1,1),
    "aoa_el_rad":  aoa_el_rad.reshape(-1,1),
    "path_type":   pt_mat,
    "n_paths":     n_paths.reshape(-1,1).astype(float),
    "pos_east_m":  pos_east.reshape(-1,1),
    "pos_north_m": pos_north.reshape(-1,1),
    "pos_up_m":    pos_up.reshape(-1,1),
    "gnb_pos_m":   gnb_pos.reshape(1,3),
    "freq_Hz":     freq_hz,
    "P_tx_W":      p_tx_w,
})
ok(f"Salvo: {MAT_OUT}")
ok("=== relabel_paths.py concluído ===")
