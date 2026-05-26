"""
diagnose_refracted.py — Diagnóstico dos caminhos refratados.

Gera:
  1. output/material_audit.txt  — materiais da cena e suas propriedades RT
  2. stdout completo do diagnóstico de 5 UEs classificados como 'refracted'
  3. enum InteractionType da versão instalada do Sionna
"""

import os, pickle, math
import numpy as np
import sionna.rt as rt
from sionna.rt.constants import InteractionType

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def sec(msg):  print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)

# ══════════════════════════════════════════════════════════════════
# 1. ENUM InteractionType
# ══════════════════════════════════════════════════════════════════
sec("1. ENUM InteractionType (versão Sionna instalada)")
print(f"sionna version : {rt.__version__ if hasattr(rt, '__version__') else 'N/A'}")
import sionna
print(f"sionna package : {sionna.__version__}")

itype_members = {k: v for k, v in vars(InteractionType).items()
                 if not k.startswith('_')}
for name, val in sorted(itype_members.items(), key=lambda x: str(x[1])):
    print(f"  InteractionType.{name:20s} = {val!r}")

# ══════════════════════════════════════════════════════════════════
# 2. MATERIAL AUDIT
# ══════════════════════════════════════════════════════════════════
sec("2. Material Audit — carregando cena")
with open("output/gnb_config.pkl", "rb") as f:
    cfg = pickle.load(f)
pos_gnb = np.array(cfg["pos_enu"], dtype=float)

scene = rt.load_scene("output/interlagos.xml")
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]
scene.tx_array  = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"])
scene.rx_array  = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(rt.Transmitter(name="gnb", position=pos_gnb, orientation=cfg["orientacao"]))
ok("Cena carregada")

audit_lines = []
audit_lines.append(f"Sionna version : {sionna.__version__}")
audit_lines.append(f"Frequência     : {cfg['freq_hz']/1e9:.2f} GHz")
audit_lines.append(f"Num objetos    : {len(scene.objects)}")
audit_lines.append("")

for obj_name, obj in scene.objects.items():
    mat = obj.radio_material
    line = f"Objeto: {obj_name}"
    audit_lines.append(line)
    audit_lines.append("-" * len(line))
    if mat is None:
        audit_lines.append("  radio_material: None")
    else:
        audit_lines.append(f"  radio_material name      : {mat.name}")
        for attr in ["relative_permittivity", "conductivity",
                     "scattering_coefficient", "xpd_coefficient",
                     "scattering_pattern"]:
            try:
                val = getattr(mat, attr)
                audit_lines.append(f"  {attr:<30s}: {val}")
            except Exception as e:
                audit_lines.append(f"  {attr:<30s}: ERRO — {e}")
        # propriedades extras se disponíveis
        for attr in ["relative_permeability", "is_placeholder",
                     "well_conditioned"]:
            try:
                val = getattr(mat, attr)
                audit_lines.append(f"  {attr:<30s}: {val}")
            except Exception:
                pass
    audit_lines.append("")

audit_path = "output/material_audit.txt"
with open(audit_path, "w") as f:
    f.write("\n".join(audit_lines))
ok(f"material_audit.txt salvo: {audit_path}")

# Imprime na stdout também
for l in audit_lines:
    print(l)

# ══════════════════════════════════════════════════════════════════
# 3. DIAGNÓSTICO DOS 5 UEs REFRACTED
# ══════════════════════════════════════════════════════════════════
sec("3. Diagnóstico dos 5 UEs classificados como 'refracted'")

meas = np.load("output/measurements.npz", allow_pickle=True)
path_type = meas["path_type"]
pos_east  = meas["pos_east"]
pos_north = meas["pos_north"]
pos_up    = meas["pos_up"]
rsrp_dbm  = meas["rsrp_dbm"]

refracted_idx = np.where(path_type == "refracted")[0]
print(f"Total UEs refracted no measurements.npz: {len(refracted_idx)}")

sample_idx = refracted_idx[:5]
print(f"Analisando UEs índices: {sample_idx.tolist()}\n")

solver = rt.PathSolver()

for rank, i in enumerate(sample_idx):
    pos_ue = [float(pos_east[i]), float(pos_north[i]), float(pos_up[i])]
    dist_h = math.sqrt((pos_ue[0]-pos_gnb[0])**2 + (pos_ue[1]-pos_gnb[1])**2)

    print(f"─── UE #{rank+1} (measurements idx={i}) ───────────────────────────")
    print(f"  Posição ENU  : E={pos_ue[0]:.1f} m, N={pos_ue[1]:.1f} m, Z={pos_ue[2]:.1f} m")
    print(f"  Dist. horiz. : {dist_h:.1f} m da gNB")
    print(f"  RSRP gravado : {rsrp_dbm[i]:.2f} dBm")
    print(f"  path_type    : {path_type[i]}")

    scene.add(rt.Receiver(name="rx_ue", position=pos_ue))
    try:
        paths = solver(
            scene=scene, max_depth=5,
            los=True, specular_reflection=True,
            diffraction=True, diffuse_reflection=False,
            synthetic_array=True,
        )

        valid      = np.array(paths.valid)[0, 0, :]       # (num_paths,)
        n_valid    = int(valid.sum())
        inter_all  = np.array(paths.interactions)          # (max_depth, 1, 1, num_paths)

        print(f"  num_paths    : {len(valid)} total | {n_valid} válidos")

        if n_valid == 0:
            print("  [!] Nenhum caminho válido — path_type deveria ser 'none'")
        else:
            a_r = np.array(paths.a[0]); a_i = np.array(paths.a[1])
            pow_path = (a_r**2 + a_i**2).sum(axis=tuple(range(a_r.ndim-1)))
            pow_valid = pow_path * valid.astype(float)

            print(f"\n  {'#':>3}  {'valid':>5}  {'type_seq':>30}  {'pow_rel_dB':>10}  {'phi_t°':>7}  {'theta_t°':>8}")
            phi_t_all   = np.array(paths.phi_t)[0, 0, :]
            theta_t_all = np.array(paths.theta_t)[0, 0, :]

            for p in range(len(valid)):
                if not valid[p]: continue
                inter_seq = inter_all[:, 0, 0, p]
                # converte sequência de InteractionType para nomes
                type_names = []
                for iv in inter_seq:
                    matched = [k for k,v in itype_members.items() if v == iv]
                    type_names.append(matched[0] if matched else str(iv))
                type_str = "→".join(type_names)

                pow_db = 10*math.log10(float(pow_valid[p]) + 1e-30)
                phi_deg   = math.degrees(float(phi_t_all[p]))
                theta_deg = math.degrees(float(theta_t_all[p]))
                dom = " ← DOM" if p == int(np.argmax(pow_valid)) else ""
                print(f"  {p:>3}  {'Y':>5}  {type_str:>30}  {pow_db:>10.2f}  {phi_deg:>7.1f}  {theta_deg:>8.1f}{dom}")

            # classifica o dominante
            dom_idx   = int(np.argmax(pow_valid))
            dom_inter = inter_all[:, 0, 0, dom_idx]
            if np.all(dom_inter == InteractionType.NONE):
                classify = "LoS"
            elif np.any(dom_inter == InteractionType.DIFFRACTION):
                classify = "diffracted"
            elif np.any(dom_inter == InteractionType.SPECULAR):
                classify = "reflected"
            elif np.any(dom_inter == InteractionType.REFRACTION):
                classify = "refracted"
            elif np.any(dom_inter == InteractionType.DIFFUSE):
                classify = "diffuse"
            else:
                classify = f"unknown ({dom_inter})"

            total_pwr = float(pow_valid.sum())
            rsrp_calc = 10*math.log10(cfg.get("P_TX_W", cfg.get("p_tx_w", 0.2)) * total_pwr + 1e-30) + 30
            print(f"\n  Classificação dominante  : {classify}")
            print(f"  RSRP recalculado         : {rsrp_calc:.2f} dBm")
            print(f"  Diff vs gravado          : {rsrp_calc - rsrp_dbm[i]:.3f} dB")

            if classify != path_type[i]:
                print(f"  [!] DISCREPÂNCIA: gravado='{path_type[i]}' recalculado='{classify}'")
            else:
                print(f"  [OK] Classificação consistente")

    except Exception as exc:
        print(f"  [ERRO] PathSolver: {exc}")
    finally:
        scene.remove("rx_ue")
    print()

ok("=== diagnose_refracted.py concluído ===")
