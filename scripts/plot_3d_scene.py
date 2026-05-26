"""
plot_3d_scene.py — Render 3D matplotlib da cena com torre gNB e UEs.

Mostra:
  - Footprint dos prédios (topo dos polígonos)
  - Torre gNB (linha vertical vermelha até 62 m)
  - UEs coloridos por RSRP [dBm]

Uso:
    python scripts/plot_3d_scene.py [--out output/scene_3d.png]
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="output/scene_3d.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ─── Leitura do PLY ───────────────────────────────────────────────────────────
def read_ply(path):
    """Retorna verts (N,3) e faces (M,3) de um PLY binário.
       Coordenadas Mitsuba: x=East, y=Up, z=-(North) → converte para ENU.
    """
    with open(path, 'rb') as f:
        n_verts = n_faces = 0
        props = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
            elif line.startswith('property float') and n_faces == 0:
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        dt_v = np.dtype([(p, np.float32) for p in props])
        verts_raw = np.frombuffer(f.read(n_verts * dt_v.itemsize), dtype=dt_v)
        # faces: 1 byte (n=3) + 3× uint32
        if n_faces > 0:
            dt_f = np.dtype([('n', np.uint8), ('i0', np.uint32), ('i1', np.uint32), ('i2', np.uint32)])
            faces_raw = np.frombuffer(f.read(n_faces * dt_f.itemsize), dtype=dt_f)
            faces = np.stack([faces_raw['i0'], faces_raw['i1'], faces_raw['i2']], axis=1)
        else:
            faces = np.empty((0, 3), dtype=np.uint32)

    # Mitsuba → ENU: East=x, North=-z, Up=y
    east  =  verts_raw['x'].astype(float)
    north = -verts_raw['z'].astype(float)
    up    =  verts_raw['y'].astype(float)
    verts = np.stack([east, north, up], axis=1)
    return verts, faces

# ─── Carrega prédios ──────────────────────────────────────────────────────────
info("Carregando prédios...")
bld_verts, bld_faces = read_ply("output/meshes/map_2_osm_buildings.ply")
ok(f"Prédios: {len(bld_verts)} vértices, {len(bld_faces)} faces")

# ─── Carrega medidas ──────────────────────────────────────────────────────────
info("Carregando medidas...")
meas = np.load("output/measurements.npz", allow_pickle=True)
ue_e = meas['pos_east']
ue_n = meas['pos_north']
ue_u = meas['pos_up']
rsrp = meas['rsrp_dbm']
gnb_pos = meas['gnb_pos']

# ─── gNB config ───────────────────────────────────────────────────────────────
with open("output/gnb_config.pkl", "rb") as f:
    cfg = pickle.load(f)
h_gnb = cfg['h_gnb']
gnb_e, gnb_n = gnb_pos[0], gnb_pos[1]

ok(f"gNB: E={gnb_e:.1f} m, N={gnb_n:.1f} m, H={h_gnb:.1f} m AGL")
ok(f"UEs: {len(ue_e)} | RSRP [{rsrp.min():.1f}, {rsrp.max():.1f}] dBm")

# ─── Figura ───────────────────────────────────────────────────────────────────
info("Gerando figura 3D...")
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Prédios — faces dos telhados (faces com up_médio > 1 m)
roof_polys = []
roof_heights = []
for tri in bld_faces:
    pts = bld_verts[tri]          # (3, 3) ENU
    z_mean = pts[:, 2].mean()
    if z_mean > 0.5:              # só telhados
        roof_polys.append(pts)
        roof_heights.append(z_mean)

if roof_polys:
    roof_cmap = plt.cm.Greys
    z_norm = np.array(roof_heights)
    z_norm = (z_norm - z_norm.min()) / (z_norm.max() - z_norm.min() + 1e-9)
    roof_colors = [(*roof_cmap(0.3 + 0.4 * z)[:3], 0.7) for z in z_norm]
    poly_col = Poly3DCollection(
        [p[:, [0, 1, 2]] for p in roof_polys],
        facecolors=roof_colors, edgecolors='none', linewidths=0,
    )
    ax.add_collection3d(poly_col)
    ok(f"Telhados renderizados: {len(roof_polys)} polígonos")

# Paredes (faces com z misto — pelo menos um vértice no chão)
wall_polys = []
for tri in bld_faces:
    pts = bld_verts[tri]
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    if zmax > 0.5 and zmin < 0.5:
        wall_polys.append(pts)

if wall_polys:
    wall_col = Poly3DCollection(
        [p[:, [0, 1, 2]] for p in wall_polys],
        facecolors=(0.75, 0.73, 0.70, 0.5), edgecolors='none',
    )
    ax.add_collection3d(wall_col)

# UEs — scatter colorido por RSRP
rsrp_vmin, rsrp_vmax = -110, -35
sc = ax.scatter(
    ue_e, ue_n, ue_u,
    c=rsrp, cmap='jet', vmin=rsrp_vmin, vmax=rsrp_vmax,
    s=18, alpha=0.85, zorder=5, depthshade=True,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.55, aspect=20)
cbar.set_label("RSRP (dBm)", fontsize=11)

# Torre gNB — linha vertical + marcador no topo
ax.plot([gnb_e, gnb_e], [gnb_n, gnb_n], [0, h_gnb],
        color='red', linewidth=3, zorder=10, label=f'gNB ({h_gnb:.0f} m AGL)')
ax.scatter([gnb_e], [gnb_n], [h_gnb],
           color='red', s=120, marker='^', zorder=11, depthshade=False)

# ─── Câmera e formatação ──────────────────────────────────────────────────────
ax.set_xlabel("Leste (m)", fontsize=10)
ax.set_ylabel("Norte (m)", fontsize=10)
ax.set_zlabel("Altura (m)", fontsize=10)
ax.set_title(
    f"Interlagos — Sionna RT | 3.5 GHz | {len(ue_e)} UEs | gNB {h_gnb:.0f} m AGL",
    fontsize=13, pad=12
)
ax.legend(loc='upper left', fontsize=10)
ax.view_init(elev=40, azim=45)
ax.set_zlim(0, max(h_gnb + 10, 30))

plt.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=180, bbox_inches='tight')
plt.close(fig)
ok(f"Render 3D salvo: {args.out}")
