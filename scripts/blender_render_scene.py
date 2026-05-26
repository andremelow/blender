"""
blender_render_scene.py — Render Blender headless da cena Interlagos.

Carrega os PLYs de prédios e vias, posiciona a torre gNB e os UEs
coloridos por RSRP, configura câmera aérea e renderiza em CYCLES/EEVEE.

Uso (headless):
    blender --background --python scripts/blender_render_scene.py
"""

import bpy
import bmesh
import math
import os
import sys
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT_PNG  = os.path.join(ROOT, "output", "blender_scene.png")
RES_X, RES_Y = 1920, 1080

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ─── Limpa cena padrão ────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 2) else 'BLENDER_EEVEE'
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.filepath = OUT_PNG
scene.render.image_settings.file_format = 'PNG'
scene.eevee.taa_render_samples = 64

# ─── Leitura de PLY ───────────────────────────────────────────────────────────
def read_ply_enu(path):
    """Retorna verts ENU (N,3) e faces (M,3) de PLY binário Mitsuba."""
    with open(path, 'rb') as f:
        n_v = n_f = 0; props = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('element vertex'):  n_v = int(line.split()[-1])
            elif line.startswith('element face'):  n_f = int(line.split()[-1])
            elif line.startswith('property float') and n_f == 0:
                props.append(line.split()[-1])
            elif line == 'end_header': break
        dt = np.dtype([(p, np.float32) for p in props])
        vr = np.frombuffer(f.read(n_v * dt.itemsize), dtype=dt)
        if n_f > 0:
            dtf = np.dtype([('n', np.uint8),
                            ('i0', np.uint32), ('i1', np.uint32), ('i2', np.uint32)])
            fr = np.frombuffer(f.read(n_f * dtf.itemsize), dtype=dtf)
            faces = np.stack([fr['i0'], fr['i1'], fr['i2']], axis=1)
        else:
            faces = np.empty((0,3), dtype=np.uint32)
    # Mitsuba → Blender: x=E, y=N (=-z_mitsuba), z=Up (=y_mitsuba)
    east  =  vr['x'].astype(float)
    north = -vr['z'].astype(float)
    up    =  vr['y'].astype(float)
    return np.stack([east, north, up], axis=1), faces

def ply_to_blender(path, name, mat):
    verts, faces = read_ply_enu(path)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [(v[0], v[1], v[2]) for v in verts],
        [],
        [(int(f[0]), int(f[1]), int(f[2])) for f in faces]
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj

# ─── Materiais ────────────────────────────────────────────────────────────────
def make_mat(name, color, roughness=0.8, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat

mat_building = make_mat("building", (0.72, 0.70, 0.67), roughness=0.85)
mat_road     = make_mat("road",     (0.25, 0.25, 0.25), roughness=0.95)
mat_ground   = make_mat("ground",   (0.45, 0.50, 0.38), roughness=0.95)

# ─── Carrega malhas ───────────────────────────────────────────────────────────
info("Carregando prédios...")
ply_to_blender("output/meshes/map_2_osm_buildings.ply", "buildings", mat_building)
ok("Prédios OK")

for road_ply, mat in [
    ("output/meshes/map_2_osm_roads_primary_mesh.ply",   mat_road),
    ("output/meshes/map_2_osm_roads_secondary_mesh.ply", mat_road),
    ("output/meshes/map_2_osm_roads_residential_mesh.ply", mat_road),
]:
    if os.path.exists(road_ply):
        ply_to_blender(road_ply, os.path.basename(road_ply), mat)

# Plano de chão
bpy.ops.mesh.primitive_plane_add(size=1000, location=(0, 0, 0))
bpy.context.active_object.name = "ground"
bpy.context.active_object.data.materials.append(mat_ground)

# ─── Torre gNB ────────────────────────────────────────────────────────────────
with open("output/gnb_config.pkl", "rb") as f:
    cfg = pickle.load(f)
gnb_e, gnb_n, h_gnb = cfg['pos_enu']
info(f"gNB: E={gnb_e:.1f} N={gnb_n:.1f} H={h_gnb:.1f} m")

# Haste da torre (cilindro — raio maior para visibilidade)
bpy.ops.mesh.primitive_cylinder_add(
    radius=3.0, depth=h_gnb,
    location=(gnb_e, gnb_n, h_gnb / 2)
)
tower_obj = bpy.context.active_object
tower_obj.name = "gnb_tower"
mat_tower = make_mat("tower", (0.95, 0.95, 0.95), roughness=0.3, metallic=0.8)
tower_obj.data.materials.append(mat_tower)

# Faixas laranja ANAC (3 anéis ao longo da torre)
for frac in [0.25, 0.5, 0.75]:
    bpy.ops.mesh.primitive_cylinder_add(
        radius=3.2, depth=5.0,
        location=(gnb_e, gnb_n, h_gnb * frac)
    )
    stripe = bpy.context.active_object
    stripe.name = f"tower_stripe_{frac}"
    mat_s = make_mat(f"stripe_{frac}", (1.0, 0.45, 0.0), roughness=0.3)
    stripe.data.materials.append(mat_s)

# Antena no topo (cone vermelho)
bpy.ops.mesh.primitive_cone_add(
    radius1=6.0, radius2=1.0, depth=8,
    location=(gnb_e, gnb_n, h_gnb + 4)
)
ant_obj = bpy.context.active_object
ant_obj.name = "antenna"
mat_ant = make_mat("antenna", (0.9, 0.1, 0.1), roughness=0.2, metallic=0.6)
ant_obj.data.materials.append(mat_ant)

# ─── UEs coloridos por RSRP (grupos de cor) ───────────────────────────────────
info("Criando UEs...")
meas = np.load("output/measurements.npz", allow_pickle=True)
ue_e   = meas['pos_east']
ue_n   = meas['pos_north']
ue_u   = meas['pos_up']
rsrp   = meas['rsrp_dbm']

VMIN, VMAX = -110.0, -35.0
N_BINS = 12  # número de faixas de cor

def jet_color(t):
    t = max(0.0, min(1.0, t))
    if   t < 0.25: r, g, b = 0,       t*4,              1
    elif t < 0.50: r, g, b = 0,       1,                1-(t-0.25)*4
    elif t < 0.75: r, g, b = (t-0.5)*4, 1,              0
    else:          r, g, b = 1,       1-(t-0.75)*4,     0
    return (r, g, b)

# Agrupa UEs em N_BINS faixas de RSRP e cria um objeto por faixa
bins = np.linspace(VMIN, VMAX, N_BINS + 1)
for b in range(N_BINS):
    lo, hi = bins[b], bins[b+1]
    mask = (rsrp >= lo) & (rsrp < hi)
    if not mask.any():
        continue
    t = ((lo + hi) / 2 - VMIN) / (VMAX - VMIN)
    color = jet_color(t)

    # Cria um disco flat (quad) em cada posição UE desse bin
    bm = bmesh.new()
    for i in np.where(mask)[0]:
        cx, cy, cz = float(ue_e[i]), float(ue_n[i]), float(ue_u[i])
        r = 5.0  # raio visual em metros
        v0 = bm.verts.new((cx-r, cy-r, cz))
        v1 = bm.verts.new((cx+r, cy-r, cz))
        v2 = bm.verts.new((cx+r, cy+r, cz))
        v3 = bm.verts.new((cx-r, cy+r, cz))
        bm.faces.new([v0, v1, v2, v3])

    mesh_b = bpy.data.meshes.new(f"ue_bin_{b}")
    bm.to_mesh(mesh_b); bm.free()
    obj_b = bpy.data.objects.new(f"ue_bin_{b}", mesh_b)
    bpy.context.collection.objects.link(obj_b)

    mat_b = make_mat(f"ue_mat_{b}", color, roughness=0.0, metallic=0.0)
    mat_b.use_nodes = True
    nt = mat_b.node_tree; nodes = nt.nodes; links = nt.links
    nodes.clear()
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value    = (*color, 1.0)
    emit.inputs["Strength"].default_value = 2.5
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj_b.data.materials.append(mat_b)

ok(f"{len(ue_e)} UEs configurados em {N_BINS} faixas de RSRP")

# ─── Câmera ───────────────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 35
cam_obj = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Câmera centrada na torre, vista aérea oblíqua
cam_obj.location = (gnb_e - 50, gnb_n - 550, 350)
cam_obj.rotation_euler = (math.radians(48), 0, math.radians(-5))
cam_data.lens = 28

# ─── Iluminação ───────────────────────────────────────────────────────────────
# Sol
bpy.ops.object.light_add(type='SUN', location=(200, -300, 500))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(20))

# Ambient via world
world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value    = (0.53, 0.74, 0.95, 1.0)
bg.inputs["Strength"].default_value = 0.6

# ─── Render ───────────────────────────────────────────────────────────────────
info(f"Renderizando → {OUT_PNG}  ({RES_X}×{RES_Y})")
bpy.ops.render.render(write_still=True)
ok(f"Render salvo: {OUT_PNG}")
