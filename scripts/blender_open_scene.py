"""
blender_open_scene.py — Abre cena Interlagos no Blender GUI para visualização interativa.

Uso:
    DISPLAY=:1 ~/tools/blender-3.6/blender --python scripts/blender_open_scene.py
"""

import bpy, bmesh, math, os, pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

# ─── Limpa cena ───────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'

# ─── PLY helper ───────────────────────────────────────────────────────────────
def read_ply_enu(path):
    with open(path, 'rb') as f:
        n_v = n_f = 0; props = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('element vertex'):  n_v = int(line.split()[-1])
            elif line.startswith('element face'):   n_f = int(line.split()[-1])
            elif line.startswith('property float') and n_f == 0:
                props.append(line.split()[-1])
            elif line == 'end_header': break
        dt = np.dtype([(p, np.float32) for p in props])
        vr = np.frombuffer(f.read(n_v * dt.itemsize), dtype=dt)
        if n_f > 0:
            dtf = np.dtype([('n','u1'),('i0','u4'),('i1','u4'),('i2','u4')])
            fr  = np.frombuffer(f.read(n_f * dtf.itemsize), dtype=dtf)
            faces = np.stack([fr['i0'], fr['i1'], fr['i2']], axis=1)
        else:
            faces = np.empty((0,3), dtype='u4')
    east  =  vr['x'].astype(float)
    north = -vr['z'].astype(float)
    up    =  vr['y'].astype(float)
    return np.stack([east, north, up], axis=1), faces

def make_mat(name, color, roughness=0.8, metallic=0.0, emission=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value  = roughness
    bsdf.inputs["Metallic"].default_value   = metallic
    if emission:
        bsdf.inputs["Emission"].default_value   = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 2.0
    return mat

def ply_to_obj(path, name, mat):
    verts, faces = read_ply_enu(path)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(v[0],v[1],v[2]) for v in verts], [],
                     [(int(f[0]),int(f[1]),int(f[2])) for f in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj

# ─── Materiais base ───────────────────────────────────────────────────────────
mat_bld  = make_mat("building",  (0.72, 0.70, 0.67))
mat_road = make_mat("road",      (0.20, 0.20, 0.20), roughness=0.9)
mat_gnd  = make_mat("ground",    (0.40, 0.48, 0.35), roughness=0.95)

# ─── Malhas OSM ───────────────────────────────────────────────────────────────
info("Carregando prédios...")
ply_to_obj("output/meshes/map_2_osm_buildings.ply", "buildings", mat_bld)
for ply, mat in [
    ("output/meshes/map_2_osm_roads_primary_mesh.ply",     mat_road),
    ("output/meshes/map_2_osm_roads_secondary_mesh.ply",   mat_road),
    ("output/meshes/map_2_osm_roads_residential_mesh.ply", mat_road),
]:
    if os.path.exists(ply):
        ply_to_obj(ply, os.path.basename(ply), mat)

bpy.ops.mesh.primitive_plane_add(size=1100, location=(0,0,0))
bpy.context.active_object.name = "ground"
bpy.context.active_object.data.materials.append(mat_gnd)
ok("Malhas OK")

# ─── Torre gNB ────────────────────────────────────────────────────────────────
with open("output/gnb_config.pkl","rb") as f:
    cfg = pickle.load(f)
gnb_e, gnb_n, h_gnb = cfg['pos_enu']
info(f"gNB: E={gnb_e:.1f} N={gnb_n:.1f} H={h_gnb:.1f} m")

bpy.ops.mesh.primitive_cylinder_add(radius=3.0, depth=h_gnb,
    location=(gnb_e, gnb_n, h_gnb/2))
bpy.context.active_object.name = "gnb_tower"
bpy.context.active_object.data.materials.append(
    make_mat("tower", (0.95,0.95,0.95), roughness=0.3, metallic=0.8))

for frac in [0.25, 0.5, 0.75]:
    bpy.ops.mesh.primitive_cylinder_add(radius=3.3, depth=6,
        location=(gnb_e, gnb_n, h_gnb*frac))
    bpy.context.active_object.data.materials.append(
        make_mat(f"stripe{frac}", (1.0,0.45,0.0), roughness=0.3))

bpy.ops.mesh.primitive_cone_add(radius1=7, radius2=1, depth=10,
    location=(gnb_e, gnb_n, h_gnb+5))
bpy.context.active_object.name = "antenna"
bpy.context.active_object.data.materials.append(
    make_mat("antenna", (0.9,0.1,0.1), roughness=0.2, metallic=0.6))
ok("Torre gNB criada")

# ─── UEs por faixa de RSRP ────────────────────────────────────────────────────
info("Criando UEs...")
meas  = np.load("output/measurements.npz", allow_pickle=True)
ue_e  = meas['pos_east'];  ue_n = meas['pos_north']
ue_u  = meas['pos_up'];    rsrp = meas['rsrp_dbm']

VMIN, VMAX, N_BINS = -110.0, -35.0, 12

def jet(t):
    t = max(0.0, min(1.0, t))
    if   t < 0.25: return (0,       t*4,         1)
    elif t < 0.50: return (0,       1,            1-(t-0.25)*4)
    elif t < 0.75: return ((t-0.5)*4, 1,          0)
    else:          return (1,       1-(t-0.75)*4, 0)

bins = np.linspace(VMIN, VMAX, N_BINS+1)
for b in range(N_BINS):
    mask = (rsrp >= bins[b]) & (rsrp < bins[b+1])
    if not mask.any(): continue
    t     = ((bins[b]+bins[b+1])/2 - VMIN) / (VMAX - VMIN)
    color = jet(t)
    bm = bmesh.new()
    for i in np.where(mask)[0]:
        cx, cy, cz = float(ue_e[i]), float(ue_n[i]), float(ue_u[i])
        r = 5.0
        v0=bm.verts.new((cx-r,cy-r,cz)); v1=bm.verts.new((cx+r,cy-r,cz))
        v2=bm.verts.new((cx+r,cy+r,cz)); v3=bm.verts.new((cx-r,cy+r,cz))
        bm.faces.new([v0,v1,v2,v3])
    mesh_b = bpy.data.meshes.new(f"ue_{b}")
    bm.to_mesh(mesh_b); bm.free()
    obj_b = bpy.data.objects.new(f"UE_RSRP_{b}", mesh_b)
    bpy.context.collection.objects.link(obj_b)
    mat_b = make_mat(f"ue_mat_{b}", color, roughness=0.0, emission=color)
    obj_b.data.materials.append(mat_b)

ok(f"{len(ue_e)} UEs criados em {N_BINS} faixas de RSRP")

# ─── Câmera ───────────────────────────────────────────────────────────────────
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 28
cam_obj  = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (gnb_e - 50, gnb_n - 550, 350)
cam_obj.rotation_euler = (math.radians(48), 0, math.radians(-5))

# ─── Sol ──────────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type='SUN', location=(200,-300,500))
sun = bpy.context.active_object
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(20))

world = bpy.data.worlds.new("world"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value    = (0.53,0.74,0.95,1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

# ─── Viewport shading ─────────────────────────────────────────────────────────
# Força Material Preview na viewport ativa
for area in bpy.context.screen.areas if bpy.context.screen else []:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

ok("=== Cena pronta — interaja no Blender! ===")
ok("Dicas: numpad 0 = câmera | numpad 5 = ortho/persp | scroll = zoom | MMB = orbitar")
