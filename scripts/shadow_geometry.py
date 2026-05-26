"""
shadow_geometry.py — Sobreposição de UEs bad-regime com footprints dos edifícios.

Carrega os meshes PLY da cena (coordenadas Mitsuba: x=East, y=Up, z=-North),
projeta na planta (East-North), traça linhas gNB→UE e verifica obstrução.

Obstrução: o segmento gNB→UE intersecta o convex-hull 2D de algum edifício?

Saída: output/shadow_geometry.png
"""

import os, struct, argparse, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--npz",     default="output/measurements_v2.npz")
parser.add_argument("--meshdir", default="output/meshes")
parser.add_argument("--out",     default="output/shadow_geometry.png")
args = parser.parse_args()

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)

COLORS   = {"reflected": "#3498db", "diffracted": "#e67e22"}
BAD_CATS = ["reflected", "diffracted"]

# ─── Leitor PLY binário (little-endian float32 vertices) ─────────────────────
def ler_ply_vertices(path):
    """
    Lê vértices de arquivo PLY binary_little_endian.
    Retorna array (N, 3) com colunas [x, y, z] no sistema Mitsuba.

    Conta apenas as propriedades do elemento "vertex" (não das faces).
    O cabeçalho de face tem "property list ..." que NÃO é float32, então
    contar todas as linhas "property" causaria stride errado e coordenadas lixo.
    """
    with open(path, "rb") as f:
        raw = f.read()

    header_end = raw.find(b"end_header\n")
    if header_end == -1:
        return None
    header     = raw[:header_end].decode("ascii", errors="ignore")
    data_start = header_end + len("end_header\n")

    n_verts          = 0
    n_vertex_props   = 0
    in_vertex_block  = False

    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts         = int(line.split()[-1])
            in_vertex_block = True
        elif line.startswith("element "):   # outro elemento: sai do bloco vertex
            in_vertex_block = False
        elif line.startswith("property ") and in_vertex_block:
            n_vertex_props += 1             # conta apenas propriedades de vértice

    if n_verts == 0 or n_vertex_props < 3:
        return None

    # Lê n_verts × n_vertex_props float32 — stride correto, sem invadir dados de face
    verts = np.frombuffer(raw, dtype="<f4",
                          count=n_verts * n_vertex_props,
                          offset=data_start)
    return verts.reshape(n_verts, n_vertex_props)[:, :3].copy()

# ─── Carrega triângulos 2D dos edifícios (apenas buildings.ply) ───────────────
info("Carregando mesh de edifícios...")
BUILDINGS_PLY = os.path.join(args.meshdir, "map_2_osm_buildings.ply")

def ler_ply_completo(path):
    """Lê vértices e faces de PLY binary_little_endian. Retorna (verts, faces)."""
    with open(path, "rb") as f:
        raw = f.read()
    header_end = raw.find(b"end_header\n")
    header     = raw[:header_end].decode("ascii", errors="ignore")
    data_start = header_end + len("end_header\n")

    n_verts = 0; n_vertex_props = 0
    n_faces = 0
    in_vertex = False

    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1]); in_vertex = True
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1]); in_vertex = False
        elif line.startswith("element "):
            in_vertex = False
        elif line.startswith("property ") and in_vertex:
            n_vertex_props += 1

    verts_bytes = n_verts * n_vertex_props * 4
    verts = np.frombuffer(raw, dtype="<f4", count=n_verts * n_vertex_props,
                          offset=data_start).reshape(n_verts, n_vertex_props)[:, :3].copy()

    # Lê faces: cada face = 1 byte (n_verts da face) + n_verts * 4 bytes (int32)
    faces    = []
    offset   = data_start + verts_bytes
    for _ in range(n_faces):
        n_v  = raw[offset]; offset += 1
        idxs = struct.unpack_from(f"<{n_v}i", raw, offset); offset += n_v * 4
        faces.append(list(idxs))
    return verts, faces

verts_mit, faces = ler_ply_completo(BUILDINGS_PLY)
# Converte Mitsuba → East-North: East=x, North=-z
verts_en = np.column_stack([verts_mit[:, 0], -verts_mit[:, 2]])
info(f"  Edifícios: {len(verts_mit)} vértices, {len(faces)} faces")
info(f"  East range: [{verts_en[:,0].min():.1f}, {verts_en[:,0].max():.1f}]")
info(f"  North range: [{verts_en[:,1].min():.1f}, {verts_en[:,1].max():.1f}]")

# Footprints: convex hull de cada componente conexa via faces
# Simplificação: uma face = polígono (triângulo). Agrupa vértices adjacentes por seed.
# Para visualização, usamos o convex hull do conjunto completo de vértices por cluster.
# Como o PLY agrupa todos os prédios numa mesh, separamos por componente conexa (union-find).
parent = list(range(len(verts_mit)))

def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for face in faces:
    for i in range(1, len(face)):
        union(face[0], face[i])

# Agrupa vértices por componente
from collections import defaultdict
components = defaultdict(list)
for i in range(len(verts_en)):
    components[find(i)].append(i)

footprints = []
for root, idxs in components.items():
    pts = verts_en[idxs]
    if len(pts) < 3: continue
    try:
        hull = ConvexHull(pts)
        footprints.append(pts[hull.vertices])
    except Exception:
        continue

info(f"  {len(footprints)} componentes conexas (edifícios) extraídas")

# ─── Carrega dados de UEs ─────────────────────────────────────────────────────
d      = np.load(args.npz, allow_pickle=True)
pos_e  = d["pos_east"]; pos_n  = d["pos_north"]
ptype  = d["path_type"]
gnb    = d["gnb_pos"]
n_ues  = len(pos_e)

# ─── Funções de interseção segmento-polígono convexo ─────────────────────────
def cross2d(o, a, b):
    """Produto vetorial 2D de (a-o) × (b-o)."""
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def segmentos_cruzam(p1, p2, p3, p4):
    """Retorna True se o segmento p1→p2 cruza p3→p4."""
    d1 = cross2d(p3, p4, p1)
    d2 = cross2d(p3, p4, p2)
    d3 = cross2d(p1, p2, p3)
    d4 = cross2d(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def ponto_no_poligono(ponto, poligono):
    """Ray-casting 2D: True se o ponto está dentro do polígono."""
    x, y  = ponto
    n     = len(poligono)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poligono[i]
        xj, yj = poligono[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
            inside = not inside
        j = i
    return inside

def segmento_intersecta_footprint(a, b, footprint):
    """
    True se o segmento a→b intersecta o convex-hull footprint.
    Verifica: (1) algum ponto dentro, (2) alguma aresta cruzada.
    """
    poly = footprint
    n    = len(poly)
    if ponto_no_poligono(a, poly) or ponto_no_poligono(b, poly):
        return True
    for i in range(n):
        j = (i + 1) % n
        if segmentos_cruzam(a, b, poly[i], poly[j]):
            return True
    return False

# ─── Verifica obstrução para cada bad UE ─────────────────────────────────────
gnb2d   = np.array([gnb[0], gnb[1]])
bad_mask = np.isin(ptype, BAD_CATS)
bad_idx  = np.where(bad_mask)[0]

n_obstruidos = 0
obstruido    = np.zeros(n_ues, dtype=bool)

info(f"Verificando obstrução para {len(bad_idx)} UEs bad-regime...")
for i in bad_idx:
    ue2d = np.array([pos_e[i], pos_n[i]])
    for fp in footprints:
        if segmento_intersecta_footprint(gnb2d, ue2d, fp):
            obstruido[i] = True
            break

n_obstruidos = int(obstruido[bad_mask].sum())
pct = 100 * n_obstruidos / max(len(bad_idx), 1)
info(f"  {n_obstruidos}/{len(bad_idx)} bad UEs com edifício no segmento gNB→UE ({pct:.1f}%)")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
freq_ghz = float(d["freq_hz"]) / 1e9

# Footprints dos edifícios
for fp in footprints:
    poly = plt.Polygon(fp, closed=True, fc="#cccccc", ec="#888888",
                       lw=0.5, alpha=0.7)
    ax.add_patch(poly)

# Todos os UEs (fundo)
ax.scatter(pos_e, pos_n, color="lightgray", s=8, alpha=0.4, zorder=1)

# Linhas gNB → bad UE (tracejadas)
for t in BAD_CATS:
    mask = (ptype == t)
    for i in np.where(mask)[0]:
        ax.plot([gnb[0], pos_e[i]], [gnb[1], pos_n[i]],
                color=COLORS[t], lw=0.6, alpha=0.35, ls="--", zorder=2)

# Bad UEs sobrepostos
for t in BAD_CATS:
    mask = ptype == t
    if not mask.any(): continue
    ax.scatter(pos_e[mask], pos_n[mask], color=COLORS[t], s=30,
               alpha=0.95, zorder=4, label=f"{t} (n={mask.sum()})")
    # Marca os não obstruídos com X
    mask_no = mask & ~obstruido
    if mask_no.any():
        ax.scatter(pos_e[mask_no], pos_n[mask_no], marker="x",
                   color=COLORS[t], s=50, lw=1.5, zorder=5,
                   label=f"{t} sem bloqueio (n={mask_no.sum()})")

# gNB
ax.scatter(*gnb[:2], marker="^", c="red", s=180, zorder=6, label="gNB")

ax.set_xlabel("Leste (m)"); ax.set_ylabel("Norte (m)")
ax.set_title(
    f"(L) Geometria de Sombra — {freq_ghz:.1f} GHz | "
    f"{n_obstruidos}/{len(bad_idx)} bad UEs com bloqueio ({pct:.1f}%)"
)

patches = [mpatches.Patch(color=COLORS[t], label=t) for t in BAD_CATS]
patches += [mpatches.Patch(color="#cccccc", label="edifícios"),
            mpatches.Patch(color="red",     label="gNB")]
ax.legend(handles=patches, fontsize=9, loc="lower right")
ax.set_aspect("equal"); ax.autoscale(); ax.grid(True, alpha=0.2)

print()
ok(f"Resultado: {n_obstruidos}/{len(bad_idx)} bad UEs têm pelo menos um edifício "
   f"no segmento gNB→UE ({pct:.1f}%) — esperado >95%")

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close()
ok(f"Plot salvo: {args.out}")
