"""
inspect_scene.py — Inspeciona a geometria da cena Mitsuba/Sionna RT.

Sistema de coordenadas:
  O XML foi exportado com axis_up="Y" (Mitsuba Y-up). A transformação
  Blender → Mitsuba é: X→X (Leste), Z→Y (Altitude), Y→-Z (Norte).
  Sionna usa Z-up ENU internamente, então ao adicionar TX/RX use ENU:
    ENU.X  = Mitsuba.X  (Leste)
    ENU.Y  = -Mitsuba.Z (Norte)
    ENU.Z  = Mitsuba.Y  (Altitude)
  Para inspecionar meshes, lemos os vértices brutos (Mitsuba Y-up) e
  reportamos Altitude = eixo Y do vértice.

Uso:
    python scripts/inspect_scene.py
"""

import sys
import os
import numpy as np

# Garante que rodamos do raiz do projeto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import sionna.rt as rt

# ─── Configuração da gNB (para filtro de proximidade) ────────────────────────
LAT0     = -23.7019   # latitude da origem da cena (graus)
LON0     = -46.7009   # longitude da origem da cena (graus)
LAT_GNB  = -23.701926
LON_GNB  = -46.700974
H_GNB    = 25.0       # altura da gNB em metros
R_EARTH  = 6_378_137  # raio da Terra (m)
RAIO_M   = 50.0       # raio de busca ao redor da gNB (m)

def lat_lon_to_enu(lat, lon, lat0, lon0):
    """Converte (lat, lon) para coordenadas ENU em relação a (lat0, lon0)."""
    m_per_deg_lat = np.pi / 180 * R_EARTH
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat0))
    east  = (lon - lon0) * m_per_deg_lon
    north = (lat - lat0) * m_per_deg_lat
    return east, north

def enu_to_mitsuba(east, north, up):
    """Converte ENU → Mitsuba Y-up (inverso de blender→mitsuba)."""
    # Blender: X=E, Y=N, Z=U → Mitsuba: X=E, Y=U, Z=-N
    return np.array([east, up, -north])

def mitsuba_to_enu(mx, my, mz):
    """Converte ponto Mitsuba Y-up → ENU Z-up."""
    return np.array([mx, -mz, my])  # East, North, Up

# ─── 1. Carrega cena ──────────────────────────────────────────────────────────
XML = "output/interlagos.xml"
print(f"[INFO] Carregando cena: {XML}")
scene = rt.load_scene(XML)
print(f"[ OK ] Cena carregada.\n")

# ─── 2. Inventário ────────────────────────────────────────────────────────────
print("=" * 60)
print("INVENTÁRIO")
print("=" * 60)
print(f"  Objetos na cena: {len(scene.objects)}")

mat_count = {}
for name, obj in scene.objects.items():
    mat = obj.radio_material.name if obj.radio_material else "NONE"
    mat_count[mat] = mat_count.get(mat, 0) + 1
print("  Materiais:")
for mat, cnt in sorted(mat_count.items()):
    print(f"    {mat}: {cnt} objeto(s)")
print()

# ─── 3. Bounding box global (vértices Mitsuba Y-up → ENU) ────────────────────
print("=" * 60)
print("BOUNDING BOX GLOBAL (ENU: X=Leste, Y=Norte, Z=Altitude)")
print("=" * 60)

all_verts_enu = []
for name, obj in scene.objects.items():
    v = np.array(obj.mi_mesh.vertex_positions_buffer()).reshape(-1, 3)
    # converte coluna a coluna: Mitsuba(X,Y,Z) → ENU(X,-Z,Y)
    enu = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    all_verts_enu.append(enu)

all_enu = np.vstack(all_verts_enu)
print(f"  X (Leste)   : [{all_enu[:,0].min():.1f}, {all_enu[:,0].max():.1f}] m")
print(f"  Y (Norte)   : [{all_enu[:,1].min():.1f}, {all_enu[:,1].max():.1f}] m")
print(f"  Z (Altitude): [{all_enu[:,2].min():.1f}, {all_enu[:,2].max():.1f}] m")
print()

# ─── 4. Detalhes por objeto ───────────────────────────────────────────────────
print("=" * 60)
print("DETALHES POR OBJETO")
print("=" * 60)

objetos_info = []
for name, obj in scene.objects.items():
    v = np.array(obj.mi_mesh.vertex_positions_buffer()).reshape(-1, 3)
    enu = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])

    # Centroide ENU e altura máxima
    centro_enu = enu.mean(axis=0)
    z_min  = enu[:, 2].min()
    z_max  = enu[:, 2].max()
    altura = z_max - z_min

    mat = obj.radio_material.name if obj.radio_material else "NONE"
    faces = obj.mi_mesh.face_count()

    objetos_info.append({
        "name":    name,
        "mat":     mat,
        "centro":  centro_enu,
        "z_min":   z_min,
        "z_max":   z_max,
        "altura":  altura,
        "faces":   faces,
        "verts":   len(v),
    })

    print(f"  {name}")
    print(f"    Material   : {mat}")
    print(f"    Vértices   : {len(v):,}  |  Faces: {faces:,}")
    print(f"    Centro ENU : E={centro_enu[0]:.1f} m, N={centro_enu[1]:.1f} m, "
          f"Z={centro_enu[2]:.1f} m")
    print(f"    Z (alt)    : [{z_min:.1f}, {z_max:.1f}] m  (altura ~{altura:.1f} m)")
    print()

# ─── 5. 10 objetos mais altos ────────────────────────────────────────────────
print("=" * 60)
print("10 OBJETOS MAIS ALTOS (por Z_max)")
print("=" * 60)
top10 = sorted(objetos_info, key=lambda o: o["z_max"], reverse=True)[:10]
for i, o in enumerate(top10, 1):
    c = o["centro"]
    print(f"  {i:2d}. {o['name']:<20s}  centro=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) m"
          f"  Z_max={o['z_max']:.1f} m  altura≈{o['altura']:.1f} m")
print()

# ─── 6. Posição da gNB e objetos próximos ────────────────────────────────────
east_gnb, north_gnb = lat_lon_to_enu(LAT_GNB, LON_GNB, LAT0, LON0)
print("=" * 60)
print("POSIÇÃO DA gNB")
print("=" * 60)
print(f"  GPS         : lat={LAT_GNB}, lon={LON_GNB}")
print(f"  Origem cena : lat={LAT0}, lon={LON0}")
print(f"  ENU local   : E={east_gnb:.2f} m, N={north_gnb:.2f} m, Z={H_GNB:.1f} m")
print()

print(f"Objetos num raio de {RAIO_M:.0f} m (horizontal) da gNB:")
proximos = []
for o in objetos_info:
    c = o["centro"]
    dist_h = np.sqrt((c[0] - east_gnb)**2 + (c[1] - north_gnb)**2)
    if dist_h <= RAIO_M:
        proximos.append((dist_h, o))

if proximos:
    proximos.sort(key=lambda x: x[0])
    for dist_h, o in proximos:
        c = o["centro"]
        print(f"  {o['name']:<20s}  dist_h={dist_h:.1f} m  "
              f"centro=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  Z_max={o['z_max']:.1f} m")
else:
    print("  Nenhum objeto encontrado no raio especificado.")
    print(f"  (Nota: com {len(objetos_info)} objeto(s) na cena, o Blosm exportou")
    print(f"   todos os edifícios como uma única malha fundida.)")

print()
print("[ OK ] Inspeção concluída.")
