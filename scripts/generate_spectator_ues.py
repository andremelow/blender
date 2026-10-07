"""
generate_spectator_ues.py — Gera posições de UEs como espectadores do GP F1
de São Paulo no Autódromo José Carlos Pace (Interlagos).

Setores e capacidades extraídos do mapa oficial de densidade:
  /home/melo/Downloads/interlagos_density_map.svg
  Capacidade total (alvará 2025): 97.519 pessoas.

Para N=10.000 UEs, cada setor recebe n_i = round(10000 * cap_i / 97519).

Frame ENU: origem em lat=-23.7019, lon=-46.7009 (centro da cena).

Uso:
    python scripts/generate_spectator_ues.py [--n-ues N] [--seed S]
"""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ─── Constantes ───────────────────────────────────────────────────────────────
UE_HEIGHT = 1.5   # m

# ─── Setores (gerado por tools/sector_editor.html) ───────────────────────────
# Total de pessoas: 68,000  |  N_UEs = 20,000 (15k circuito + 5k cidade)
# Cada setor: (nome, capacidade_real, [(E, N), ...])
SECTORS = [
    # ── Setores do circuito (evento) ──────────────────────────────────────────
    # V  cap=5,000
    ("V", 5_000, [
        ( 198.4, -557.9),
        ( 198.4, -496.7),
        ( 257.5, -496.7),
        ( 257.5, -557.9),
    ]),

    # t  cap=5,000
    ("t", 5_000, [
        ( 434.7, -568.9),
        ( 434.7, -512.0),
        ( 541.9, -512.0),
        ( 541.9, -568.9),
    ]),

    # A  cap=20,000
    ("A", 20_000, [
        ( 563.7, -461.0),
        ( 563.7,  188.8),
        ( 716.8,  188.8),
        ( 716.8, -461.0),
    ]),

    # CGR  cap=30,000
    ("CGR", 30_000, [
        ( -18.0, -524.3),
        ( -18.0,  105.7),
        ( 200.8,  105.7),
        ( 200.8, -524.3),
    ]),

    # ── Setores da cidade (mobilidade urbana) ─────────────────────────────────
    # Setor_5  cap=1,000  (oeste do circuito)
    ("Setor_5", 1_000, [
        (-275.9, -524.1),
        (-275.9,  -42.7),
        (-100.9,  -42.7),
        (-100.9, -524.1),
    ]),

    # Setor_6  cap=1,000  (leste, parcialmente fora da cena 1200m)
    ("Setor_6", 1_000, [
        ( 896.6, -655.4),
        ( 896.6,  342.4),
        (1264.2,  342.4),
        (1264.2, -655.4),
    ]),

    # Setor_7  cap=3,000  (sul, parcialmente fora da cena)
    ("Setor_7", 3_000, [
        (-538.5, -1298.6),
        (-538.5,  -642.3),
        (1167.9,  -642.3),
        (1167.9, -1298.6),
    ]),

    # Setor_8  cap=2,000  (oeste do circuito, norte)
    ("Setor_8", 2_000, [
        (-626.0, -620.4),
        (-626.0,  920.0),
        (-289.1,  920.0),
        (-289.1, -620.4),
    ]),

    # Setor_9  cap=1,000  (norte, fora da cena)
    ("Setor_9", 1_000, [
        (-293.4,  574.3),
        (-293.4, 1401.5),
        ( 940.4, 1401.5),
        ( 940.4,  574.3),
    ]),
]

# Setores urbanos (cidade) — recebem N_CITY UEs; os demais recebem N_CIRCUIT
CITY_SECTORS = {"Setor_5", "Setor_6", "Setor_7", "Setor_8", "Setor_9"}

N_CIRCUIT = 15_000   # UEs nos setores do evento
N_CITY    =  5_000   # UEs nos setores urbanos
N_TOTAL   = N_CIRCUIT + N_CITY

TOTAL_CAP = sum(s[1] for s in SECTORS)

# ─── Traçado do circuito — 107 pontos OSM (para visualização e exclusão) ──────
# Mesmos dados de plot_spectators.py — polígono fechado, sentido anti-horário.
CIRCUIT_ENU = np.array([
    ( 54.0,  224.2), ( 31.9,  122.1), ( 29.9,   81.7), ( 34.8,   33.6),
    ( 49.0,  -32.5), ( 93.8, -205.6), (160.2, -458.1), (167.3, -472.6),
    (178.3, -489.0), (195.0, -502.3), (217.8, -504.0), (235.6, -491.1),
    (267.0, -458.7), (281.1, -453.9), (292.3, -455.9), (323.8, -474.3),
    (364.2, -490.2), (402.0, -491.9), (430.2, -487.6), (473.4, -467.2),
    (506.9, -436.1), (529.7, -395.7), (563.0, -285.3), (591.8, -183.0),
    (610.2, -105.5), (693.3,  200.2), (697.3,  221.0), (694.5,  240.2),
    (685.2,  255.6), (670.8,  265.1), (601.0,  279.2), (571.5,  283.8),
    (542.9,  283.4), (509.9,  273.3), (496.7,  265.9), (467.6,  240.6),
    (413.3,  168.7), (333.4,   58.0), (271.6,  -26.8), (255.1,  -43.4),
    (231.0,  -54.7), (204.8,  -57.0), (184.0,  -55.0), (156.2,  -46.2),
    (130.4,  -30.5), (118.2,  -17.1), (108.6,    8.6), ( 92.5,  102.9),
    ( 91.9,  122.8), ( 94.5,  138.1), (103.3,  148.0), (115.6,  153.2),
    (128.9,  152.4), (145.2,  144.0), (163.2,  130.4), (178.7,  123.1),
    (193.1,  121.5), (212.3,  127.2), (226.5,  140.3), (232.7,  152.4),
    (234.7,  165.1), (231.5,  182.0), (218.3,  200.2), (179.6,  238.3),
    (157.9,  262.9), (148.6,  277.6), (140.6,  297.8), (128.3,  349.5),
    (128.0,  363.4), (131.3,  372.5), (139.4,  380.3), (149.3,  384.8),
    (157.7,  384.9), (165.6,  381.3), (184.8,  363.3), (225.8,  318.2),
    (246.1,  298.9), (268.5,  286.3), (296.1,  279.8), (318.7,  279.5),
    (335.3,  282.5), (352.9,  289.3), (368.9,  299.5), (383.5,  313.5),
    (407.4,  348.4), (480.8,  468.2), (483.5,  473.6), (484.9,  481.3),
    (484.2,  490.0), (481.2,  496.7), (471.8,  505.8), (459.0,  512.7),
    (417.3,  529.1), (385.2,  540.7), (366.1,  545.9), (348.2,  545.1),
    (303.9,  537.7), (268.8,  530.2), (238.6,  518.9), (203.0,  502.3),
    (178.9,  490.2), (159.0,  477.9), (131.4,  454.6), (107.8,  422.1),
    ( 93.3,  385.7), ( 63.5,  264.1), ( 54.0,  224.2),
])

TRACK_HALF = 10.0   # m (metade da largura — exclusão da superfície)


def sample_in_polygon(poly, n, rng):
    """Amostragem uniforme por rejection sampling dentro de poly."""
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    while len(pts) < n:
        batch = max(4 * (n - len(pts)), 100)
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        for x, y in zip(xs, ys):
            if poly.contains(Point(x, y)):
                pts.append((x, y))
                if len(pts) == n:
                    break
    return np.array(pts[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs("output", exist_ok=True)

    print(f"[spectators] Gerando {N_TOTAL} UEs — Interlagos GP F1")
    print(f"[spectators] Circuito: {N_CIRCUIT} UEs  |  Cidade: {N_CITY} UEs")
    print(f"[spectators] Capacidade total real: {TOTAL_CAP:,}")

    # Traçado como zona de exclusão
    from shapely.geometry import LineString
    track_excl = LineString(CIRCUIT_ENU).buffer(TRACK_HALF)

    # Aloca UEs por grupo (circuito vs cidade), proporcionalmente à capacidade
    circuit_secs = [(n, c, v) for n, c, v in SECTORS if n not in CITY_SECTORS]
    city_secs    = [(n, c, v) for n, c, v in SECTORS if n in CITY_SECTORS]
    cap_circuit  = sum(c for _, c, _ in circuit_secs)
    cap_city     = sum(c for _, c, _ in city_secs)

    counts = {}
    for name, cap, _ in circuit_secs:
        counts[name] = max(1, round(N_CIRCUIT * cap / cap_circuit))
    for name, cap, _ in city_secs:
        counts[name] = max(1, round(N_CITY * cap / cap_city))
    # Ajusta para totalizar exatamente N_TOTAL
    diff = N_TOTAL - sum(counts.values())
    biggest = max(counts, key=counts.get)
    counts[biggest] += diff

    # Amostra posições
    print("[spectators] Setores:")
    all_pts  = []
    all_names= []
    zone_polys = []
    actual_counts = []

    for name, cap, verts in SECTORS:
        n_z = counts[name]
        group = "cidade" if name in CITY_SECTORS else "circuito"
        poly = Polygon(verts).difference(track_excl)
        if poly.is_empty:
            print(f"  {name}: polígono vazio — pulando")
            continue
        pts = sample_in_polygon(poly, n_z, rng)
        all_pts.append(pts)
        all_names.extend([name] * n_z)
        zone_polys.append((name, cap, poly))
        actual_counts.append(n_z)
        pct_total  = 100 * cap / TOTAL_CAP
        pct_sample = 100 * n_z / N_TOTAL
        print(f"  [{group}] {name:<12s}: cap={cap:6,} ({pct_total:4.1f}%)  →  {n_z:5d} UEs ({pct_sample:4.1f}%)")

    positions_2d = np.vstack(all_pts)
    n_total = len(positions_2d)

    positions = np.column_stack([
        positions_2d[:, 0],
        positions_2d[:, 1],
        np.full(n_total, UE_HEIGHT),
    ])

    np.savez(
        "output/ue_spectators.npz",
        positions   = positions,
        valid_mask  = np.ones(n_total, dtype=bool),
        east_vals   = positions_2d[:, 0],
        north_vals  = positions_2d[:, 1],
        zone_names  = np.array(all_names),
        n_ues       = np.int64(n_total),
        scenario    = "interlagos_f1_spectators_v3",
    )
    print(f"\n[spectators] Salvo: output/ue_spectators.npz  ({n_total} UEs)")

    # ── Visualização ───────────────────────────────────────────────────────────
    # Circuito: paleta YlOrRd (quente) por capacidade relativa ao grupo
    # Cidade:   paleta YlGnBu (frio) por capacidade relativa ao grupo
    max_circuit_cap = max(c for n, c, _ in SECTORS if n not in CITY_SECTORS)
    max_city_cap    = max(c for n, c, _ in SECTORS if n in CITY_SECTORS)

    def circuit_color(cap):
        r = cap / max_circuit_cap
        if r > 0.80: return "#800026"
        if r > 0.50: return "#e31a1c"
        if r > 0.25: return "#fd8d3c"
        return "#feb24c"

    def city_color(cap):
        r = cap / max_city_cap
        if r > 0.60: return "#253494"
        if r > 0.30: return "#2c7fb8"
        return "#7fcdbb"

    DENSITY_COLOR = {
        n: (circuit_color(c) if n not in CITY_SECTORS else city_color(c))
        for n, c, _ in SECTORS
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        # Limites automáticos cobrindo todos os setores + circuito
        all_E = [v[0] for s in SECTORS for v in s[2]] + list(CIRCUIT_ENU[:, 0])
        all_N = [v[1] for s in SECTORS for v in s[2]] + list(CIRCUIT_ENU[:, 1])
        pad = 60
        ax.set_xlim(min(all_E) - pad, max(all_E) + pad)
        ax.set_ylim(min(all_N) - pad, max(all_N) + pad)
        ax.set_aspect("equal")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#444444")
        ax.set_xlabel("Leste (m)", color="#cccccc", fontsize=10)
        ax.set_ylabel("Norte (m)", color="#cccccc", fontsize=10)
        # Traçado
        tc = CIRCUIT_ENU
        ax.fill(tc[:, 0], tc[:, 1], color="#2a2a2a", zorder=1)
        ax.plot(tc[:, 0], tc[:, 1], color="#888888", lw=2, zorder=2)
        # gNB
        ax.plot(-7.54, -2.89, "w^", ms=10, markeredgecolor="yellow",
                markeredgewidth=1.5, zorder=10)
        ax.text(-7.54 + 8, -2.89 + 8, "gNB", color="yellow", fontsize=8, zorder=11)

    # Painel esquerdo: polígonos de setor
    ax = axes[0]
    for name, cap, poly in zone_polys:
        col = DENSITY_COLOR.get(name, "#cccccc")
        if poly.geom_type == "Polygon":
            xe, ye = poly.exterior.xy
            ax.fill(xe, ye, color=col, alpha=0.75, zorder=3)
            ax.plot(xe, ye, color="white", lw=0.5, alpha=0.4, zorder=4)
        elif poly.geom_type == "MultiPolygon":
            for part in poly.geoms:
                xe, ye = part.exterior.xy
                ax.fill(xe, ye, color=col, alpha=0.75, zorder=3)
        # Label
        cx = poly.centroid.x; cy = poly.centroid.y
        short = name.split("_")[0]
        ax.text(cx, cy, short, color="white", fontsize=6.5,
                ha="center", va="center", fontweight="bold", zorder=5)

    ax.set_title("Zonas de Arquibancada\n(cor = densidade pessoas/m²)",
                 color="white", fontsize=11, fontweight="bold")

    # Painel direito: pontos UE
    ax = axes[1]
    offset = 0
    for (name, cap, poly), n_z in zip(zone_polys, actual_counts):
        col = DENSITY_COLOR.get(name, "#cccccc")
        pts = positions_2d[offset: offset + n_z]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c=col, alpha=0.5,
                   zorder=4, rasterized=True)
        offset += n_z

    ax.set_title(f"UEs amostrados por setor  (N={n_total:,}: {N_CIRCUIT:,} circuito + {N_CITY:,} cidade)",
                 color="white", fontsize=11, fontweight="bold")

    # Legenda de grupos
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#800026", label="Circuito — alta densidade"),
        Patch(facecolor="#e31a1c", label="Circuito — média-alta"),
        Patch(facecolor="#fd8d3c", label="Circuito — média"),
        Patch(facecolor="#feb24c", label="Circuito — baixa"),
        Patch(facecolor="#253494", label="Cidade — alta"),
        Patch(facecolor="#2c7fb8", label="Cidade — média"),
        Patch(facecolor="#7fcdbb", label="Cidade — baixa"),
    ]
    axes[0].legend(handles=legend_items, loc="lower left", fontsize=7.5,
                   framealpha=0.7, facecolor="#222222", edgecolor="#888888",
                   labelcolor="white")

    fig.suptitle(
        "Espectadores GP F1 São Paulo — Interlagos  "
        f"(N={n_total:,} UEs: {N_CIRCUIT:,} circuito + {N_CITY:,} cidade  |  gNB h=62 m)",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("output/ue_spectators.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("[spectators] Figura: output/ue_spectators.png")

    print(f"\nEast:  [{positions[:, 0].min():.0f}, {positions[:, 0].max():.0f}] m")
    print(f"North: [{positions[:, 1].min():.0f}, {positions[:, 1].max():.0f}] m")
    print(f"Semente: {args.seed}")
    print(f"Circuito: {sum(counts[n] for n in counts if n not in CITY_SECTORS)} UEs  |  Cidade: {sum(counts[n] for n in counts if n in CITY_SECTORS)} UEs")


if __name__ == "__main__":
    main()
