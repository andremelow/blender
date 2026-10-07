"""
plot_spectators.py — Mapa de distribuição dos 10 000 espectadores F1.

Painéis:
  A — path_type (LoS / transmitted / diffracted / reflected) sobre OSM
  B — RSRP [dBm] sobre OSM
  C — Densidade KDE 2D (heatmap) sobre OSM
  D — Histograma RSRP por path_type

Saída: output/spectators_map.png
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import contextily as ctx
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ─── Constantes geográficas ───────────────────────────────────────────────────
LAT0, LON0 = -23.7019, -46.7009
R_EARTH    = 6_378_137.0
D_EAST     = R_EARTH * math.cos(math.radians(LAT0)) * math.pi / 180.
D_NORTH    = R_EARTH * math.pi / 180.

def enu_to_latlon(east, north):
    lat = LAT0 + north / D_NORTH
    lon = LON0 + east  / D_EAST
    return lat, lon

def latlon_to_webmercator(lat, lon):
    """EPSG:3857 — metros."""
    x = math.radians(lon) * 6_378_137.
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6_378_137.
    return x, y

# ─── Traçado do circuito — coordenadas reais extraídas do OSM ────────────────
# 27 segmentos nomeados (Arquibancadas, Esse, Pinherinho, Bico de Pato,
# Mergulho, Junção, Café, Descida do Lago, Reta Oposta, Curva do Sol,
# S do Senna, Ferradura, Curva do Laranjinha, Subida dos Boxes …)
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

# ─── Carrega dados ────────────────────────────────────────────────────────────
print("Carregando output/measurements_spectators.npz ...")
d     = np.load("output/measurements_spectators.npz", allow_pickle=True)
east  = d["pos_east"]
north = d["pos_north"]
rsrp  = d["rsrp_dbm"].astype(float)
pt    = np.array(d["path_type"], dtype=str)

# Converte ENU → Web Mercator para contextily
lats, lons = enu_to_latlon(east, north)
wx = np.array([latlon_to_webmercator(la, lo)[0] for la, lo in zip(lats, lons)])
wy = np.array([latlon_to_webmercator(la, lo)[1] for la, lo in zip(lats, lons)])

# Circuito em Web Mercator
circ_lats, circ_lons = enu_to_latlon(CIRCUIT_ENU[:, 0], CIRCUIT_ENU[:, 1])
circ_wx = np.array([latlon_to_webmercator(la, lo)[0] for la, lo in zip(circ_lats, circ_lons)])
circ_wy = np.array([latlon_to_webmercator(la, lo)[1] for la, lo in zip(circ_lats, circ_lons)])

# gNB em Web Mercator
gnb_wx, gnb_wy = latlon_to_webmercator(LAT0 - 0.001926/D_NORTH * D_NORTH,
                                        LON0 - 0.000974/D_EAST  * D_EAST)
gnb_wx, gnb_wy = latlon_to_webmercator(*enu_to_latlon(-7.54, -2.89))

# Margem da figura — usa o circuito real como referência de bbox
circ_lats_arr = np.array([enu_to_latlon(e, n)[0] for e, n in CIRCUIT_ENU])
circ_lons_arr = np.array([enu_to_latlon(e, n)[1] for e, n in CIRCUIT_ENU])
margin = 0.008   # graus
lon_min = min(circ_lons_arr.min(), lons.min()) - margin
lon_max = max(circ_lons_arr.max(), lons.max()) + margin
lat_min = min(circ_lats_arr.min(), lats.min()) - margin
lat_max = max(circ_lats_arr.max(), lats.max()) + margin
wx_min, wy_min   = latlon_to_webmercator(lat_min, lon_min)
wx_max, wy_max   = latlon_to_webmercator(lat_max, lon_max)

# ─── Paleta path_type ─────────────────────────────────────────────────────────
PT_COLORS = {
    "LoS":         "#2ecc71",
    "transmitted": "#9b59b6",
    "diffracted":  "#e67e22",
    "reflected":   "#3498db",
    "none":        "#95a5a6",
}
PT_ORDER = ["LoS", "transmitted", "diffracted", "reflected", "none"]

# ─── Figura ───────────────────────────────────────────────────────────────────
print("Gerando figura ...")
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25,
                        left=0.05, right=0.97, top=0.93, bottom=0.06)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, :])

OSM_ZOOM = 15

def setup_map_ax(ax):
    ax.set_xlim(wx_min, wx_max)
    ax.set_ylim(wy_min, wy_max)
    ax.set_aspect("equal")
    try:
        ctx.add_basemap(ax, zoom=OSM_ZOOM, source=ctx.providers.OpenStreetMap.Mapnik,
                        crs="EPSG:3857", attribution_size=6)
    except Exception as e:
        print(f"  Basemap indisponível: {e}")
    ax.set_xticks([]); ax.set_yticks([])
    # Circuito
    ax.plot(circ_wx, circ_wy, "k-", lw=2.5, zorder=5, alpha=0.8)
    # gNB
    ax.plot(gnb_wx, gnb_wy, "r^", ms=10, markeredgecolor="white",
            markeredgewidth=1.2, zorder=10)

# ─── Painel A — path_type ─────────────────────────────────────────────────────
setup_map_ax(axA)
for ptype in PT_ORDER:
    mask = pt == ptype
    if not mask.any():
        continue
    axA.scatter(wx[mask], wy[mask], s=2, c=PT_COLORS[ptype],
                alpha=0.6, zorder=4, rasterized=True)
legend_els = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=PT_COLORS[p], markersize=7,
                     label=f"{p}  ({(pt==p).sum():,})")
              for p in PT_ORDER if (pt == p).any()]
axA.legend(handles=legend_els, loc="upper left", fontsize=7.5,
           framealpha=0.85, edgecolor="#aaa")
axA.set_title("A — Tipo de caminho dominante", fontsize=11, fontweight="bold")

# ─── Painel B — RSRP ──────────────────────────────────────────────────────────
setup_map_ax(axB)
vmin, vmax = np.percentile(rsrp, [2, 98])
sc = axB.scatter(wx, wy, s=2, c=rsrp, cmap="RdYlGn",
                 vmin=vmin, vmax=vmax, alpha=0.7, zorder=4, rasterized=True)
cb = plt.colorbar(sc, ax=axB, fraction=0.03, pad=0.02)
cb.set_label("RSRP (dBm)", fontsize=9)
cb.ax.tick_params(labelsize=8)
axB.set_title("B — RSRP por UE", fontsize=11, fontweight="bold")

# ─── Painel C — Densidade KDE ─────────────────────────────────────────────────
setup_map_ax(axC)
# KDE em coordenadas ENU (mais rápido), depois mapeia para Web Mercator grid
print("  Calculando KDE 2D ...")
kde  = gaussian_kde(np.vstack([east, north]), bw_method=0.03)
# Grid de avaliação em ENU
e_grid = np.linspace(east.min() - 20, east.max() + 20, 300)
n_grid = np.linspace(north.min() - 20, north.max() + 20, 300)
EG, NG = np.meshgrid(e_grid, n_grid)
Z = kde(np.vstack([EG.ravel(), NG.ravel()])).reshape(EG.shape)
# Converte grid para Web Mercator
lats_g, lons_g = enu_to_latlon(e_grid, n_grid)
wxg = np.array([latlon_to_webmercator(la, lo)[0]
                for la, lo in zip(lats_g, lons_g)])
wyg = np.array([latlon_to_webmercator(la, lo)[1]
                for la, lo in zip(enu_to_latlon(e_grid, n_grid[0] * np.ones_like(e_grid))[0],
                                  lons_g)])
# Usa extent aproximado em Web Mercator para o pcolormesh
wxg_min, wyg_min = latlon_to_webmercator(float(lats_g.min()), float(lons_g.min()))
wxg_max, wyg_max = latlon_to_webmercator(float(lats_g.max()), float(lons_g.max()))
pcm = axC.pcolormesh(
    np.linspace(wxg_min, wxg_max, 300),
    np.linspace(wyg_min, wyg_max, 300),
    Z,
    cmap="hot_r", alpha=0.65, zorder=3, shading="auto",
)
cb2 = plt.colorbar(pcm, ax=axC, fraction=0.03, pad=0.02)
cb2.set_label("Densidade KDE", fontsize=9)
cb2.ax.tick_params(labelsize=8)
axC.set_title("C — Densidade de espectadores (KDE)", fontsize=11, fontweight="bold")

# ─── Painel D — Histograma RSRP por path_type ─────────────────────────────────
bins = np.linspace(rsrp.min() - 1, rsrp.max() + 1, 50)
for ptype in PT_ORDER:
    mask = pt == ptype
    if not mask.any():
        continue
    axD.hist(rsrp[mask], bins=bins, alpha=0.6, color=PT_COLORS[ptype],
             label=f"{ptype} (n={mask.sum():,})", density=True, histtype="stepfilled")
    axD.hist(rsrp[mask], bins=bins, alpha=0.9, color=PT_COLORS[ptype],
             density=True, histtype="step", lw=1.5)

axD.axvline(np.median(rsrp), color="black", ls="--", lw=1.5,
            label=f"mediana: {np.median(rsrp):.1f} dBm")
axD.set_xlabel("RSRP (dBm)", fontsize=11)
axD.set_ylabel("Densidade de probabilidade", fontsize=11)
axD.set_title("D — Distribuição de RSRP por tipo de caminho", fontsize=11, fontweight="bold")
axD.legend(fontsize=9, framealpha=0.85)
axD.grid(True, alpha=0.3)

fig.suptitle(
    "Distribuição de espectadores F1 — Interlagos 3.5 GHz  "
    f"(N={len(rsrp):,} UEs  |  gNB h=62 m  |  Sionna RT)",
    fontsize=13, fontweight="bold",
)

out = "output/spectators_map.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Salvo: {out}")
