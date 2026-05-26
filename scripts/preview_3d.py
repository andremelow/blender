"""
preview_3d.py — Visualização 3D da cena com a gNB.

Uso:
    python scripts/preview_3d.py          # widget interativo (Jupyter)
    python scripts/preview_3d.py --save   # render estático → output/preview_3d.png
"""

import os
import pickle
import sys
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true",
                    help="Salva render estático em output/preview_3d.png (headless)")
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import sionna.rt as rt

XML_IN  = "output/interlagos.xml"
PKL_CFG = "output/gnb_config.pkl"

def info(msg): print(f"[INFO] {msg}", flush=True)
def ok(msg):   print(f"[ OK ] {msg}", flush=True)
def warn(msg): print(f"[WARN] {msg}", flush=True)

# ─── Carrega gNB config ───────────────────────────────────────────────────────
if os.path.exists(PKL_CFG):
    with open(PKL_CFG, "rb") as f:
        cfg = pickle.load(f)
else:
    cfg = {
        "pos_enu": [-7.54, -2.89, 25.0],
        "freq_hz": 3.5e9, "bw_hz": 100e6,
        "num_rows": 8, "num_cols": 8, "spacing": 0.5,
        "pattern": "tr38901", "polariz": "VH",
        "orientacao": [0., 0., 0.],
    }

pos_gnb = cfg["pos_enu"]
info(f"gNB ENU: E={pos_gnb[0]:.2f} m, N={pos_gnb[1]:.2f} m, Z={pos_gnb[2]:.1f} m")

# ─── Carrega cena ─────────────────────────────────────────────────────────────
info(f"Carregando cena: {XML_IN}")
scene = rt.load_scene(XML_IN)
scene.frequency = cfg["freq_hz"]
scene.bandwidth = cfg["bw_hz"]

scene.tx_array = rt.PlanarArray(
    num_rows=cfg["num_rows"], num_cols=cfg["num_cols"],
    vertical_spacing=cfg["spacing"], horizontal_spacing=cfg["spacing"],
    pattern=cfg["pattern"], polarization=cfg["polariz"],
)
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

scene.add(rt.Transmitter(name="gnb", position=pos_gnb, orientation=cfg["orientacao"]))
scene.add(rt.Receiver(name="rx_test", position=[100., 0., 1.5]))
ok("gNB e RX adicionados.")

# ─── RadioMap (necessário para overlay no preview) ────────────────────────────
info("Calculando RadioMap para overlay (cell_size=10m para ser rápido)...")
rm_solver = rt.RadioMapSolver()
rm = rm_solver(
    scene=scene,
    center=[-76., 1.5, 126.],
    orientation=[0., 0., math.pi / 2],
    size=[1880., 1860.],
    cell_size=[10., 10.],
    samples_per_tx=500_000,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    diffraction=True,
)
ok("RadioMap calculado.")

# ─── Preview 3D ───────────────────────────────────────────────────────────────
if args.save:
    out_png = "output/preview_3d.png"
    info(f"Renderizando cena (Mitsuba) → {out_png} ...")
    # Câmera isométrica levemente aérea centrada na cena
    cam = rt.Camera(position=[0., -400., 1400.], look_at=[0., 0., 0.])
    result = scene.render(
        camera=cam,
        radio_map=rm,
        rm_metric="path_gain",
        show_devices=True,
        show_orientations=True,
        resolution=(1280, 720),
        num_samples=512,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if hasattr(result, "savefig"):
        result.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(result)
    else:
        # mi.Bitmap
        import numpy as np
        arr = np.array(result)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(arr)
        ax.axis("off")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
    ok(f"Render salvo: {out_png}")
else:
    info("Abrindo preview 3D interativo (feche a janela para encerrar)...")
    info("  Controles: mouse para orbitar, scroll para zoom, clique para path picker")
    try:
        scene.preview(
            radio_map=rm,
            rm_metric="path_gain",
            show_devices=True,
            show_orientations=True,
            resolution=(1280, 720),
            fov=60.0,
        )
        ok("Preview encerrado.")
    except Exception as e:
        warn(f"preview() falhou: {e}")
        warn("Use --save para render estático em modo headless.")
