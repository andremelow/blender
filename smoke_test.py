"""
smoke_test.py — Valida a cena Mitsuba XML com Sionna RT 2.x (backend DrJit).

Uso:
    source ./venv/bin/activate
    python smoke_test.py [--xml output/interlagos.xml] [--max-depth 5]

Saída:
    - Número de caminhos encontrados
    - Perda de percurso estimada (dB)
    - output/preview.png com renderização da cena
"""

import argparse
import os
import sys
import math

# ─── Dependências base ────────────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # backend sem display (headless)
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"[ERRO] Dependência ausente: {exc}", file=sys.stderr)
    print("Execute: source ./venv/bin/activate && pip install numpy matplotlib")
    sys.exit(1)

# ─── Sionna RT 2.x usa DrJit, não TensorFlow ─────────────────────────────────
try:
    import drjit as dr
except ImportError:
    print("[ERRO] drjit não encontrado. Instale com: pip install 'sionna[rt]'")
    sys.exit(1)

try:
    import sionna
    from sionna.rt import (
        load_scene,
        Transmitter,
        Receiver,
        PlanarArray,
        PathSolver,
        Camera,
    )
except ImportError as exc:
    print(f"[ERRO] Sionna RT não encontrado: {exc}", file=sys.stderr)
    print("Execute: pip install sionna")
    sys.exit(1)


def info(msg):  print(f"[INFO]  {msg}", flush=True)
def ok(msg):    print(f"[ OK ]  {msg}", flush=True)
def erro(msg):  print(f"[ERRO]  {msg}", file=sys.stderr, flush=True); sys.exit(1)


# ─── Carregamento da cena ─────────────────────────────────────────────────────
def carregar_cena(xml_path: str):
    xml_path = os.path.abspath(xml_path)
    if not os.path.isfile(xml_path):
        erro(
            f"Arquivo XML não encontrado: {xml_path}\n"
            "Execute primeiro: make scene  (ou blender -b -P build_scene.py)"
        )
    info(f"Carregando cena: {xml_path}")
    cena = load_scene(xml_path)
    ok(f"Cena carregada — {len(cena.objects)} objeto(s).")
    return cena


# ─── Configuração de antenas ──────────────────────────────────────────────────
def configurar_antenas(cena):
    array_iso = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    cena.tx_array = array_iso
    cena.rx_array = array_iso
    ok("Arrays de antena configurados (isotropic, polarização V).")


# ─── Adição de TX e RX ────────────────────────────────────────────────────────
def adicionar_tx_rx(cena):
    tx = Transmitter(name="tx", position=[0.0, 0.0, 25.0])
    rx = Receiver(name="rx",    position=[100.0, 0.0, 1.5])
    cena.add(tx)
    cena.add(rx)
    info("TX adicionado em: [0, 0, 25] m")
    info("RX adicionado em: [100, 0, 1.5] m")


# ─── Cálculo de caminhos ──────────────────────────────────────────────────────
def calcular_caminhos(cena, max_depth: int = 5):
    info(f"Rodando PathSolver (max_depth={max_depth}, LOS + reflexão especular + difração)...")
    solver = PathSolver()
    caminhos = solver(
        scene=cena,
        max_depth=max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,   # desativado para velocidade no smoke test
        refraction=False,
        diffraction=True,
    )
    return caminhos


# ─── Análise dos caminhos ─────────────────────────────────────────────────────
def analisar_caminhos(caminhos) -> float:
    """Imprime métricas básicas; retorna path loss em dB."""

    # objects: [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    #      ou  [max_depth, num_rx, num_tx, num_paths]  (synthetic_array=True)
    shape = list(caminhos.objects.shape)
    num_caminhos = shape[-1] if shape else 0
    ok(f"Número de caminhos encontrados: {num_caminhos}")
    info(f"Shape de paths.objects: {shape}")

    if num_caminhos == 0:
        ok("Path Loss: infinito (nenhum caminho alcançou o receptor).")
        return float("inf")

    # Resposta impulsiva do canal — out_type='numpy' para operar com numpy
    a, tau = caminhos.cir(out_type="numpy")
    # a: complexos, shape [..., num_paths, num_time_steps]
    # tau: atrasos em segundos, shape [..., num_paths]

    # Potência total como soma dos módulos quadrados dos coeficientes
    potencia = float(np.sum(np.abs(a) ** 2))
    info(f"Potência recebida total Σ|a|²: {potencia:.6e}")

    if potencia <= 0:
        ok("Path Loss: infinito.")
        return float("inf")

    path_loss_db = -10.0 * math.log10(potencia)
    ok(f"Path Loss estimado: {path_loss_db:.2f} dB")

    # Atraso mínimo e máximo (excluindo zeros)
    tau_np = np.array(tau).flatten()
    tau_np = tau_np[tau_np > 0]
    if tau_np.size > 0:
        info(f"Atraso mínimo: {tau_np.min()*1e9:.1f} ns  |  máximo: {tau_np.max()*1e9:.1f} ns")

    # SNR simples: assume Pt = 1 W, ruído térmico N0 = -174 dBm/Hz, B = 10 MHz
    banda_hz = 10e6
    noise_w = 10 ** ((-174 - 30) / 10) * banda_hz    # ≈ 4e-14 W
    snr_db = 10.0 * math.log10(potencia / noise_w)
    ok(f"SNR estimado (Pt=1 W, B=10 MHz, N0=-174 dBm/Hz): {snr_db:.2f} dB")

    return path_loss_db


# ─── Renderização e salvamento do preview ─────────────────────────────────────
def salvar_preview(cena, caminhos, output_png: str):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    info(f"Gerando preview: {output_png}")

    try:
        # Posiciona câmera isométrica acima do centro da cena (vista de cima)
        cam = Camera(
            position=[50.0, -200.0, 150.0],
            look_at=[50.0, 0.0, 0.0],
        )
        # render() retorna plt.Figure quando return_bitmap=False (padrão)
        fig = cena.render(
            camera=cam,
            paths=caminhos,
            resolution=(1280, 720),
            show_devices=True,
            num_samples=256,
        )
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        ok(f"Preview salvo: {output_png}")

    except Exception as exc:
        info(f"Render Mitsuba falhou ({exc}) — gerando diagrama matplotlib.")
        _salvar_diagrama_fallback(caminhos, output_png)


def _salvar_diagrama_fallback(caminhos, output_png: str):
    """Diagrama 2D simples com número de caminhos."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Sionna RT — Interlagos (fallback: sem render 3D)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-50, 200)
    ax.set_ylim(-100, 100)
    ax.set_aspect("equal")

    ax.scatter([0],   [0],   s=300, c="red",  zorder=5, label="TX (0, 0, 25 m)",   marker="^")
    ax.scatter([100], [0],   s=300, c="blue", zorder=5, label="RX (100, 0, 1.5 m)", marker="v")

    shape = list(caminhos.objects.shape)
    n = shape[-1] if shape else 0
    ax.text(50, 20, f"{n} caminhos encontrados", ha="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="lightyellow"))

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    ok(f"Diagrama de fallback salvo: {output_png}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Smoke test Sionna RT 2.x — pipeline OSM")
    parser.add_argument("--xml",       default="output/interlagos.xml",
                        help="Caminho do XML Mitsuba gerado pelo build_scene.py")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Profundidade máxima de reflexões (padrão: 5)")
    parser.add_argument("--preview",   default="output/preview.png",
                        help="Caminho para salvar imagem de preview")
    args = parser.parse_args()

    print()
    info("=== smoke_test.py — Pipeline OSM → Sionna RT ===")
    info(f"Sionna:  {sionna.__version__}")
    info(f"DrJit:   {dr.__version__}")
    print()

    cena = carregar_cena(args.xml)
    print()

    configurar_antenas(cena)
    adicionar_tx_rx(cena)
    print()

    caminhos = calcular_caminhos(cena, max_depth=args.max_depth)
    print()

    analisar_caminhos(caminhos)
    print()

    salvar_preview(cena, caminhos, args.preview)
    print()

    ok("=== Smoke test concluído! ===")


if __name__ == "__main__":
    main()
