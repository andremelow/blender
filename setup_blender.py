"""
setup_blender.py — Configura os add-ons Blosm e Mitsuba-Blender no Blender headless.

Uso:
    BLOSM_ZIP=~/Downloads/blosm.zip  blender -b -P setup_blender.py
    (ou via Makefile: make setup)

Variáveis de ambiente:
    BLOSM_ZIP    — caminho do zip do Blosm (obrigatório)
    MITSUBA_ZIP  — caminho do zip do mitsuba-blender (padrão: ./addons/mitsuba-blender.zip)
"""

import bpy
import sys
import os
import subprocess
import importlib

# ─── Utilitários ──────────────────────────────────────────────────────────────
def info(msg):  print(f"[INFO]  {msg}", flush=True)
def ok(msg):    print(f"[ OK ]  {msg}", flush=True)
def erro(msg):  print(f"[ERRO]  {msg}", file=sys.stderr, flush=True); sys.exit(1)


# ─── 1. Garante pip disponível no Python interno do Blender ───────────────────
def garantir_pip():
    info("Verificando pip no Python do Blender...")
    try:
        import pip  # noqa: F401
        ok("pip já disponível.")
    except ImportError:
        import ensurepip
        ensurepip.bootstrap()
        ok("pip instalado via ensurepip.")
    return sys.executable


# ─── 2. Instala o pacote mitsuba dentro do Python do Blender ──────────────────
def instalar_mitsuba(python_bin):
    info("Instalando pacote 'mitsuba' no Python interno do Blender...")
    try:
        # mitsuba-blender 0.4.x requer mitsuba 3.5.x (WriteXML foi removido na 3.6+)
        subprocess.check_call(
            [python_bin, "-m", "pip", "install", "mitsuba==3.5.0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        ok("mitsuba instalado com sucesso.")
    except subprocess.CalledProcessError as exc:
        erro(
            f"Falha ao instalar mitsuba (código {exc.returncode}).\n"
            "Tente manualmente:\n"
            f"  {python_bin} -m pip install mitsuba==3.5.0"
        )


# ─── 3. Instala e habilita um add-on a partir de .zip ─────────────────────────
def instalar_addon(zip_path: str, module_name: str, nome_exibicao: str):
    zip_path = os.path.abspath(os.path.expanduser(zip_path))
    if not os.path.isfile(zip_path):
        erro(
            f"Arquivo não encontrado: {zip_path}\n"
            f"Verifique a variável de ambiente correspondente e tente novamente."
        )

    info(f"Instalando add-on '{nome_exibicao}' de: {zip_path}")
    bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)

    # Habilita pelo nome do módulo
    resultado = bpy.ops.preferences.addon_enable(module=module_name)
    if resultado != {'FINISHED'}:
        # Tenta descobrir o nome real do módulo inspecionando o que acabou de ser instalado
        info(f"  addon_enable('{module_name}') não finalizou — tentando detectar módulo...")
        module_name = _detectar_modulo(zip_path, module_name)
        bpy.ops.preferences.addon_enable(module=module_name)

    ok(f"Add-on '{nome_exibicao}' habilitado (módulo: {module_name}).")
    return module_name


def _detectar_modulo(zip_path: str, fallback: str) -> str:
    """Inspeciona o zip para encontrar o nome do módulo (diretório raiz ou __init__.py)."""
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        nomes = zf.namelist()
    # Primeiro diretório raiz dentro do zip é tipicamente o nome do módulo
    for nome in nomes:
        partes = nome.split("/")
        if len(partes) >= 2 and partes[0]:
            candidato = partes[0]
            info(f"  Candidato a módulo detectado: '{candidato}'")
            return candidato
    return fallback


# ─── 4. Salva preferências do Blender ─────────────────────────────────────────
def salvar_preferencias():
    bpy.ops.wm.save_userpref()
    ok("Preferências do Blender salvas.")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    blosm_zip   = os.environ.get("BLOSM_ZIP",   os.path.expanduser("~/Downloads/blosm.zip"))
    mitsuba_zip = os.environ.get("MITSUBA_ZIP",  os.path.join(script_dir, "addons", "mitsuba-blender.zip"))

    info("=== setup_blender.py ===")
    info(f"Blender Python: {sys.executable}")
    info(f"BLOSM_ZIP:      {blosm_zip}")
    info(f"MITSUBA_ZIP:    {mitsuba_zip}")
    print()

    python_bin = garantir_pip()
    print()

    instalar_mitsuba(python_bin)
    print()

    instalar_addon(blosm_zip,   "blosm",           "Blosm (OSM importer)")
    instalar_addon(mitsuba_zip, "mitsuba_blender",  "Mitsuba-Blender")
    print()

    salvar_preferencias()
    print()
    ok("=== Setup concluído! Próximo passo: make scene ===")


main()
