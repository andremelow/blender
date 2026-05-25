"""
build_scene.py — Importa OSM via Blosm, aplica materiais ITU e exporta Mitsuba XML.

Uso:
    blender -b -P build_scene.py -- [min_lat max_lat min_lon max_lon]

Argumentos (após '--'):
    min_lat  max_lat  min_lon  max_lon   — bounding box; usa Interlagos se omitido

Exemplo:
    blender -b -P build_scene.py -- -23.7064 -23.6974 -46.7064 -46.6954
"""

import bpy
import sys
import os

# ─── Utilitários ──────────────────────────────────────────────────────────────
def info(msg):  print(f"[INFO]  {msg}", flush=True)
def ok(msg):    print(f"[ OK ]  {msg}", flush=True)
def erro(msg):  print(f"[ERRO]  {msg}", file=sys.stderr, flush=True); sys.exit(1)


# ─── 1. Análise de argumentos ─────────────────────────────────────────────────
def parse_args() -> dict:
    """Extrai bbox dos argumentos após '--', ou retorna Interlagos como padrão."""
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
    except ValueError:
        args = []

    if len(args) >= 4:
        try:
            return dict(
                min_lat=float(args[0]),
                max_lat=float(args[1]),
                min_lon=float(args[2]),
                max_lon=float(args[3]),
            )
        except ValueError:
            erro(f"Argumentos inválidos: {args}. Esperado: min_lat max_lat min_lon max_lon")

    info("Nenhuma bbox fornecida — usando padrão Interlagos (SP).")
    return dict(min_lat=-23.7064, max_lat=-23.6974, min_lon=-46.7064, max_lon=-46.6954)


# ─── 2. Limpeza da cena padrão ────────────────────────────────────────────────
def limpar_cena():
    info("Limpando cena padrão do Blender...")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove malhas órfãs
    for bloco in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights]:
        for item in bloco:
            bloco.remove(item)
    ok("Cena limpa.")


# ─── 3. Import OSM via Blosm ──────────────────────────────────────────────────
def importar_osm(bbox: dict):
    info(f"Importando OSM: lat [{bbox['min_lat']}, {bbox['max_lat']}], "
         f"lon [{bbox['min_lon']}, {bbox['max_lon']}]")

    # Diretório de cache para dados OSM baixados
    osm_data_dir = os.path.join(os.getcwd(), "osm_data")
    os.makedirs(osm_data_dir, exist_ok=True)

    # Configura preferências do Blosm se disponíveis
    addon_prefs = bpy.context.preferences.addons.get("blosm")
    if addon_prefs:
        addon_prefs.preferences.dataDir = osm_data_dir
    else:
        erro(
            "Add-on 'blosm' não encontrado nas preferências do Blender.\n"
            "Execute primeiro: make setup  (ou blender -b -P setup_blender.py)"
        )

    # Configura propriedades de cena do Blosm e chama o operador
    addon = bpy.context.scene.blosm
    addon.dataType   = "osm"
    addon.osmSource  = "server"
    addon.mode       = "3Dsimple"
    addon.minLat     = bbox["min_lat"]
    addon.maxLat     = bbox["max_lat"]
    addon.minLon     = bbox["min_lon"]
    addon.maxLon     = bbox["max_lon"]
    addon.buildings  = True
    addon.highways   = True
    addon.water      = True
    addon.forests    = False

    resultado = bpy.ops.blosm.import_data()

    if resultado == {"CANCELLED"}:
        erro(
            "Importação OSM cancelada pelo Blosm.\n"
            "Possíveis causas: sem conexão com a internet, bbox inválida, "
            "ou API do Overpass indisponível.\n"
            "Tente acessar manualmente: https://overpass-api.de"
        )

    n_objetos = len([o for o in bpy.data.objects if o.type == "MESH"])
    ok(f"OSM importado — {n_objetos} objetos de malha criados.")


# ─── 4. Aplicar transformações (equivalente a Ctrl+A → All Transforms) ────────
def aplicar_transforms():
    info("Aplicando todas as transformações (location, rotation, scale)...")
    import mathutils

    for obj in list(bpy.data.objects):
        # Garante visibilidade para poder operar no objeto
        obj.hide_set(False)
        obj.hide_viewport = False

        if obj.type == "CURVE":
            # Converte curva para mesh via API de dados (sem depender de operadores)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
            new_obj = bpy.data.objects.new(obj.name + "_mesh", mesh)
            bpy.context.collection.objects.link(new_obj)
            new_obj.matrix_world = obj.matrix_world.copy()
            bpy.data.objects.remove(obj, do_unlink=True)

    # Aplica transforms em todos os objetos restantes
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.data.transform(obj.matrix_world)
            obj.matrix_world = mathutils.Matrix.Identity(4)

    ok("Transformações aplicadas.")


# ─── 5. Criação de materiais ITU ──────────────────────────────────────────────
# Parâmetros de rugosidade aproximados para os modelos de material ITU-R P.2040
_MATERIAIS_ITU = {
    "itu_concrete":   {"cor": (0.72, 0.70, 0.68, 1.0), "roughness": 0.85, "ior": 1.50},
    "itu_marble":     {"cor": (0.90, 0.90, 0.92, 1.0), "roughness": 0.15, "ior": 1.60},
    "itu_wet_ground": {"cor": (0.20, 0.28, 0.35, 1.0), "roughness": 0.05, "ior": 1.33},
}


def criar_material_itu(nome: str) -> bpy.types.Material:
    """Cria (ou recupera) um material Principled BSDF com parâmetros ITU."""
    if nome in bpy.data.materials:
        return bpy.data.materials[nome]

    params = _MATERIAIS_ITU[nome]
    mat = bpy.data.materials.new(name=nome)
    mat.use_nodes = True

    arvore = mat.node_tree
    arvore.nodes.clear()

    bsdf = arvore.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value   = params["cor"]
    bsdf.inputs["Roughness"].default_value    = params["roughness"]
    bsdf.inputs["IOR"].default_value          = params["ior"]

    saida = arvore.nodes.new("ShaderNodeOutputMaterial")
    arvore.links.new(bsdf.outputs["BSDF"], saida.inputs["Surface"])

    return mat


def criar_todos_materiais() -> dict:
    info("Criando materiais ITU...")
    materiais = {nome: criar_material_itu(nome) for nome in _MATERIAIS_ITU}
    for nome in materiais:
        ok(f"  Material criado: {nome}")
    return materiais


# ─── 6. Atribuição de materiais por coleção ───────────────────────────────────
# Mapeamento: substring do nome da coleção → material ITU
_MAPA_COLECAO = {
    "building":  "itu_concrete",
    "edifici":   "itu_concrete",
    "water":     "itu_wet_ground",
    "agua":      "itu_wet_ground",
    "lago":      "itu_wet_ground",
    "highway":   "itu_marble",
    "road":      "itu_marble",
    "street":    "itu_marble",
    "via":       "itu_marble",
}


def material_para_objeto(obj: bpy.types.Object, materiais: dict) -> bpy.types.Material:
    """Escolhe material com base no nome das coleções do objeto."""
    for col in obj.users_collection:
        nome_col = col.name.lower()
        for chave, mat_nome in _MAPA_COLECAO.items():
            if chave in nome_col:
                return materiais[mat_nome]
    # Padrão: concreto (cobre a maioria dos edifícios não classificados)
    return materiais["itu_concrete"]


def atribuir_materiais(materiais: dict):
    info("Atribuindo materiais por coleção...")
    contagem = {nome: 0 for nome in materiais}

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mat = material_para_objeto(obj, materiais)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        contagem[mat.name] += 1

    for mat_nome, qtd in contagem.items():
        ok(f"  {mat_nome}: {qtd} objeto(s)")


# ─── 7. Exportação para Mitsuba XML ───────────────────────────────────────────
def exportar_mitsuba(output_path: str):
    info(f"Exportando cena para Mitsuba XML: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Seleciona todos os objetos de malha antes de exportar
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)

    # Sistema de eixos: Blender usa Z-up; Mitsuba/Sionna usam Y-up
    resultado = bpy.ops.export_scene.mitsuba(
        filepath=output_path,
        axis_forward="-Z",
        axis_up="Y",
        export_ids=True,          # inclui IDs para identificar objetos no Sionna
        ignore_background=True,
    )

    if resultado == {"CANCELLED"}:
        erro(
            "Exportação Mitsuba cancelada.\n"
            "Verifique se o add-on Mitsuba-Blender está habilitado:\n"
            "  make setup  (ou blender -b -P setup_blender.py)"
        )

    if not os.path.isfile(output_path):
        erro(f"Arquivo XML não foi gerado em: {output_path}")

    tamanho_kb = os.path.getsize(output_path) / 1024
    ok(f"Exportado: {output_path}  ({tamanho_kb:.1f} KB)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    bbox = parse_args()

    info("=== build_scene.py ===")
    info(f"BBox: lat [{bbox['min_lat']}, {bbox['max_lat']}] | "
         f"lon [{bbox['min_lon']}, {bbox['max_lon']}]")
    print()

    limpar_cena()
    print()

    importar_osm(bbox)
    print()

    aplicar_transforms()
    print()

    materiais = criar_todos_materiais()
    print()

    atribuir_materiais(materiais)
    print()

    output_xml = os.path.join(os.getcwd(), "output", "interlagos.xml")
    exportar_mitsuba(output_xml)
    print()

    ok("=== Cena construída! Próximo passo: make test ===")


main()
