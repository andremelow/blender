import sionna.rt as rt
import numpy as np

scene = rt.load_scene("output/interlagos.xml")
scene.frequency = 3.5e9
scene.bandwidth = 100e6

# 1) Inventário
print(f"Objetos: {len(scene.objects)}")
mats = {}
for name, obj in scene.objects.items():
    m = obj.radio_material.name if obj.radio_material else "NONE"
    mats[m] = mats.get(m, 0) + 1
print("Materiais:", mats)

# 2) Bounding box
positions = []
for name, obj in scene.objects.items():
    # cada obj tem mesh; pega o centroide aproximado
    positions.append(obj.position)   # pode ser que sua versão use .object_position
positions = np.array(positions)
print(f"X range: {positions[:,0].min():.1f} a {positions[:,0].max():.1f} m")
print(f"Y range: {positions[:,1].min():.1f} a {positions[:,1].max():.1f} m")
print(f"Z range: {positions[:,2].min():.1f} a {positions[:,2].max():.1f} m")

# 3) Teste de propagação
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")

scene.add(rt.Transmitter("tx", position=[0.,0.,25.]))
scene.add(rt.Receiver("rx",    position=[100.,0.,1.5]))

solver = rt.PathSolver()
paths = solver(scene=scene, max_depth=3)
a, tau = paths.cir(out_type="numpy")
print(f"Caminhos encontrados: {a.shape[-2]}")
print(f"Potência total: {10*np.log10(np.sum(np.abs(a)**2)):.1f} dB rel.")
print(f"Atraso 1º caminho: {tau.flatten()[0]*1e9:.2f} ns "
      f"(esperado ~{100/3e8*1e9:.2f} ns para LoS de 100 m)")
