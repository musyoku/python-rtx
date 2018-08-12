import three as THREE

geometry = THREE.SphereGeometry(1)
material = THREE.MeshStandardMaterial()
mesh = THREE.Mesh(geometry, material)

scene = THREE.Scene()
scene.add(mesh)

renderer = THREE.RayTracingCPURenderer()
