import math
import numpy as np
import three as THREE
import matplotlib.pyplot as plt

geometry = THREE.SphereGeometry(1)
material = THREE.MeshStandardMaterial()
mesh = THREE.Mesh(geometry, material)

scene = THREE.Scene()
scene.add(mesh)

screen_width = 128
screen_height = 128

render_options = THREE.RayTracingOptions()
render_options.num_rays_per_pixel = 32

renderer = THREE.RayTracingCPURenderer()
camera = THREE.PerspectiveCamera(
    eye=(0, 0, 1),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 2,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

buffer = np.zeros((screen_height, screen_width, 3), dtype="int32")

pos = (0, 0, -1)
while True:
    mesh.set_position(pos)
    renderer.render(scene, camera, render_options, buffer)
    plt.imshow(buffer, interpolation="none")
    plt.pause(0.1)

    pos = (pos[0], pos[1], pos[2] - 0.01)
