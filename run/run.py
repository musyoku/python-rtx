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

renderer = THREE.RayTracingCPURenderer()
camera = THREE.PerspectiveCamera(
    eye=(3, 1, 0),
    center=(0, 0.5, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 2,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

buffer = np.zeros((screen_height, screen_width, 3), dtype="int32")

renderer.render(scene, camera, buffer)

plt.imshow(buffer, interpolation="none")
plt.pause(10)