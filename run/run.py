import math
import numpy as np
import three as THREE
import matplotlib.pyplot as plt

scene = THREE.Scene()

shift = [-1, 0, 1]
for n in range(27):
    geometry = THREE.SphereGeometry(0.5)
    material = THREE.MeshStandardMaterial()
    sphere = THREE.Mesh(geometry, material)
    sphere.set_position((shift[n % 3], shift[(n // 3) % 3], shift[n // 9]))
    scene.add(sphere)

geometry = THREE.SphereGeometry(100)
material = THREE.MeshStandardMaterial()
base = THREE.Mesh(geometry, material)
base.set_position((0, -101.5, -1))
scene.add(base)

screen_width = 128
screen_height = 128

render_options = THREE.RayTracingOptions()
render_options.num_rays_per_pixel = 1
render_options.path_depth = 2

renderer = THREE.RayTracingCPURenderer()
camera = THREE.PerspectiveCamera(
    eye=(0, 0, 2),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 2,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")

camera_rad = 0
radius = 2
while True:
    eye = (radius * math.sin(camera_rad), 1.0, radius * math.cos(camera_rad))
    camera.look_at(eye=eye, center=(0, 0, 0), up=(0, 1, 0))

    renderer.render(scene, camera, render_options, buffer)
    # linear -> sRGB
    pixels = np.power(buffer, 1.0 / 2.2)
    # [0, 1] -> [0, 255]
    pixels = (pixels * 255).astype("uint8")
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1.0 / 60.0)

    # camera_rad += math.pi / 10