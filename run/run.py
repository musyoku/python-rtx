import math
import numpy as np
import three as THREE
import matplotlib.pyplot as plt

scene = THREE.Scene()

metal_material = THREE.MeshMetalMaterial(
    roughness=0.8, specular_reflectance=0.8)
shift = [-1.125, 0, 1.125]
colors = [(0.25, 1.0, 1.0), (1.0, 1.0, 0.25), (1.0, 0.25, 1.0)]
for n in range(27):
    color = colors[(n + n // 3) % 3]
    geometry = THREE.SphereGeometry(0.5)
    if n % 2 == 0:
        material = THREE.MeshLambertMaterial(
            color=color, diffuse_reflectance=0.8)
    else:
        material = metal_material
    sphere = THREE.Mesh(geometry, material)
    sphere.set_position((shift[n % 3], shift[(n // 3) % 3], shift[n // 9]))
    scene.add(sphere)

geometry = THREE.SphereGeometry(100)
material = THREE.MeshLambertMaterial((1.0, 1.0, 1.0), 0.8)
base = THREE.Mesh(geometry, material)
base.set_position((0, -101.5, -1))
# scene.add(base)

# scene = THREE.Scene()
faces = np.array(
    [
        [0, 2, 1],
        [2, 3, 1],
        [0, 1, 4],
        [4, 1, 5],
        [2, 0, 4],
        [6, 2, 4],
        [7, 2, 6],
        [7, 3, 2],
        [1, 7, 5],
        [1, 3, 7],
        [4, 5, 6],
        [6, 5, 7],
    ], dtype="int32")
vertices = np.array(
    [
        [0.5, 0.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, -0.5],
        [0.5, 1.0, 0.5],
        [0.5, 1.0, -0.5],
        [-0.5, 1.0, 0.5],
        [-0.5, 1.0, -0.5],
    ],
    dtype="float32")
geometry = THREE.StandardGeometry(faces, vertices)
triangle = THREE.Mesh(geometry, material)
triangle.set_position((0, 0, 1))
triangle.set_scale((2.0, 2.0, 2.0))
scene.add(triangle)

screen_width = 256
screen_height = 256

render_options = THREE.RayTracingOptions()
render_options.num_rays_per_pixel = 64
render_options.path_depth = 4

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

camera_rad = math.pi / 5
# camera_rad = 0
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

    camera_rad += math.pi / 10
