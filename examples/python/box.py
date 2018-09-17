import math
import time
import numpy as np
import rtx
import geometry as gm
import matplotlib.pyplot as plt

scene = rtx.Scene()

box_size = 6

# 1
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((0, 0, 0))
geometry.set_position((0, 0, -box_size / 2))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 2
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_size / 2, 0, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 3
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((0, math.pi, 0))
geometry.set_position((0, 0, box_size / 2))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 4
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-box_size / 2, 0, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# ceil
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((math.pi / 2, 0, 0))
geometry.set_position((0, box_size / 2, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# floor
geometry = rtx.PlainGeometry(box_size, box_size)
geometry.set_rotation((-math.pi / 2, 0, 0))
geometry.set_position((0, -box_size / 2, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# light
geometry = rtx.PlainGeometry(box_size / 2, box_size / 2)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((0.01 - box_size / 2, -box_size / 4, 0))
material = rtx.EmissiveMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

geometry = rtx.PlainGeometry(box_size / 2, box_size / 2)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_size / 2 - 0.01, -box_size / 4, 0))
material = rtx.EmissiveMaterial(1.0)
mapping = rtx.SolidColorMapping((0, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

# place bunny
faces, vertices = gm.load("../geometries/bunny")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((0, -box_size / 2 - bottom[2] * 2.5, 0))
geometry.set_scale((3, 3, 3))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

screen_width = 768
screen_height = 512

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 32
rt_args.max_bounce = 4

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 256
cuda_args.num_blocks = 1024

renderer = rtx.Renderer()

camera = rtx.PerspectiveCamera(
    eye=(0, -1, 6),
    center=(0, -1, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
# renderer.render(scene, camera, render_options, render_buffer)
camera_rad = 0
# camera_rad = math.pi / 10 * 2
start = time.time()
total_iterations = 100
for n in range(total_iterations):
    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)


end = time.time()
print(total_iterations / (end - start))
