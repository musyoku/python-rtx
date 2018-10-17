import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import geometry as gm
import rtx

scene = rtx.Scene(ambient_color=(0, 0, 0))

box_width = 6
box_height = 5

# 1
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, 0, 0))
geometry.set_position((0, 0, -box_width / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 2
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_width / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 3
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi, 0))
geometry.set_position((0, 0, box_width / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 4
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-box_width / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# ceil
geometry = rtx.PlainGeometry(box_width, box_width)
geometry.set_rotation((math.pi / 2, 0, 0))
geometry.set_position((0, box_height / 2, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# floor
geometry = rtx.PlainGeometry(box_width, box_width)
geometry.set_rotation((-math.pi / 2, 0, 0))
geometry.set_position((0, -box_height / 2, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# light
geometry = rtx.PlainGeometry(box_width / 2, box_height / 2)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((0.01 - box_width / 2, -box_height / 4, 0))
material = rtx.EmissiveMaterial(5.0, visible=False)
mapping = rtx.SolidColorMapping((1, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

geometry = rtx.PlainGeometry(box_width / 2, box_width / 2)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((box_width / 2 - 0.01, -box_height / 4, 0))
material = rtx.EmissiveMaterial(5.0)
mapping = rtx.SolidColorMapping((0, 1, 1))
light = rtx.Object(geometry, material, mapping)
scene.add(light)

# place bunny
faces, vertices = gm.load("../geometries/bunny")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((0, -box_height / 2 - (bottom[1] + 0.01) * 3, 0))
geometry.set_scale((3, 3, 3))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

screen_width = 96
screen_height = 64

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 128
rt_args.max_bounce = 4
rt_args.next_event_estimation_enabled = True

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 64
cuda_args.num_rays_per_thread = 128

renderer = rtx.Renderer()

camera = rtx.PerspectiveCamera(
    eye=(0, -0.5, 12),
    center=(0, -0.5, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
total_iterations = 30
for n in range(total_iterations):
    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)

    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)

image = Image.fromarray(np.uint8(pixels * 255))
image.save("result.png")
