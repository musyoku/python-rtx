import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import geometry as gm
import rtx

scene = rtx.Scene((0, 0, 0))

box_width = 6
box_height = 6

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
material = rtx.EmissiveMaterial(1.0)
texture = np.array(Image.open("texture.png"), dtype=np.float32) / 255
uv_coordinates = np.array(
    [
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
    ], dtype=np.float32)
mapping = rtx.TextureMapping(texture, uv_coordinates)
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 3
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi, 0))
geometry.set_position((0, 0, box_width / 2))
material = rtx.LambertMaterial(0.95)
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 4
geometry = rtx.PlainGeometry(box_width, box_height)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-box_width / 2, 0, 0))
material = rtx.EmissiveMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
texture = np.array(Image.open("/home/musyoku/sandbox/gqn-dataset-renderer/textures/light-grey-terrazzo.png").convert("RGB"), dtype=np.float32) / 255
uv_coordinates = np.array(
    [
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
    ], dtype=np.float32)
mapping = rtx.TextureMapping(texture, uv_coordinates)
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
floor = rtx.Object(geometry, material, mapping)
scene.add(floor)

# place bunny
faces, vertices = gm.load("../geometries/bunny")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((0, -box_height / 2 - bottom[1] * 3, 0))
geometry.set_scale((3, 3, 3))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

screen_width = 768
screen_height = 512

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 128
rt_args.max_bounce = 4
rt_args.next_event_estimation_enabled = True

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 64
cuda_args.num_rays_per_thread = 64

renderer = rtx.Renderer()

camera = rtx.PerspectiveCamera(
    eye=(0, -0.5, 6),
    center=(0, -0.5, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
total_iterations = 6
for n in range(total_iterations):
    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    pixels = np.clip(render_buffer, 0, 1)

    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)
    
image = Image.fromarray(np.uint8(pixels * 255))
image.save("result.png")