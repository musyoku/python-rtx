import math
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import geometry as gm
import rtx

scene = rtx.Scene()

# floor
geometry = rtx.PlainGeometry(100, 100)
geometry.set_position((0, 0, 0))
geometry.set_rotation((-math.pi / 2, 0, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
floor = rtx.Object(geometry, material, mapping)
scene.add(floor)

# place ball
geometry = rtx.SphereGeometry(0.5)
geometry.set_position((0, 0.5, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((0, 1, 1))
sphere = rtx.Object(geometry, material, mapping)
scene.add(sphere)

# place light
geometry = rtx.PlainGeometry(2.0, 2.0)
geometry.set_rotation((0, 0, math.pi / 2))
geometry.set_position((0, 1, -3))
material = rtx.EmissiveMaterial(1.0)
texture = np.array(Image.open("texture.png"), dtype=np.float32) / 255
mapping = rtx.TextureMapping(texture)
rect_area_light = rtx.Object(geometry, material, mapping)
scene.add(rect_area_light)

print("#triangles", scene.num_triangles())

screen_width = 384
screen_height = 256

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 64
rt_args.max_bounce = 4

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 256
cuda_args.num_blocks = 1024

renderer = rtx.Renderer()
camera = rtx.PerspectiveCamera(
    eye=(-3, 3, 3),
    center=(0, 0.5, -0.25),
    up=(0, 1, 0),
    fov_rad=math.pi / 4,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
# renderer.render(scene, camera, rt_args, render_buffer)
light_rad = 0
radius = 5.5
start = time.time()

total_iterations = 1000
for n in range(total_iterations):
    # if n % 10 == 0:
    #     geometry.set_rotation((math.pi / 4, 0, light_rad))
    #     light_rad += math.pi / 2

    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    print(np.amax(render_buffer))
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)
