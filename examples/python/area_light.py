import math
import time
import numpy as np
import rtx
import geometry as gm
import matplotlib.pyplot as plt

scene = rtx.Scene()

# floor
geometry = rtx.PlainGeometry(100, 100)
geometry.set_position((0, 0, 0))
geometry.set_rotation((-math.pi / 2, 0, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
floor = rtx.Object(geometry, material, mapping)
scene.add(floor)

# place bunny
faces, vertices = gm.load("../geometries/bunny")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((-2.25, -bottom[2], 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

# place teapot
faces, vertices = gm.load("../geometries/teapot")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((-0.75, -bottom[2] * 1.5, 0))
geometry.set_scale((1.5, 1.5, 1.5))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
teapot = rtx.Object(geometry, material, mapping)
scene.add(teapot)

# place dragon
faces, vertices = gm.load("../geometries/dragon")
bottom = np.amin(vertices, axis=0)
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_position((0.75, -bottom[2] * 1.5, 0))
geometry.set_scale((1.5, 1.5, 1.5))
geometry.set_rotation((0, -math.pi / 4, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
dragon = rtx.Object(geometry, material, mapping)
scene.add(dragon)

# place ball
geometry = rtx.SphereGeometry(0.5)
geometry.set_position((2.25, 0.5, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
sphere = rtx.Object(geometry, material, mapping)
scene.add(sphere)

# place light
geometry = rtx.PlainGeometry(2.0, 0.5)
geometry.set_rotation((math.pi / 4, 0, 0))
geometry.set_position((0, 1, -2))
material = rtx.EmissiveMaterial(5.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
rect_area_light = rtx.Object(geometry, material, mapping)
scene.add(rect_area_light)

print("#triangles", scene.num_triangles())

screen_width = 384
screen_height = 256

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 32
rt_args.max_bounce = 4

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 256
cuda_args.num_blocks = 1024

renderer = rtx.Renderer()
camera = rtx.PerspectiveCamera(
    eye=(-3, 3, 3),
    center=(0, 0.5, 0),
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

total_iterations = 100
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


