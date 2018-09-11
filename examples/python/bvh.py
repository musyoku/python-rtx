import math
import time
import numpy as np
import rtx
import geometry as gm
import matplotlib.pyplot as plt

scene = rtx.Scene()

box_size = 6

# ceil
geometry = rtx.PlainGeometry(100, 100)
material = rtx.LambertMaterial((1.0, 1.0, 1.0), 1.0)
ceil = rtx.Mesh(geometry, material)
ceil.set_rotation((math.pi / 2, 0, 0))
ceil.set_position((0, box_size / 2, 0))
scene.add(ceil)

# floor
geometry = rtx.PlainGeometry(100, 100)
material = rtx.LambertMaterial((1.0, 1.0, 1.0), 1.0)
floor = rtx.Mesh(geometry, material)
floor.set_rotation((-math.pi / 2, 0, 0))
floor.set_position((0, -box_size / 2, 0))
scene.add(floor)

# place bunny
faces, vertices = gm.load("../geometries/teapot")
geometry = rtx.StandardGeometry(faces, vertices, 25)
material = rtx.LambertMaterial(color=(1.0, 1.0, 1.0), diffuse_reflectance=1.0)
bunny = rtx.Mesh(geometry, material)
bunny.set_scale((3, 3, 3))
bunny.set_position((-1, -1, -1))
bunny.set_rotation((-math.pi / 6, 0, 0))
scene.add(bunny)

faces, vertices = gm.load("../geometries/bunny")
geometry = rtx.StandardGeometry(faces, vertices, 25)
material = rtx.LambertMaterial(color=(1.0, 1.0, 1.0), diffuse_reflectance=1.0)
bunny = rtx.Mesh(geometry, material)
bunny.set_scale((2.4, 2.4, 2.4))
bunny.set_position((1, -1, 1))
bunny.set_rotation((math.pi / 6, 0, 0))
scene.add(bunny)

# place teapot
faces, vertices = gm.load("../geometries/teapot")
geometry = rtx.StandardGeometry(faces, vertices, 25)
material = rtx.LambertMaterial(color=(1.0, 1.0, 1.0), diffuse_reflectance=1.0)
teapot = rtx.Mesh(geometry, material)
teapot.set_scale((2, 2, 2))
teapot.set_position((1, 1, -1))
teapot.set_rotation((math.pi / 6, 0, 0))
scene.add(teapot)

faces, vertices = gm.load("../geometries/bunny")
geometry = rtx.StandardGeometry(faces, vertices, 25)
material = rtx.LambertMaterial(color=(1.0, 1.0, 1.0), diffuse_reflectance=1.0)
teapot = rtx.Mesh(geometry, material)
teapot.set_scale((1.6, 1.6, 1.6))
teapot.set_position((-1, 1, 1))
teapot.set_rotation((-math.pi / 6, 0, 0))
scene.add(teapot)

# place light
light = rtx.RectAreaLight(5.0, 5.0, brightness=10, color=(1, 0, 0))
light.set_rotation((math.pi / 2, 0, 0))
light.set_position((0, box_size / 2 - 0.1, 0))
scene.add(light)

print("#triangles", scene.num_triangles())

screen_width = 512
screen_height = 512

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 32
rt_args.max_bounce = 4

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 256
cuda_args.num_blocks = 1024

renderer = rtx.Renderer()
camera = rtx.PerspectiveCamera(
    eye=(0, 0, -1),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
# renderer.render(scene, camera, rt_args, render_buffer)
camera_rad = 0
# camera_rad = math.pi / 10 * 2
radius = 5.5
start = time.time()

eye = (radius * math.sin(camera_rad), 0.0, radius * math.cos(camera_rad))
camera.look_at(eye=eye, center=(0, 0, 0), up=(0, 1, 0))

total_iterations = 100
for n in range(total_iterations):
    eye = (radius * math.sin(camera_rad), 0.0, radius * math.cos(camera_rad))
    # camera.look_at(eye=eye, center=(0, 0, 0), up=(0, 1, 0))

    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)

    # camera_rad += math.pi / 10

end = time.time()
print(total_iterations / (end - start))
