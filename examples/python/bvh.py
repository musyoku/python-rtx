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
geometry.set_rotation((math.pi / 2, 0, 0))
geometry.set_position((0, box_size / 2, 0))
material = rtx.LambertMaterial(1.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
ceil = rtx.Object(geometry, material, mapping)
scene.add(ceil)

# floor
geometry = rtx.PlainGeometry(100, 100)
geometry.set_rotation((-math.pi / 2, 0, 0))
geometry.set_position((0, -box_size / 2, 0))
material = rtx.LambertMaterial(0.9)
mapping = rtx.SolidColorMapping((1, 1, 1))
floor = rtx.Object(geometry, material, mapping)
scene.add(floor)

# place bunny
faces, vertices = gm.load("../geometries/teapot")
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_scale((3, 3, 3))
geometry.set_position((-1, -1, -1))
geometry.set_rotation((-math.pi / 6, 0, 0))
material = rtx.LambertMaterial(0.8)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

faces, vertices = gm.load("../geometries/bunny")
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_scale((2.4, 2.4, 2.4))
geometry.set_position((1, -1, 1))
geometry.set_rotation((math.pi / 6, 0, 0))
material = rtx.LambertMaterial(0.7)
mapping = rtx.SolidColorMapping((1, 1, 1))
bunny = rtx.Object(geometry, material, mapping)
scene.add(bunny)

# place teapot
faces, vertices = gm.load("../geometries/teapot")
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_scale((2, 2, 2))
geometry.set_position((1, 1, -1))
geometry.set_rotation((math.pi / 6, 0, 0))
material = rtx.LayeredMaterial(rtx.LambertMaterial(0.5), rtx.LambertMaterial(0.4), rtx.LambertMaterial(0.3))
mapping = rtx.SolidColorMapping((1, 1, 1))
teapot = rtx.Object(geometry, material, mapping)
scene.add(teapot)

faces, vertices = gm.load("../geometries/bunny")
geometry = rtx.StandardGeometry(faces, vertices, 25)
geometry.set_scale((1.6, 1.6, 1.6))
geometry.set_position((-1, 1, 1))
geometry.set_rotation((-math.pi / 6, 0, 0))
material = rtx.LambertMaterial(0.6)
mapping = rtx.SolidColorMapping((1, 1, 1))
teapot = rtx.Object(geometry, material, mapping)
scene.add(teapot)

# place ball
geometry = rtx.SphereGeometry(1.0)
geometry.set_position((2, 0, 0))
material = rtx.LambertMaterial(1.0)
material = rtx.EmissiveMaterial(5.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
sphere = rtx.Object(geometry, material, mapping)
scene.add(sphere)

# place light
geometry = rtx.PlainGeometry(5.0, 5.0)
geometry.set_rotation((math.pi / 2, 0, 0))
geometry.set_position((0, box_size / 2 - 0.1, 0))
material = rtx.EmissiveMaterial(5.0)
mapping = rtx.SolidColorMapping((1, 1, 1))
rect_area_light = rtx.Object(geometry, material, mapping)
scene.add(rect_area_light)

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
