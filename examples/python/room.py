import math
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import geometry as gm
import rtx


def load_texture_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    texture = np.array(image, dtype=np.float32) / 255
    return texture


grid_size = 7
wall_height = 2
eps = 10
scene = rtx.Scene(ambient_color=(0.5, 1, 1))

# 1
geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
geometry.set_rotation((0, 0, 0))
geometry.set_position((0, 0, -grid_size / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 2
geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
geometry.set_rotation((0, -math.pi / 2, 0))
geometry.set_position((grid_size / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 3
geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
geometry.set_rotation((0, math.pi, 0))
geometry.set_position((0, 0, grid_size / 2))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# 4
geometry = rtx.PlainGeometry(grid_size + eps, wall_height)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-grid_size / 2, 0, 0))
material = rtx.LambertMaterial(0.95)
mapping = rtx.SolidColorMapping((1, 1, 1))
wall = rtx.Object(geometry, material, mapping)
scene.add(wall)

# floor
geometry = rtx.PlainGeometry(grid_size + eps, grid_size + eps)
geometry.set_rotation((-math.pi / 2, 0, 0))
geometry.set_position((0, -wall_height / 2, 0))
material = rtx.LambertMaterial(0.95)
texture = load_texture_image("/home/musyoku/sandbox/gqn-dataset-renderer/textures/pink dust.png")
uv_coordinates = np.array(
    [
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
    ], dtype=np.float32)
mapping = rtx.TextureMapping(texture, uv_coordinates)
floor = rtx.Object(geometry, material, mapping)
scene.add(floor)


# place cylinder
geometry = rtx.CylinderGeometry(1, 1)
geometry.set_position((0, -1, 2))
material = rtx.LambertMaterial(0.4)
mapping = rtx.SolidColorMapping((0, 1, 0))
cylinder = rtx.Object(geometry, material, mapping)
scene.add(cylinder)

# place cone
geometry = rtx.ConeGeometry(0.5, 1)
geometry.set_position((2, -1, 0))
material = rtx.LambertMaterial(0.4)
mapping = rtx.SolidColorMapping((1, 0, 0))
cone = rtx.Object(geometry, material, mapping)
scene.add(cone)



# Place lights
size = 50
group = rtx.ObjectGroup()
geometry = rtx.PlainGeometry(size, size)
geometry.set_rotation((0, math.pi / 2, 0))
geometry.set_position((-10, 0, 0))
material = rtx.EmissiveMaterial(3, visible=False)
mapping = rtx.SolidColorMapping((1, 1, 1))
light = rtx.Object(geometry, material, mapping)
group.add(light)

group.set_rotation((-math.pi / 3, math.pi / 2, 0))
scene.add(group)

screen_width = 64
screen_height = 64

rt_args = rtx.RayTracingArguments()
rt_args.num_rays_per_pixel = 2048
rt_args.max_bounce = 3
rt_args.next_event_estimation_enabled = False
rt_args.supersampling_enabled = True

cuda_args = rtx.CUDAKernelLaunchArguments()
cuda_args.num_threads = 64
cuda_args.num_rays_per_thread = 128

renderer = rtx.Renderer()

camera = rtx.PerspectiveCamera(
    eye=(0, 0, 6),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

view_radius = 2
rotation = 0.0
render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
total_iterations = 300
for n in range(total_iterations):
    eye = (view_radius * math.cos(rotation), 0.0,
            view_radius * math.sin(rotation))
    center = (0, 0, 0)
    camera.look_at(eye, center, up=(0, 1, 0))

    renderer.render(scene, camera, rt_args, cuda_args, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    pixels = np.uint8(pixels * 255)
    pixels = cv2.bilateralFilter(pixels, 3, 25, 25)

    plt.imshow(pixels, interpolation="none")
    plt.title("NEE (1024spp)")
    plt.pause(1e-8)

    rotation += math.pi / 36
    # group.set_rotation((0, 0, rotation))

image = Image.fromarray(np.uint8(pixels * 255))
image.save("result.png")
