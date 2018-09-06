import math
import numpy as np
import rtx
import matplotlib.pyplot as plt

scene = rtx.Scene()

box_size = 1

# 1
geometry = rtx.PlainGeometry(box_size, box_size)
material = rtx.MeshLambertMaterial((1.0, 1.0, 1.0), 1.0)
plain = rtx.Mesh(geometry, material)
scene.add(plain)

screen_width = 512
screen_height = 512

render_options = rtx.RayTracingOptions()
render_options.num_rays_per_pixel = 1
render_options.path_depth = 1

renderer = rtx.RayTracingCUDARenderer()
camera = rtx.PerspectiveCamera(
    eye=(0, 0, -1),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 3,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

render_buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
# renderer.render(scene, camera, render_options, render_buffer)
camera_rad = 0
# camera_rad = math.pi / 10
radius = 5.5
while True:
    eye = (radius * math.sin(camera_rad), 0.0, radius * math.cos(camera_rad))
    camera.look_at(eye=eye, center=(0, 0, 0), up=(0, 1, 0))

    renderer.render(scene, camera, render_options, render_buffer)
    # linear -> sRGB
    pixels = np.power(np.clip(render_buffer, 0, 1), 1.0 / 2.2)
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1e-8)

    camera_rad += math.pi / 10