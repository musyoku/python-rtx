import math
import numpy as np
import rtx
import matplotlib.pyplot as plt

scene = rtx.Scene()

# floor
geometry = rtx.PlainGeometry(20, 20)
material = rtx.MeshLambertMaterial((1.0, 1.0, 1.0), 0.8)
floor = rtx.Mesh(geometry, material)
floor.set_rotation((math.pi / 2, 0, 0))
floor.set_position((0, 0, 0))
scene.add(floor)

# ball
geometry = rtx.SphereGeometry(1.0)
material = rtx.MeshLambertMaterial(
    color=(1.0, 1.0, 1.0), diffuse_reflectance=0.8)
sphere = rtx.Mesh(geometry, material)
sphere.set_position((2, 1, 0))
scene.add(sphere)

# light
geometry = rtx.SphereGeometry(1.0)
material = rtx.MeshEmissiveMaterial(color=(1.0, 1.0, 1.0))
light = rtx.Mesh(geometry, material)
light.set_position((-1, 3, -1))
scene.add(light)

screen_width = 128
screen_height = 128

render_options = rtx.RayTracingOptions()
render_options.num_rays_per_pixel = 512
render_options.path_depth = 6

renderer = rtx.RayTracingCPURenderer()
camera = rtx.PerspectiveCamera(
    eye=(0, 0, 0),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 4,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")

camera_rad = 0
while True:
    camera.look_at(eye=(3, 3, 3), center=(0, 0, 0), up=(0, 1, 0))

    renderer.render(scene, camera, render_options, buffer)
    # linear -> sRGB
    pixels = np.power(buffer, 1.0 / 2.2)
    # display
    plt.imshow(pixels, interpolation="none")
    plt.pause(1.0 / 60.0)
