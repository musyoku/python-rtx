import math
import numpy as np
import rtx
import matplotlib.pyplot as plt

scene = rtx.Scene()

screen_width = 128
screen_height = 128

render_options = rtx.RayTracingOptions()
render_options.num_rays_per_pixel = 512
render_options.path_depth = 6

renderer = rtx.RayTracingCUDARenderer()
camera = rtx.PerspectiveCamera(
    eye=(0, 0, 0),
    center=(0, 0, 0),
    up=(0, 1, 0),
    fov_rad=math.pi / 4,
    aspect_ratio=screen_width / screen_height,
    z_near=0.01,
    z_far=100)

buffer = np.zeros((screen_height, screen_width, 3), dtype="float32")
renderer.render(scene, camera, render_options, buffer)
