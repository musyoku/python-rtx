#include "ray_tracing.h"

namespace three {
RayTracingOptions::RayTracingOptions()
{
    _num_rays_per_pixel = 1;
}
int RayTracingOptions::num_rays_per_pixel()
{
    return _num_rays_per_pixel;
}
void RayTracingOptions::set_num_rays_per_pixel(int num)
{
    _num_rays_per_pixel = num;
}

int RayTracingOptions::path_depth()
{
    return _path_depth;
}
void RayTracingOptions::set_path_depth(int depth)
{
    _path_depth = depth;
}
}