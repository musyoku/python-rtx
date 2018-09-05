#include "ray_tracing.h"

namespace rtx {
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

int RayTracingOptions::max_bounce()
{
    return _max_bounce;
}
void RayTracingOptions::set_max_bounce(int bounce)
{
    _max_bounce = bounce;
}
}