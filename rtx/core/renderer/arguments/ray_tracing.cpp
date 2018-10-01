#include "ray_tracing.h"

namespace rtx {
RayTracingArguments::RayTracingArguments()
{
    _num_rays_per_pixel = 1;
}
int RayTracingArguments::num_rays_per_pixel()
{
    return _num_rays_per_pixel;
}
void RayTracingArguments::set_num_rays_per_pixel(int num)
{
    _num_rays_per_pixel = num;
}

int RayTracingArguments::max_bounce()
{
    return _max_bounce;
}
void RayTracingArguments::set_max_bounce(int bounce)
{
    _max_bounce = bounce;
}
bool RayTracingArguments::next_event_estimation_enabled()
{
    return _next_event_estimation_enabled;
}
void RayTracingArguments::set_next_event_estimation_enabled(bool enabled)
{
    _next_event_estimation_enabled = enabled;
}
}