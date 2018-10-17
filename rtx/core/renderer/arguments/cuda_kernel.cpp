#include "cuda_kernel.h"

namespace rtx {
CUDAKernelLaunchArguments::CUDAKernelLaunchArguments()
{
    _num_threads = 256;
    _num_rays_per_thread = 256;
}
int CUDAKernelLaunchArguments::num_threads()
{
    return _num_threads;
}
void CUDAKernelLaunchArguments::set_num_threads(int num)
{
    _num_threads = num;
}
int CUDAKernelLaunchArguments::num_rays_per_thread()
{
    return _num_rays_per_thread;
}
void CUDAKernelLaunchArguments::set_num_rays_per_thread(int num)
{
    _num_rays_per_thread = num;
}
}